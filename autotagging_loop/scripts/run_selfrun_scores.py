"""Self-run the 9 missing leaderboard cells on FULL benchmark datasets via OpenRouter.

These scores are flagged `basis: self_run` — they are NOT official-source cells and
must stay separate from data/leaderboard_scores.json until a human decides to merge.

Modes:
  --selftest   offline grader self-checks (no network, no spend)
  --dry-run    load datasets, print exact item counts + cost estimate (no API calls)
  --run        the paid run (writes data/self_run_score_backfill.json)

A JSONL response cache (results/selfrun_cache.jsonl) makes --run resumable so a crash
near the end does not re-bill completed calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = ROOT / "results" / "selfrun_cache.jsonl"
OUT_PATH = ROOT / "data" / "self_run_score_backfill.json"

# ── what to run ──────────────────────────────────────────────────────────────
# (benchmark leaderboard-name, model leaderboard-name)
CELLS: list[tuple[str, str]] = [
    ("ARC Challenge", "Claude-Sonnet-4"),
    ("BBH", "Claude-Sonnet-4"),
    ("MBPP", "Claude-Sonnet-4"),
    ("BBH", "GPT-4o"),
    ("HMMT Feb 2025", "GPT-4o"),
    ("ARC Challenge", "Qwen2.5-72B"),
    ("HMMT Feb 2025", "Qwen2.5-72B"),
    ("Drop", "GPT-oss-20b"),
    ("MBPP", "GPT-oss-20b"),
]

# NOTE: OPENROUTER_BASE_URL in .env is a LiteLLM proxy that prefixes every model
# with "openrouter/". Cells whose slug the endpoint does not serve are auto-skipped.
# Known official/matrix cells used to verify the harness reproduces sane scores.
VALIDATION: dict[tuple[str, str], float] = {
    ("Drop", "GPT-4o"): 0.834,
    ("ARC Challenge", "GPT-4o"): 0.967,
    ("BBH", "GPT-oss-20b"): 0.841,
    ("HMMT Feb 2025", "GPT-oss-20b"): 0.233,
    ("MBPP", "GPT-4o"): 0.862,
    ("BBH", "DeepSeek-v3"): 0.875,   # non-reasoning BBH pipeline check
}

MODEL_SLUGS: dict[str, str] = {
    "Claude-Sonnet-4": "openrouter/anthropic/claude-sonnet-4",
    "GPT-4o": "openrouter/openai/gpt-4o",
    "Qwen2.5-72B": "openrouter/qwen/qwen-2.5-72b-instruct",  # not deployed on this proxy
    "GPT-oss-20b": "openrouter/openai/gpt-oss-20b",
    "DeepSeek-v3": "openrouter/deepseek/deepseek-chat",  # validation only (official BBH 0.875)
}

# OpenRouter price estimate, USD per 1M tokens (input, output). For dry-run only;
# real billing is whatever OpenRouter charges. ponytail: hardcoded, refresh if slugs change.
PRICING: dict[str, tuple[float, float]] = {
    "openrouter/anthropic/claude-sonnet-4": (3.0, 15.0),
    "openrouter/openai/gpt-4o": (2.5, 10.0),
    "openrouter/qwen/qwen-2.5-72b-instruct": (0.13, 0.40),
    "openrouter/openai/gpt-oss-20b": (0.05, 0.20),
    "openrouter/deepseek/deepseek-chat": (0.27, 1.10),
}

BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]

# max output tokens per benchmark (caps cost; math/code need room for CoT)
MAX_TOKENS = {"ARC Challenge": 512, "BBH": 512, "Drop": 256,
              "HMMT Feb 2025": 2048, "MBPP": 1024,
              # item-level pool (Phase 0)
              "HumanEval": 1024, "MATH-500": 2048, "GSM8K": 1024,
              "MMLU-Pro": 512, "GPQA-Diamond": 512}

# Reasoning models spend the token budget on hidden reasoning before emitting the
# final answer; at the normal caps `content` comes back empty and auto-scores 0.
# Give them a flat high ceiling. ponytail: flat 4096, raise if BBH-gptoss still truncates.
REASONING_MAX_TOKENS = {"GPT-oss-20b": 4096}


def _max_tokens(bench: str, model: str) -> int:
    return REASONING_MAX_TOKENS.get(model, MAX_TOKENS[bench])


# ── dataset loaders → list[dict] with at least {id, prompt, gold} ─────────────
def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def load_arc() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for i, r in enumerate(ds):
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        lines = "\n".join(f"({lab}) {t}" for lab, t in zip(labels, texts))
        prompt = (
            f"{r['question']}\n{lines}\n\n"
            "Answer with the single letter of the correct option after 'Answer:'."
        )
        out.append({"id": r.get("id", f"arc{i}"), "prompt": prompt,
                    "gold": r["answerKey"], "labels": labels})
    return out


def load_bbh() -> list[dict]:
    from datasets import load_dataset
    out = []
    for task in BBH_TASKS:
        ds = load_dataset("lukaemon/bbh", task, split="test")
        for i, r in enumerate(ds):
            prompt = (
                f"{r['input']}\n\n"
                "Think briefly, then give the final answer after 'Answer:'."
            )
            out.append({"id": f"{task}/{i}", "prompt": prompt,
                        "gold": str(r["target"]).strip()})
    return out


def _drop_gold(ans: dict) -> list[str]:
    # ponytail: primary `answer` only (the official headline gold). Skip validated_answers
    # parallel-array maxing — add it if we need exact official DROP-F1 parity.
    cands: list[str] = []
    spans = ans.get("spans") or []
    if spans:
        cands.append(" ".join(spans))
    num = ans.get("number")
    if num not in (None, ""):
        cands.append(str(num))
    d = ans.get("date") or {}
    date_str = " ".join(str(d.get(k, "")) for k in ("day", "month", "year")).strip()
    if date_str:
        cands.append(date_str)
    return cands


def _drop_validated_gold(va: dict) -> list[str]:
    # validated_answers is parallel arrays; reconstruct each annotator's answer.
    # Official DROP maxes F1 over primary answer + these, so ignoring them deflates scores.
    spans_list = va.get("spans") or []
    nums = va.get("number") or []
    dates = va.get("date") or []
    cands: list[str] = []
    for i in range(max(len(spans_list), len(nums), len(dates))):
        cands += _drop_gold({
            "spans": spans_list[i] if i < len(spans_list) else [],
            "number": nums[i] if i < len(nums) else "",
            "date": dates[i] if i < len(dates) else {},
        })
    return cands


def load_drop() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/drop", split="validation")
    out = []
    for i, r in enumerate(ds):
        gold = _drop_gold(r.get("answer") or {})
        gold += _drop_validated_gold(r.get("validated_answers") or {})
        gold = list(dict.fromkeys(gold))  # dedupe, keep order
        if not gold:
            continue
        prompt = (
            f"Passage: {r['passage']}\n\nQuestion: {r['question']}\n\n"
            "Answer concisely after 'Answer:'."
        )
        out.append({"id": r.get("query_id", f"drop{i}"), "prompt": prompt,
                    "gold": gold})
    return out


def load_hmmt() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
    out = []
    for i, r in enumerate(ds):
        prob = r.get("problem") or r.get("question")
        ans = r.get("answer")
        prompt = (
            f"{prob}\n\n"
            "Solve it. Put the final answer in \\boxed{} at the end."
        )
        out.append({"id": str(r.get("problem_idx", i)), "prompt": prompt,
                    "gold": str(ans).strip()})
    return out


def load_mbpp() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    out = []
    for r in ds:
        tests = r["test_list"]
        prompt = (
            f"{r['text']}\n\n"
            f"Your function must satisfy:\n" + "\n".join(tests) + "\n\n"
            "Return only the Python function in a ```python code block."
        )
        out.append({"id": str(r["task_id"]), "prompt": prompt, "gold": "",
                    "tests": tests, "setup": r.get("test_setup_code", "")})
    return out


def load_humaneval() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for i, r in enumerate(ds):
        prompt = (
            f"{r['prompt']}\n\n"
            "Complete the function. Return only Python code in a ```python code block."
        )
        out.append({"id": str(r.get("task_id", f"humaneval/{i}")), "prompt": prompt,
                    "gold": r.get("canonical_solution", ""), "test": r["test"],
                    "entry_point": r["entry_point"], "base_prompt": r["prompt"],
                    "group": "programming"})
    return out


def load_math500() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, r in enumerate(ds):
        prompt = (
            f"{r['problem']}\n\n"
            "Solve it. Put the final answer in \\boxed{} at the end."
        )
        subject = str(r.get("subject", "unknown")).lower().replace(" ", "_")
        level = str(r.get("level", "unknown")).lower().replace(" ", "_")
        out.append({"id": str(r.get("unique_id", f"math-500/{i}")), "prompt": prompt,
                    "gold": str(r["answer"]).strip(), "subject": subject,
                    "level": level, "group": f"{subject}:{level}"})
    return out


def _strip_gsm8k_answer(raw: str) -> str:
    m = re.search(r"####\s*(.+?)\s*$", raw or "", re.MULTILINE)
    return m.group(1).strip() if m else (raw or "").strip()


def load_gsm8k() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(ds):
        prompt = (
            f"{r['question']}\n\n"
            "Solve it. Put only the final number after 'Answer:'."
        )
        out.append({"id": str(r.get("id", f"gsm8k/{i}")), "prompt": prompt,
                    "gold": _strip_gsm8k_answer(r["answer"]),
                    "group": "math_word_problem"})
    return out


def _answer_letter(answer: object, options: list[str]) -> str:
    labels = _letters(len(options))
    if isinstance(answer, int) and 0 <= answer < len(labels):
        return labels[answer]
    raw = str(answer).strip()
    if len(raw) == 1 and raw.upper() in labels:
        return raw.upper()
    if raw.isdigit() and 0 <= int(raw) < len(labels):
        return labels[int(raw)]
    for label, option in zip(labels, options):
        if _norm(option) == _norm(raw):
            return label
    return raw.upper()


def load_mmlu_pro() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for i, r in enumerate(ds):
        options = list(r["options"])
        labels = _letters(len(options))
        lines = "\n".join(f"({lab}) {opt}" for lab, opt in zip(labels, options))
        prompt = (
            f"{r['question']}\n{lines}\n\n"
            "Answer with the single letter of the correct option after 'Answer:'."
        )
        category = str(r.get("category", "unknown")).lower().replace(" ", "_")
        out.append({"id": str(r.get("question_id", f"mmlu-pro/{i}")),
                    "prompt": prompt, "gold": _answer_letter(r["answer"], options),
                    "labels": labels, "category": category, "group": category})
    return out


def load_gpqa_diamond() -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception:  # noqa: BLE001 - gated HF dataset; opt-in loader skips cleanly
        return []
    out = []
    for i, r in enumerate(ds):
        question = r.get("Question") or r.get("question")
        correct = r.get("Correct Answer") or r.get("correct_answer")
        wrongs = [r.get(f"Incorrect Answer {j}") or r.get(f"incorrect_answer_{j}")
                  for j in range(1, 4)]
        pairs = [(str(correct), True)] + [(str(w), False) for w in wrongs if w]
        random.Random(i).shuffle(pairs)
        labels = _letters(len(pairs))
        lines = "\n".join(f"({lab}) {opt}" for lab, (opt, _) in zip(labels, pairs))
        gold = next(lab for lab, (_, ok) in zip(labels, pairs) if ok)
        prompt = (
            f"{question}\n{lines}\n\n"
            "Answer with the single letter of the correct option after 'Answer:'."
        )
        out.append({"id": str(r.get("Record ID", f"gpqa-diamond/{i}")),
                    "prompt": prompt, "gold": gold, "labels": labels,
                    "group": "science_reasoning"})
    return out


LOADERS = {"ARC Challenge": load_arc, "BBH": load_bbh, "Drop": load_drop,
           "HMMT Feb 2025": load_hmmt, "MBPP": load_mbpp,
           "HumanEval": load_humaneval, "MATH-500": load_math500,
           "GSM8K": load_gsm8k, "MMLU-Pro": load_mmlu_pro,
           "GPQA-Diamond": load_gpqa_diamond}


# ── graders ──────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _final_after_answer(text: str) -> str:
    m = list(re.finditer(r"answer\s*[:\-]\s*(.+)", text, re.IGNORECASE))
    return m[-1].group(1).strip() if m else text.strip().splitlines()[-1].strip() if text.strip() else ""


def grade_arc(text: str, gold: str, labels: list[str]) -> float:
    tail = _final_after_answer(text)
    m = re.search(r"[A-Z1-9]", tail.upper())
    pred = m.group(0) if m else ""
    return 1.0 if pred == str(gold).strip().upper() else 0.0


def grade_bbh(text: str, gold: str) -> float:
    pred = _final_after_answer(text)
    g = gold.strip()
    # multiple-choice targets look like "(A)"
    mc = re.fullmatch(r"\(([A-Z])\)", g)
    if mc:
        pm = re.search(r"\(?([A-Z])\)?", pred.upper())
        return 1.0 if pm and pm.group(1) == mc.group(1) else 0.0
    pred = pred.rstrip(".").strip()
    ng, npred = _norm(g), _norm(pred)
    if ng == npred:
        return 1.0
    # models append explanation after a short answer ("No, Ka does not..." for gold "No");
    # credit a leading word-boundary match for short (yes/no/single-word/number) golds.
    if len(ng.split()) <= 2 and re.match(re.escape(ng) + r"\b", npred):
        return 1.0
    # bracket-sequence answers (dyck_languages) differ only by spacing ("] ]" vs "]]")
    if re.search(r"[\[\](){}<>]", ng) and re.sub(r"\s+", "", ng) == re.sub(r"\s+", "", npred):
        return 1.0
    return 0.0


_ARTICLES = {"a", "an", "the"}


def _drop_norm_tokens(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return [t for t in s.split() if t]


def _f1(pred: str, gold: str) -> float:
    p, g = _drop_norm_tokens(pred), _drop_norm_tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    gc = list(g)
    for t in p:
        if t in gc:
            gc.remove(t)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


_WORD2NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
             "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
             "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
             "eighteen": "18", "nineteen": "19", "twenty": "20"}


def _word2num(s: str) -> str:
    return re.sub(r"\b(" + "|".join(_WORD2NUM) + r")\b",
                  lambda m: _WORD2NUM[m.group(1).lower()], s, flags=re.IGNORECASE)


def grade_drop(text: str, gold_spans: list[str]) -> float:
    pred = _final_after_answer(text)
    best = 0.0
    for g in gold_spans:
        # DROP numeric answers: compare the number, ignoring surrounding words ("Two empires" == 2)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", g.strip()):
            if _num(_word2num(pred)) == g.strip():
                return 1.0
        best = max(best, _f1(pred, g))
    return best


def grade_hmmt(text: str, gold: str) -> float:
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    pred = m[-1].strip() if m else _final_after_answer(text)

    def canon(x: str) -> str:
        x = x.replace("\\left", "").replace("\\right", "").replace("\\!", "")
        x = x.replace("\\,", "").replace("$", "").replace(" ", "")
        x = x.replace("\\dfrac", "\\frac").rstrip(".")
        return x.lower()

    if canon(pred) == canon(gold):
        return 1.0
    try:
        return 1.0 if abs(float(_num(pred)) - float(_num(gold))) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return 0.0


def _num(s: str) -> str:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else ""


def _last_num(s: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else ""


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def grade_mbpp(text: str, tests: list[str], setup: str) -> float:
    # ponytail: subprocess + timeout, not a hardened sandbox. MBPP test code is trusted
    # benchmark data and the model output runs on the user's own box; swap in nsjail/docker
    # if you ever grade untrusted models at scale.
    code = _extract_code(text)
    script = (setup or "") + "\n" + code + "\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "m.py"
        f.write_text(script, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True,
                               timeout=10, cwd=d)
            return 1.0 if r.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, OSError):
            return 0.0


def grade_humaneval(text: str, item: dict) -> float:
    code = _extract_code(text)
    if "def " not in code:
        code = f"{item.get('base_prompt', '')}\n{code}"
    script = f"{code}\n{item['test']}\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "h.py"
        f.write_text(script, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True,
                               timeout=10, cwd=d)
            return 1.0 if r.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, OSError):
            return 0.0


def grade_math500(text: str, gold: str) -> float:
    return grade_hmmt(text, gold)


def grade_gsm8k(text: str, gold: str) -> float:
    pred = _last_num(_final_after_answer(text))
    return 1.0 if pred and pred == _last_num(gold) else 0.0


def grade_mmlu_pro(text: str, gold: str, labels: list[str]) -> float:
    return grade_arc(text, gold, labels)


def grade(bench: str, text: str, item: dict) -> float:
    if bench == "ARC Challenge":
        return grade_arc(text, item["gold"], item["labels"])
    if bench == "BBH":
        return grade_bbh(text, item["gold"])
    if bench == "Drop":
        return grade_drop(text, item["gold"])
    if bench == "HMMT Feb 2025":
        return grade_hmmt(text, item["gold"])
    if bench == "MBPP":
        return grade_mbpp(text, item["tests"], item["setup"])
    if bench == "HumanEval":
        return grade_humaneval(text, item)
    if bench == "MATH-500":
        return grade_math500(text, item["gold"])
    if bench == "GSM8K":
        return grade_gsm8k(text, item["gold"])
    if bench in {"MMLU-Pro", "GPQA-Diamond"}:
        return grade_mmlu_pro(text, item["gold"], item["labels"])
    raise ValueError(bench)


# ── offline self-check ───────────────────────────────────────────────────────
def selftest() -> None:
    assert grade_arc("reasoning\nAnswer: B", "B", ["A", "B", "C", "D"]) == 1.0
    assert grade_arc("Answer: (C)", "B", ["A", "B", "C", "D"]) == 0.0
    assert grade_bbh("Answer: (A)", "(A)") == 1.0
    assert grade_bbh("so... Answer: valid", "valid") == 1.0
    assert grade_bbh("Answer: False", "True") == 0.0
    assert grade_bbh("Answer: No, Ka does not tell the truth.", "No") == 1.0  # trailing text
    assert grade_bbh("Answer: Yes, the sentence is plausible.", "yes") == 1.0
    assert grade_bbh("Answer: No, not plausible", "yes") == 0.0   # wrong yes/no stays 0
    assert grade_bbh("Answer: Not sure", "no") == 0.0             # word boundary, not substring
    assert grade_bbh("Answer: ]]", "] ]") == 1.0                  # bracket spacing (dyck)
    assert grade_bbh("Answer: 5 objects", "5") == 1.0             # number + noun
    assert grade_drop("Answer: 5 yards", ["5"]) == 1.0          # numeric match
    assert grade_drop("Answer: John Smith", ["John Smith"]) == 1.0
    assert grade_drop("Answer: Two empires.", ["2"]) == 1.0     # number-word == digit
    assert grade_drop("Answer: 5 touchdowns", ["4"]) == 0.0     # wrong number stays 0
    assert grade_hmmt("thus \\boxed{42}", "42") == 1.0
    assert grade_hmmt("Answer: 3.0", "3") == 1.0
    assert grade_hmmt("\\boxed{7}", "8") == 0.0
    ok = "```python\ndef add(a,b):\n    return a+b\n```"
    assert grade_mbpp(ok, ["assert add(1,2)==3"], "") == 1.0
    bad = "```python\ndef add(a,b):\n    return a-b\n```"
    assert grade_mbpp(bad, ["assert add(1,2)==3"], "") == 0.0
    assert grade_mbpp("```python\nwhile True: pass\n```", ["assert True"], "") == 0.0  # timeout
    he = {"base_prompt": "def add(a, b):\n", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
          "entry_point": "add"}
    assert grade_humaneval("```python\ndef add(a, b):\n    return a + b\n```", he) == 1.0
    assert grade_humaneval("```python\ndef add(a, b):\n    return a - b\n```", he) == 0.0
    assert grade_math500("therefore \\boxed{42}", "42") == 1.0
    assert grade_math500("Answer: 41", "42") == 0.0
    assert grade_gsm8k("work...\nAnswer: 1,234", "1234") == 1.0
    assert grade_gsm8k("Answer: 7", "8") == 0.0
    assert grade_mmlu_pro("Answer: B", "B", ["A", "B", "C", "D"]) == 1.0
    assert grade_mmlu_pro("Answer: D", "B", ["A", "B", "C", "D"]) == 0.0
    print("selftest OK")


# ── cost estimate (dry-run) ──────────────────────────────────────────────────
def estimate(items_by_bench: dict[str, list[dict]], cells: list[tuple[str, str]] = CELLS) -> None:
    total = 0.0
    print(f"{'cell':<32}{'items':>8}{'~$':>10}")
    print("-" * 50)
    for bench, model in cells:
        items = items_by_bench[bench]
        slug = MODEL_SLUGS[model]
        pin, pout = PRICING[slug]
        in_tok = sum(len(it["prompt"]) for it in items) / 4.0
        out_tok = len(items) * MAX_TOKENS[bench] * 0.5  # assume half of cap used
        cost = (in_tok * pin + out_tok * pout) / 1e6
        total += cost
        print(f"{bench+' / '+model:<32}{len(items):>8}{cost:>10.2f}")
    print("-" * 50)
    n = sum(len(items_by_bench[b]) for b, _ in cells)
    print(f"{'TOTAL':<32}{n:>8}{total:>10.2f}")
    print("\n(estimate only — real billing is whatever OpenRouter charges; "
          "output-token guess = 50% of the per-benchmark cap.)")


# ── paid run ─────────────────────────────────────────────────────────────────
def _client():
    from openai import AsyncOpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    return AsyncOpenAI(api_key=key, base_url=base)


def _load_cache(path: Path) -> dict[tuple[str, str, str], str]:
    cache: dict[tuple[str, str, str], str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cache[(r["slug"], r["bench"], r["id"])] = r["text"]
    return cache


def _manifest_ids(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: dict[str, set[str]] = {}
    for row in data.get("items", []):
        ids.setdefault(row["benchmark"], set()).add(str(row["item_id"]))
    return ids


def _filter_manifest(items_by_bench: dict[str, list[dict]], path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return items_by_bench
    wanted = _manifest_ids(path)
    return {
        bench: [item for item in items if str(item["id"]) in wanted.get(bench, set())]
        for bench, items in items_by_bench.items()
    }


async def run(max_items: int | None, concurrency: int, cells: list[tuple[str, str]],
              expected: dict[tuple[str, str], float] | None = None,
              cache_path: Path = CACHE_PATH,
              items_manifest: Path | None = None) -> None:
    benches = sorted({b for b, _ in cells})
    items_by_bench = {b: LOADERS[b]() for b in benches}
    items_by_bench = _filter_manifest(items_by_bench, items_manifest)
    if max_items:
        items_by_bench = {b: v[:max_items] for b, v in items_by_bench.items()}

    client = _client()
    served = {m.id for m in (await client.models.list()).data}
    wildcards = tuple(s[:-1] for s in served if s.endswith("*"))  # e.g. "openrouter/"
    skipped = [(b, m) for (b, m) in cells
               if MODEL_SLUGS[m] not in served and not MODEL_SLUGS[m].startswith(wildcards)]
    if skipped:
        for b, m in skipped:
            print(f"  SKIP {b} / {m}: '{MODEL_SLUGS[m]}' not served by endpoint")
        cells = [c for c in cells if c not in skipped]
    cache = _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_f = cache_path.open("a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}
    results: dict[tuple[str, str], list[float]] = {c: [] for c in cells}
    total_calls = sum(len(items_by_bench[b]) for b, _ in cells)

    async def one(bench: str, model: str, item: dict) -> None:
        slug = MODEL_SLUGS[model]
        key = (slug, bench, str(item["id"]))
        text = cache.get(key)
        if text is None:
            async with sem:
                for attempt in range(4):
                    try:
                        resp = await client.chat.completions.create(
                            model=slug,
                            messages=[{"role": "user", "content": item["prompt"]}],
                            max_tokens=_max_tokens(bench, model),
                            temperature=0.0,
                        )
                        text = resp.choices[0].message.content or ""
                        break
                    except Exception as e:  # noqa: BLE001 — external API, retry
                        if attempt == 3:
                            text = ""
                            print(f"  fail {key}: {e.__class__.__name__}")
                        else:
                            await asyncio.sleep(2 ** attempt)
            cache_f.write(json.dumps({"slug": slug, "bench": bench,
                                      "id": str(item["id"]), "text": text}) + "\n")
            cache_f.flush()
        results[(bench, model)].append(grade(bench, text, item))
        done["n"] += 1
        if done["n"] % 50 == 0:
            print(f"  {done['n']}/{total_calls}")

    t0 = time.time()
    await asyncio.gather(*[one(b, m, it) for (b, m) in cells
                           for it in items_by_bench[b]])
    cache_f.close()

    scores = {c: (sum(v) / len(v) if v else 0.0) for c, v in results.items()}

    if expected is not None:
        print("\n=== VALIDATION (self-run harness vs official) ===")
        print(f"  {'cell':<30}{'self':>8}{'official':>10}{'Δ':>8}")
        for (bench, model) in cells:
            s = scores[(bench, model)]
            off = expected[(bench, model)]
            print(f"  {bench+'/'+model:<30}{s:>8.3f}{off:>10.3f}{s-off:>+8.3f}"
                  f"   (n={len(results[(bench, model)])})")
        print(f"\n({time.time()-t0:.0f}s)")
        return

    out = {"_meta": {"basis": "self_run", "harness": "scripts/run_selfrun_scores.py",
                     "date": time.strftime("%Y-%m-%d"),
                     "note": "Zero-shot via OpenRouter; NOT official-source. "
                             "Do not merge into leaderboard_scores.json without flagging.",
                     "max_items": max_items},
           "scores": []}
    print("\n=== RESULTS ===")
    for (bench, model) in cells:
        score = scores[(bench, model)]
        out["scores"].append({"benchmark": bench, "model": model,
                              "score": round(score, 4), "metric": "f1" if bench == "Drop" else "accuracy",
                              "n_items": len(results[(bench, model)]), "model_slug": MODEL_SLUGS[model],
                              "basis": "self_run"})
        print(f"  {bench:<16} {model:<16} {score:.4f}  (n={len(results[(bench, model)])})")
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}  ({time.time()-t0:.0f}s)")


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="reproduce known-official cells on a subset and compare")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--only", help="substring filter on 'bench/model'")
    ap.add_argument("--cache", default=None, help="alternate cache path (parallel safety)")
    ap.add_argument("--items-manifest", default=None,
                    help="optional frozen item pool manifest; filters loaded items by benchmark/id")
    args = ap.parse_args()

    cells = CELLS
    if args.only:
        cells = [c for c in CELLS if args.only.lower() in f"{c[0]}/{c[1]}".lower()]
    cache_path = Path(args.cache) if args.cache else CACHE_PATH

    if args.selftest:
        selftest()
        return
    if args.dry_run:
        benches = sorted({b for b, _ in cells})
        items = {b: LOADERS[b]() for b in benches}
        items = _filter_manifest(items, Path(args.items_manifest) if args.items_manifest else None)
        if args.max_items:
            items = {b: v[:args.max_items] for b, v in items.items()}
        estimate(items, cells)
        return
    if args.validate:
        vcells = list(VALIDATION.keys())
        if args.only:
            vcells = [c for c in vcells if args.only.lower() in f"{c[0]}/{c[1]}".lower()]
        cap = args.max_items or 150
        asyncio.run(run(cap, args.concurrency, vcells, expected=VALIDATION,
                        cache_path=cache_path,
                        items_manifest=Path(args.items_manifest) if args.items_manifest else None))
        return
    if args.run:
        asyncio.run(run(args.max_items, args.concurrency, cells, cache_path=cache_path,
                        items_manifest=Path(args.items_manifest) if args.items_manifest else None))
        return
    ap.error("pick one of --selftest / --dry-run / --run / --validate")


if __name__ == "__main__":
    main()

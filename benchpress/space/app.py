"""BenchPress Composer — preview eval-set compositions; no login, no user token.

Preview is the primary path and needs no credentials. Optional live publishing
uses a server-side fine-grained token (``HF_TOKEN`` Space secret) restricted to
the service org (``BENCHPRESS_ORG``); published demo repos are public and use a
``demo-`` prefix for cleanup.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import gradio as gr

from benchpress_hub import build_manifest, push_composition
from benchpress_hub.composition import ABILITIES_PATH, DATASET_MAP_PATH, LEADERBOARD_PATH
from benchpress_hub.publishing import (
    CREDENTIAL_HELP,
    build_demo_repo_id,
    resolve_publisher,
    sanitize_repo_name,
    scrub_secrets,
    verify_write_access,
)
from benchpress_hub.recommend import rank_models, relevance_ranking
from autotagging_loop.runner.hf_sampling import load_dataset_map

BENCHMARKS = sorted(spec.benchmark for spec in load_dataset_map(DATASET_MAP_PATH).values())

TAG_MAP_PATH = Path(__file__).parent / "tag_map.json"
try:
    with open(TAG_MAP_PATH, encoding="utf-8") as fh:
        TAG_MAP = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    TAG_MAP = None

if TAG_MAP:
    ABILITY_CHOICES = [(tag["name"], tag["id"]) for tag in TAG_MAP["vocab"]]
else:
    try:
        with open(ABILITIES_PATH, encoding="utf-8") as fh:
            ABILITY_CHOICES = [(a["name"], a["id"]) for a in json.load(fh)]
    except FileNotFoundError:
        ABILITY_CHOICES = []

try:
    with open(LEADERBOARD_PATH, encoding="utf-8") as fh:
        LEADERBOARD = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    LEADERBOARD = {}

EXAMPLES_PATH = Path(__file__).parent / "examples.json"
try:
    with open(EXAMPLES_PATH, encoding="utf-8") as fh:
        EXAMPLES = json.load(fh)
except FileNotFoundError:
    EXAMPLES = []

HF_TOKEN = os.environ.get("HF_TOKEN")
BENCHPRESS_ORG = os.environ.get("BENCHPRESS_ORG")
# HF Spaces sets SPACE_ID; a cloned local checkout does not. On the hosted Space
# we require server-side secrets and publish to the demo org. Run locally, you
# publish with your own token / `hf auth login` — and without BENCHPRESS_ORG, to
# your own namespace (resolve_publisher falls back to whoami).
IS_SPACE = bool(os.environ.get("SPACE_ID"))
PUBLISH_READY = (bool(HF_TOKEN) and bool(BENCHPRESS_ORG)) if IS_SPACE else True


def _local_auth_status() -> tuple[str, str]:
    """Probe the local user's token once at startup for the publish banner.

    Returns ``(kind, markdown)`` with kind in {"ok", "readonly", "none"}.
    Best-effort — any failure degrades to "none". Only meaningful off-Space,
    where publishing uses the user's own token / ``hf auth login``.
    """
    try:
        api, namespace = resolve_publisher(token=HF_TOKEN, org=None)
    except Exception:
        return "none", (
            "ℹ️ HF 자격증명이 없습니다 — `hf auth login` 하거나 `HF_TOKEN`을 설정하고 다시 실행하면 "
            "본인 Hugging Face 네임스페이스로 게시할 수 있습니다. (Preview·예시 로드는 그대로 사용 가능)"
        )
    ok, reason = verify_write_access(api, namespace=namespace)
    if ok:
        return "ok", f"✅ 인증됨: **{namespace}** — write 권한 확인됨. 게시하면 본인 네임스페이스에 올라갑니다."
    return "readonly", f"⚠️ 토큰은 확인됐지만({namespace}) {reason} write 토큰으로 교체 후 다시 실행하세요."


# Local runs surface the token status in the UI and gate the button on write
# access; on a Space the server token/org gate (PUBLISH_READY) already applies.
AUTH_KIND, AUTH_MESSAGE = ("space", "") if IS_SPACE else _local_auth_status()
PUBLISH_ENABLED = PUBLISH_READY and (IS_SPACE or AUTH_KIND == "ok")

PUBLISH_UNAVAILABLE_MSG = (
    "Live publishing is temporarily unavailable. You can still inspect the "
    "generated manifest and load one of the pre-published examples below."
)
COOLDOWN_SECONDS = 60


def _examples_markdown(examples: list[dict[str, str]]) -> str:
    if not examples:
        return "## 예시 조합\n\n예시 준비 중"
    lines = ["## 예시 조합 (사전 게시, 바로 로드 가능)", ""]
    for ex in examples:
        lines += [
            f"### [{ex['title']}](https://huggingface.co/datasets/{ex['repo_id']})",
            "",
            ex.get("description", ""),
            "",
            "```python",
            ex["snippet"],
            "```",
            "",
        ]
    return "\n".join(lines)


def _make_manifest(
    benchmarks: list[str],
    n_samples: float,
    abilities: list[str],
    name: str,
) -> dict[str, Any]:
    """Single manifest path for preview and publish; gr.Error = input problem."""
    if not benchmarks:
        raise gr.Error("벤치마크를 하나 이상 선택하세요.")
    try:
        safe_name = sanitize_repo_name((name or "").strip())
    except ValueError as exc:
        raise gr.Error("사용할 수 없는 이름입니다. 영숫자로 시작하는 이름을 입력하세요.") from exc
    count = int(n_samples)
    if not 1 <= count <= 5000:
        raise gr.Error("벤치마크당 문항 수는 1~5000 사이여야 합니다.")
    try:
        return build_manifest(
            {bench: count for bench in benchmarks},
            name=safe_name,
            abilities=abilities,
            api=None,
        )
    except ValueError as exc:  # unknown/non-HF benchmark — safe to show
        raise gr.Error(str(exc)) from exc


def _gated_warning(manifest: dict[str, Any]) -> list[str]:
    gated = [src["benchmark"] for src in manifest["sources"] if src.get("gated")]
    if not gated:
        return []
    return [
        "",
        f"⚠️ gated source 포함: {', '.join(gated)} — 로드하려면 각 원본 repo에서 "
        "약관 동의 후 접근 권한이 있는 토큰을 사용하세요.",
    ]


def _model_recommendation_md(ranking: list[tuple[str, float]]) -> str:
    """Top-5 model recommendation over the 5 most relevant benchmarks."""
    top_benches = [bench for bench, _ in ranking[:5]]
    models = rank_models(LEADERBOARD, top_benches)
    if not models:
        return "레퍼런스 점수 데이터가 부족합니다."
    lines = ["**이 능력 조합에서 강한 모델 (상위 벤치 5개 기준):**", ""]
    lines += [f"{i}. {model} — {score:.2f}" for i, (model, score) in enumerate(models[:5], start=1)]
    return "\n".join(lines)


def recommend_by_tags(selected_tags: list[str], current_benchmarks: list[str]) -> tuple[Any, str]:
    """Reorder benchmark choices by tag relevance, preserving the user's selection."""
    current = list(current_benchmarks or [])
    if not TAG_MAP or not selected_tags:
        return gr.update(choices=BENCHMARKS, value=current), ""
    try:
        ranking = relevance_ranking(TAG_MAP["tag_scores"], selected_tags)
    except ValueError:
        return gr.update(choices=BENCHMARKS, value=current), ""
    scored = {bench for bench, _ in ranking}
    choices = [(f"{bench}  ·  관련도 {rel:.2f}", bench) for bench, rel in ranking]
    value = [bench for bench in current if bench in scored]
    return gr.update(choices=choices, value=value), _model_recommendation_md(ranking)


def preview(benchmarks: list[str], n_samples: float, abilities: list[str], name: str) -> str:
    try:
        manifest = _make_manifest(benchmarks, n_samples, abilities, name)
    except gr.Error:
        raise
    except Exception as exc:  # e.g. source repo resolution failure
        raise gr.Error(scrub_secrets(f"미리보기 실패: {exc}", [HF_TOKEN])) from exc
    lines = [f"## 미리보기: {manifest['name']}", ""]
    models = manifest["references"]["models"]
    if models:
        lines += [
            "**기대점수 (레퍼런스 모델, n_samples 가중평균):**",
            "",
            "| model | expected score |",
            "|---|---|",
        ]
        lines += [f"| {model} | {score} |" for model, score in models.items()]
        lines += [""]
    lines += [
        "**Sources:**",
        "",
        "| benchmark | source repo | split | n_samples | gated |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {src['benchmark']} | {src['repo_id']} | {src['split']} "
        f"| {src['n_samples']} | {'yes' if src.get('gated') else 'no'} |"
        for src in manifest["sources"]
    ]
    lines += _gated_warning(manifest)
    lines += ["", "```json", json.dumps(manifest, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines)


def _publish_core(name: str, selections: dict[str, int], abilities: list[str]) -> dict[str, Any]:
    """Resolve creds → verify write access → build + push a composition.

    Shared by the Gradio handler and the JSON ``/api/publish`` route. Raises
    ``ValueError`` with a user-facing (secret-scrubbed) message on any failure;
    on success returns ``{repo_id, url, references, gated}``.
    """
    if not selections:
        raise ValueError("벤치마크를 하나 이상 선택하세요.")
    try:
        safe_name = sanitize_repo_name((name or "").strip())
    except ValueError as exc:
        raise ValueError("사용할 수 없는 이름입니다. 영숫자로 시작하는 이름을 입력하세요.") from exc
    # Server-side gate: on a Space this handler is still reachable as an API, so
    # never fall through to the token owner's personal namespace without ORG.
    if not PUBLISH_READY:
        raise ValueError(PUBLISH_UNAVAILABLE_MSG)
    try:
        api, namespace = resolve_publisher(token=HF_TOKEN, org=BENCHPRESS_ORG)
    except Exception as exc:
        print(scrub_secrets(repr(exc), [HF_TOKEN]))
        raise ValueError(PUBLISH_UNAVAILABLE_MSG if IS_SPACE else CREDENTIAL_HELP) from exc
    ok, reason = verify_write_access(api, namespace=namespace)
    if not ok:
        raise ValueError(scrub_secrets(f"게시 불가: {reason}", [HF_TOKEN]))
    repo_id = build_demo_repo_id(namespace, safe_name)
    try:
        manifest = build_manifest(selections, name=safe_name, abilities=abilities, api=api)
        url = push_composition(repo_id, manifest, api=api)
    except ValueError:
        raise  # unknown/non-HF benchmark or bad input — safe to show
    except Exception as exc:
        print(scrub_secrets(repr(exc), [HF_TOKEN]))
        raise ValueError(PUBLISH_UNAVAILABLE_MSG) from exc
    return {
        "repo_id": repo_id,
        "url": url,
        "references": manifest["references"]["models"],
        "gated": [src["benchmark"] for src in manifest["sources"] if src.get("gated")],
    }


def publish(
    benchmarks: list[str],
    n_samples: float,
    abilities: list[str],
    name: str,
    last_publish: float,
) -> tuple[str, float]:
    now = time.time()
    if now - last_publish < COOLDOWN_SECONDS:
        raise gr.Error("잠시 후 다시 시도하세요 (60초 쿨다운)")
    count = int(n_samples)
    if not 1 <= count <= 5000:
        raise gr.Error("벤치마크당 문항 수는 1~5000 사이여야 합니다.")
    selections = {bench: count for bench in (benchmarks or [])}
    try:
        result = _publish_core(name, selections, abilities)
    except ValueError as exc:
        return str(exc), last_publish
    lines = [
        f"✅ 게시 완료: [{result['repo_id']}]({result['url']})",
        "",
        "```python",
        "from benchpress_hub import load_composition",
        "",
        f'ds = load_composition("{result["repo_id"]}")',
        "```",
    ]
    if result["references"]:
        lines += ["", "**기대점수 (레퍼런스 모델, n_samples 가중평균):**", ""]
        lines += [f"- {model}: **{score}**" for model, score in result["references"].items()]
    if result["gated"]:
        lines += ["", f"⚠️ gated source 포함: {', '.join(result['gated'])} — 로드하려면 각 원본 repo 약관 동의 후 접근 권한이 있는 토큰이 필요합니다."]
    return "\n".join(lines), now


with gr.Blocks(title="BenchPress Composer") as demo:
    gr.Markdown(
        "# 🏋️ BenchPress Composer\n"
        "능력 기반 맞춤 평가셋 recipe를 미리보고, 데모 org에 게시합니다. "
        "로그인·계정·토큰 없이 사용할 수 있습니다. "
        "데이터는 저장되지 않습니다 — `manifest.json`(고정 시험지 recipe)만 저장되고, "
        "로드 시점에 원본에서 streaming으로 가져옵니다."
    )
    gr.Markdown(_examples_markdown(EXAMPLES))
    gr.Markdown("## 조합 만들기")
    benchmarks_input = gr.CheckboxGroup(BENCHMARKS, label="벤치마크")
    abilities_input = gr.CheckboxGroup(
        ABILITY_CHOICES,
        label="평가 능력 (태그) — 선택하면 관련 벤치마크가 추천 정렬됩니다",
    )
    tag_recommend_md = gr.Markdown()
    # ponytail: uniform per-benchmark count; per-bench counts via SDK, UI when asked
    n_samples_input = gr.Number(100, label="벤치마크당 문항 수", precision=0, minimum=1, maximum=5000)
    name_input = gr.Textbox(label="조합 이름", placeholder="my-eval-mix")
    publish_label = "내 조합 게시 (demo org)" if IS_SPACE else "내 조합 게시 (내 HF namespace)"
    with gr.Row():
        preview_btn = gr.Button("Preview", variant="primary")
        publish_btn = gr.Button(publish_label, interactive=PUBLISH_ENABLED)
    if not IS_SPACE:
        gr.Markdown(AUTH_MESSAGE)
    elif not PUBLISH_READY:
        gr.Markdown(
            "ℹ️ 게시 기능이 비활성화되어 있습니다 (HF_TOKEN / BENCHPRESS_ORG 미설정). "
            "Preview와 예시 로드는 그대로 사용할 수 있습니다."
        )
    output = gr.Markdown()
    last_publish_state = gr.State(0.0)
    abilities_input.change(
        recommend_by_tags,
        [abilities_input, benchmarks_input],
        [benchmarks_input, tag_recommend_md],
    )
    preview_btn.click(
        preview,
        [benchmarks_input, n_samples_input, abilities_input, name_input],
        output,
    )
    publish_btn.click(
        publish,
        [benchmarks_input, n_samples_input, abilities_input, name_input, last_publish_state],
        [output, last_publish_state],
    )

# ---- JSON publish API ----------------------------------------------------
# Lets the benchboard Builder publish in-page (its Publish button POSTs here)
# instead of only linking out. CORS is restricted to localhost so only a
# locally-run Builder can call it; the deployed static Builder ships with no
# API URL and keeps its button disabled.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

_LAST_API_PUBLISH = 0.0


@api_app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "space" if IS_SPACE else "local",
        "auth": AUTH_KIND,
        "publish_enabled": PUBLISH_ENABLED,
    }


@api_app.post("/api/publish")
async def api_publish(request: Request) -> JSONResponse:
    global _LAST_API_PUBLISH
    now = time.time()
    if now - _LAST_API_PUBLISH < COOLDOWN_SECONDS:
        return JSONResponse({"ok": False, "error": "잠시 후 다시 시도하세요 (60초 쿨다운)."}, status_code=429)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "잘못된 요청 본문(JSON 아님)."}, status_code=400)
    name = str(body.get("name", "")).strip()
    try:
        selections = {str(k): int(v) for k, v in (body.get("selections") or {}).items()}
    except (TypeError, ValueError, AttributeError):
        return JSONResponse({"ok": False, "error": "잘못된 selections 형식."}, status_code=400)
    abilities = [str(a) for a in (body.get("abilities") or [])]
    try:
        result = _publish_core(name, selections, abilities)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": scrub_secrets(str(exc), [HF_TOKEN])}, status_code=400)
    except Exception as exc:  # unexpected — never leak internals
        print(scrub_secrets(repr(exc), [HF_TOKEN]))
        return JSONResponse({"ok": False, "error": PUBLISH_UNAVAILABLE_MSG}, status_code=500)
    _LAST_API_PUBLISH = now
    return JSONResponse({"ok": True, **result})


# Gradio UI mounted at "/"; the /api routes above stay reachable (registered
# before the catch-all mount).
app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)

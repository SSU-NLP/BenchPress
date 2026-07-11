/**
 * Tag every benchmark under BOTH tagging schemes.
 *
 *   npm run tag
 *
 * Encodes Claude's judgment as deterministic rules (category base + keyword
 * map) per scheme — reproducible and reviewable. Output:
 * data/benchmark-tags.json = { ability: {id:[tags]}, survey: {id:[tags]} }.
 * See src/lib/tags.ts for both vocabularies.
 */
import path from "node:path";
import type { Benchmark } from "../src/lib/types.ts";
import { BENCHMARKS_PATH, DATA_DIR } from "./lib/paths.ts";
import { readJson, writeJson } from "./lib/io.ts";
import { schemeTagIds, type SchemeId } from "../src/lib/tags.ts";

const OUT = path.join(DATA_DIR, "benchmark-tags.json");

interface Ruleset {
  categoryBase: Record<string, string[]>;
  keyword: Array<[RegExp, string[]]>;
  overrides: Record<string, string[]>;
  fallback: string;
}

// ── Scheme 1: profiling abilities ────────────────────────────────────────────
const ABILITY: Ruleset = {
  categoryBase: {
    general: ["knowledge-recall"],
    instruction: ["instruction-following", "semantic-comprehension"],
    math: ["math", "quantitative", "deductive"],
    coding: ["coding", "deductive"],
    agent: ["agentic", "deductive"],
    multimodal: ["multimodal", "semantic-comprehension"],
    vision: ["multimodal", "semantic-comprehension", "spatial-geometric"],
    video: ["multimodal", "semantic-comprehension", "temporal"],
    multilinguality: ["multilingual"],
    long: ["long-context", "contextual-recall"],
    factuality: ["factuality", "knowledge-recall"],
    knowledge: ["knowledge-recall"],
    safety: ["safety"],
    health: ["medical", "knowledge-recall"],
    preference: ["preference"],
    korean: ["korean", "multilingual"],
    chinese: ["chinese", "multilingual"],
    other: [],
  },
  keyword: [
    [/mmlu|gpqa|supergpqa|agieval|trivia|naturalquestion|nq[-_]|openbook|sciq|arc-easy|arc-challenge/, ["knowledge-recall", "deductive"]],
    [/simpleqa|freshqa|halluc|factscore|truthful|personqa|frames|facts-grounding/, ["factuality", "knowledge-recall"]],
    [/arc-agi|raven|analog/, ["inductive", "analogical", "spatial-geometric"]],
    [/bbh|big-bench|zebralogic|autologi|logiqa|logic/, ["deductive", "inductive"]],
    [/gsm8k|math|aime|hmmt|amc|cnmo|usamo|imo|olympiad|polymath|theoremqa|frontiermath|minerva|mathvista|mgsm|hrm8k|ksm|kangaroo|mt-aime|hidden.?math/, ["math", "quantitative", "deductive"]],
    [/winogrande|winograd|hellaswag|piqa|siqa|social-?i|copa|commonsense|physical|story-?cloze/, ["commonsense-causal"]],
    [/drop|squad|race|quac|boolq|coqa|narrativeqa|multirc|comprehension|reading/, ["contextual-recall", "semantic-comprehension"]],
    [/humaneval|mbpp|swe-?bench|swe-?lancer|codeforces|livecodebench|bigcodebench|evalplus|aider|scicode|multipl-?e|cruxeval|repo|monorepo|terminal-?bench|ojbench|spreadsheet|kernel/, ["coding", "deductive"]],
    [/ruler|longbench|infinitebench|\bnih\b|needle|helmet|zeroscrolls|lv-?eval|graphwalks|mrcr|loft|longrag|long-?context|webrag/, ["long-context", "contextual-recall"]],
    [/tau-?bench|tau2|webarena|osworld|browsecomp|browse|terminal|\bmcp\b|gaia|toolbench|bfcl|nexus|gorilla|agent|openrca|paperbench|vending|metr|mle-?bench|finance-?agent|toolathlon|acebench/, ["agentic", "deductive"]],
    [/ifeval|ifbench|collie|multi-?if|instruction-?hierarchy|follow/, ["instruction-following"]],
    [/mmmu|docvqa|chartqa|ai2d|mathvista|charxiv|vqa|video|vibe-?eval|erqa|zerobench|omnidoc|vista|mmbench|seedbench|realworldqa|blink|muirbench|ocr|screenspot|textvqa|refcoco|spatial|vision|visual|lvbench|mvbench|mlvu/, ["multimodal", "semantic-comprehension"]],
    [/jailbreak|refusal|\bsafety\b|harm|toxic|\bbias\b|\bbbq\b|wmdp|\bmask\b|xstest|fortress|propensity|deception|sycophan|red-?team|cbrn|biorisk|cyber|securit|virolog|makemepay|makemesay|wildjailbreak|wildchat|strongreject|gray-?swan/, ["safety"]],
    [/\bmed|usmle|medqa|medmcqa|healthbench|clinical|kormedmcqa|medcalc|pubmed|biolp/, ["medical", "knowledge-recall"]],
    [/arena|alpaca|mt-?bench|wildbench|writingbench|writing|preference|creative|elo|multichallenge/, ["preference"]],
    [/perturb|robust|adversarial|contam|paraphrase|variation/, ["robustness", "deductive"]],
    [/mgsm|mmmlu|global-?mmlu|gmmlu|multilingual|flores|wmt|\binclude\b|blend|milu|okapi|multinrc|uhura|turkishmmlu|jmmlu|indommlu|ammlu|multi-?translation|multi-?exam/, ["multilingual"]],
    [/\bko-|korean|kmmlu|hae-?rae|kobalt|click|logickor|komt|kbl|kbank|hrm8k|kgc|seal-korean|kormed/, ["korean", "multilingual"]],
    [/c-?eval|cmmlu|cmrc|cmath|ccpm|clue|csimpleqa|alignbench|\bc3\b|chinese/, ["chinese", "multilingual"]],
    [/temporal|\btime\b|timeline|chronolog/, ["temporal"]],
    [/spatial|geometr|maze|navigation|map-?reasoning/, ["spatial-geometric"]],
  ],
  overrides: {
    "hle": ["knowledge-recall", "deductive", "quantitative", "factuality"],
    "math-p-hard": ["math", "quantitative", "deductive", "robustness"],
    "math-p-simple": ["math", "quantitative", "deductive"],
    "arc-agi-2": ["inductive", "analogical", "spatial-geometric", "deductive"],
    "gpqa": ["knowledge-recall", "deductive", "quantitative"],
    "simpleqa": ["factuality", "knowledge-recall"],
    "mmlu-pro": ["knowledge-recall", "deductive"],
    "livecodebench": ["coding", "deductive", "robustness"],
  },
  fallback: "knowledge-recall",
};

// ── Scheme 2: survey taxonomy (arXiv:2508.15361, Fig 2) — emits LEAF ids ───────
const SURVEY: Ruleset = {
  categoryBase: {
    general: ["knowledge"],
    instruction: ["robustness-t"], // IFEval sits under Risk&Reliability › Robustness
    math: ["mathematics"],
    coding: ["code-gen"],
    agent: ["agent-integrated"],
    multimodal: ["knowledge"],
    vision: ["knowledge"],
    video: ["knowledge"],
    multilinguality: ["linguistic"],
    long: ["reasoning"],
    factuality: ["hallucination"],
    knowledge: ["knowledge"],
    safety: ["safety-t"],
    health: ["biology"],
    preference: ["linguistic"],
    korean: ["knowledge"],
    chinese: ["knowledge"],
    other: [],
  },
  keyword: [
    [/mmlu\b|agieval|kola|comprehensive|broad-?knowledge|trivia|naturalquestion/, ["knowledge"]],
    [/gpqa|supergpqa|expert|graduate|theoremqa|openbook|scieval|sciknoweval/, ["cross-science"]],
    [/exam|gaokao|m3exam|\bsat\b|\bgre\b|civil-?service|c-?eval|cmmlu|kmmlu|kbl/, ["knowledge"]],
    [/winogrande|hellaswag|piqa|siqa|copa|commonsense|story-?cloze|glue|alpacaeval|bleu|rouge|bertscore|multilingual|flores|xtreme|mdia|dialog|mt-?bench|wildbench|writing|creative|arena|chat|conversation|multichallenge|holistic/, ["linguistic"]],
    [/bbh|big-bench|zebralogic|autologi|logiqa|\blogic|arc\b|arc-|raven|deduc|propositional|strategyqa|proofwriter|folio|reclor|hotpot|livebench|sysbench|prontoqa|cladder|corr2cause/, ["reasoning"]],
    [/gsm8k|\bmath\b|math-|aime|hmmt|amc|cnmo|usamo|imo|olympiad|polymath|frontiermath|minif2f|minerva|mt-aime|hrm8k|ksm|hidden.?math|mathbench|hardmath/, ["mathematics"]],
    [/physics|ugphysics|seephys|physreason|tpbench|feabench/, ["physics"]],
    [/chem|molecule|alchemy/, ["chemistry"]],
    [/\bmed|usmle|medqa|medmcqa|healthbench|clinical|kormedmcqa|medcalc|pubmed|biolp|biology|\bbio\b|lab-bench|sciassess|bixbench/, ["biology"]],
    [/\blaw|legal|kbl|lexeval|lawbench|legalbench|contract|statute/, ["law"]],
    [/patent|\bip-?bench|moZIP/i, ["ip"]],
    [/finance|econ|fin-?qa|financ|accounting|\bbank|flare/, ["finance"]],
    [/educat|tutor|\bedu-|e-eval|teach|student/, ["education"]],
    [/psych|cpsy|psycho/, ["psychology"]],
    [/humaneval|mbpp|\bapps\b|livecodebench|bigcodebench|ds-1000|classeval|scicode|multipl-?e|mmcode|usaco/, ["code-gen"]],
    [/swe-?bench|swe-?lancer|repair|debug|condefects|canitedit|codeeditor|coffe|effibench|aider|terminal-?bench/, ["code-maintenance"]],
    [/codexglue|cruxeval|repobench|codeqa|cosqa|codereview|xcodeeval|code-understand/, ["code-understanding"]],
    [/spider|\bbird\b|cosql|sparc|dusql|nl2bash|opseval|iac-eval|text-?to-?sql|database/, ["database-devops"]],
    [/verilog|rtllm|circuit|cadbench|elecbench|hardware|picbench|resbench/, ["hardware"]],
    [/halluc|truthful|factscore|haluveval|halueval|freshqa|frames|facts-grounding|personqa|faithjudge|faithbench|hhem|felm|factor|simpleqa/, ["hallucination"]],
    [/jailbreak|refusal|\bsafety\b|\bharm|toxic|\bbias\b|\bbbq\b|wmdp|\bmask\b|xstest|fortress|propensity|deception|red-?team|cbrn|biorisk|virolog|strongreject|gray-?swan|wildjailbreak|stereoset|toxigen|hatecheck|sorrybench|do-not-answer/, ["safety-t"]],
    [/cyber|securit|\bctf|exploit|vulnerab/, ["safety-t"]],
    [/ifeval|ifbench|advglue|promptrobust|perturb|robust|adversarial|paraphrase|variation|stress|\bboss\b|cif-bench/, ["robustness-t"]],
    [/contam|data-?leak|leakage|memoriz|wikimia|c2leva/, ["data-leak"]],
    [/browsecomp|browse|mobile-?bench|webwalker|flowbench|spa-bench|robotouille|llf-bench/, ["agent-planning"]],
    [/multi-?agent|multiagentbench|magic|zsc-eval/, ["multi-agent"]],
    [/\bgaia\b|agentbench|agentboard|agentquest|travelplanner|smartplay|balrog|tau-?bench|tau2|colbench|embodied/, ["agent-integrated"]],
    [/osworld|os-?world|scienceagent|agentclinic|mlgym|investorbench|tapilot|theagentcompany|\bmcp\b|mcp-atlas|finance-?agent|computer-?use|openrca|paperbench|vending|\bmetr\b|toolathlon|acebench|bfcl|nexus|webarena/, ["agent-domain"]],
    [/agentharm|safeagent|r-judge|\basb\b/, ["agent-safety"]],
    [/judgebench|or-bench|shoppingmmlu|routerbench|cdeval|normad|emotionqueen|socialstigma|docbench|flub/, ["others"]],
  ],
  overrides: {
    "hle": ["cross-science", "mathematics"],
    "math-p-hard": ["mathematics", "robustness-t"],
    "math-p-simple": ["mathematics"],
    "gpqa": ["cross-science"],
    "simpleqa": ["hallucination"],
    "livecodebench": ["code-gen", "data-leak"],
    "mmlu": ["knowledge"],
    "mmlu-pro": ["knowledge"],
  },
  fallback: "others",
};

function applyRuleset(b: Benchmark, rules: Ruleset, valid: Set<string>): string[] {
  if (rules.overrides[b.id]) {
    return [...new Set(rules.overrides[b.id])].filter((t) => valid.has(t)).sort();
  }
  const set = new Set<string>(rules.categoryBase[b.category] ?? []);
  const hay = `${b.id} ${b.name}`.toLowerCase();
  for (const [re, tags] of rules.keyword) {
    if (re.test(hay)) tags.forEach((t) => set.add(t));
  }
  if (set.size === 0) set.add(rules.fallback);
  return [...set].filter((t) => valid.has(t)).sort();
}

async function main() {
  const benchmarks = await readJson<Benchmark[]>(BENCHMARKS_PATH, []);
  const validAbility = schemeTagIds("ability");
  const validSurvey = schemeTagIds("survey");

  const out: Record<SchemeId, Record<string, string[]>> = {
    ability: {},
    survey: {},
  };
  const counts: Record<SchemeId, Map<string, number>> = {
    ability: new Map(),
    survey: new Map(),
  };

  for (const b of benchmarks) {
    const a = applyRuleset(b, ABILITY, validAbility);
    const s = applyRuleset(b, SURVEY, validSurvey);
    out.ability[b.id] = a;
    out.survey[b.id] = s;
    for (const t of a) counts.ability.set(t, (counts.ability.get(t) ?? 0) + 1);
    for (const t of s) counts.survey.set(t, (counts.survey.get(t) ?? 0) + 1);
  }

  await writeJson(OUT, out);

  for (const scheme of ["ability", "survey"] as const) {
    const avg =
      Object.values(out[scheme]).reduce((a, t) => a + t.length, 0) /
      benchmarks.length;
    console.log(`\n[${scheme}] ${benchmarks.length} benchmarks, avg ${avg.toFixed(1)} tags`);
    const sorted = [...counts[scheme].entries()].sort((a, b) => b[1] - a[1]);
    for (const [t, n] of sorted) console.log(`  ${n.toString().padStart(4)}  ${t}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

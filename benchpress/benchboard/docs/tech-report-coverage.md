# Tech-report coverage of stored models

Inventory of every model that currently has scores in `data/scores/**` (138 active
models), classified by whether benchboard already holds an ingested technical
report for it, and — for those it doesn't — whether a primary-source document
exists in the world to ingest.

Generated 2026-06-13. Legend for the "doc available" column:

- ✅ formal paper / system card / tech report (ideal ingest target)
- 🟡 model card or launch blog only (lighter source, still ingestable)
- ❌ no public document (API-only or aggregator-only)

## A. Tech report already ingested here — 27

These have a `data/scores/tech-reports/<id>.json` produced from a PDF in `docs/`.

| model | report type |
|---|---|
| claude-3.5-haiku | model card |
| claude-3.7-sonnet, claude-haiku-4.5, claude-opus-4.5/4.6/4.7, claude-sonnet-4.5/4.6 | system card |
| gpt-4o, gpt-4.5, gpt-5, o1, o3 | system card |
| gpt-oss-120b, gemini-2.5-flash, gemini-3-pro, grok-4 | model card |
| deepseek-v3, exaone, glm-5, llama-3 (covers 3.1-405b too), qwen-2.5-72b, qwen3-235b, solar-open | paper |
| kimi-k2, nova-pro | tech report |

Note: `nova-pro` is ingested (Amazon Nova family report) but missing from
`docs/_index.json` — minor index gap, the scores are present.

## B. No tech report here but scores exist — 112

Grouped by lab. "covered by" names the document that could be ingested.

### Anthropic
| model | doc | covered by |
|---|---|---|
| claude-3-haiku, claude-3-opus, claude-3-sonnet | ✅ | Claude 3 model card |
| claude-3.5-sonnet | ✅ | Claude 3.5 Sonnet model card / addendum |
| claude-opus-4, claude-opus-4.1, claude-sonnet-4 | ✅ | per-model system cards |
| claude-opus-4.8 | ✅ | system card (newest) |
| claude-fable-5, claude-mythos-5, claude-mythos-preview | 🟡 | verify — Anthropic card likely, not confirmed |

### OpenAI
| model | doc | covered by |
|---|---|---|
| gpt-4 | ✅ | GPT-4 technical report |
| o3-mini, o4-mini | ✅ | o-series system cards (o3/o4 card) |
| gpt-5.1, gpt-5.2, gpt-5.4, gpt-5.5 | ✅ | GPT-5 system-card addenda (gpt-5.2 addendum confirmed) |
| gpt-oss-20b | ✅ | gpt-oss model card — **already in `docs/`** (same card as gpt-oss-120b) |
| gpt-4.1, gpt-4.1-mini, gpt-4o-mini, gpt-4-turbo | 🟡 | launch blog / parent system card |

### Google (Gemini + Gemma)
| model | doc | covered by |
|---|---|---|
| gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro | ✅ | Gemini 1.5 paper |
| gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-flash, gemini-3.1-pro, gemini-3.5-flash | 🟡 | Gemini model cards |
| gemma-3-4b/12b/27b | ✅ | Gemma 3 technical report |
| gemma-4-26b-a4b, gemma-4-31b | ✅ | Gemma 4 technical report (Apr 2026) |

### DeepSeek
| model | doc | covered by |
|---|---|---|
| deepseek-r1, deepseek-r1-zero, deepseek-r1-distill-llama-70b, deepseek-r1-distill-qwen-32b | ✅ | DeepSeek-R1 paper |
| deepseek-v2.5 | ✅ | DeepSeek-V2 paper |
| deepseek-v3.1, deepseek-v3.2 | ✅ | V3.1 / V3.2 reports |
| deepseek-v4-flash, deepseek-v4-pro | ✅ | DeepSeek-V4 technical report (May 2026) |

### Qwen (Alibaba)
| model | doc | covered by |
|---|---|---|
| qwen2.5-7b/14b/32b | ✅ | Qwen2.5 technical report |
| qwen3-30b-a3b, qwen3-32b | ✅ | Qwen3 technical report |
| qwq-32b | ✅ | QwQ report |
| qwen3-max, qwen3.6-max | ❌ | closed-weight, API-only, no public report |

### Zhipu / Z.ai (GLM)
| model | doc | covered by |
|---|---|---|
| glm-4.5, glm-4.5-air, glm-4.6, glm-4.7, glm-4.7-flash | ✅ | GLM-4.5 paper + successor reports |

### Meta (Llama)
| model | doc | covered by |
|---|---|---|
| llama-3-8b, llama-3-70b, llama-3.1-70b | ✅ | Llama 3 herd paper — **already in `docs/` as llama-3** (re-ingest would capture these rows) |
| llama-3.2-90b | ✅ | Llama 3.2 model card |
| llama-3.3-70b | 🟡 | Llama 3.3 model card |
| llama-4-maverick, llama-4-scout | 🟡 | Llama 4 launch blog / model card |

### Mistral
| model | doc | covered by |
|---|---|---|
| magistral-medium, magistral-small | ✅ | Magistral paper |
| mistral-large-3 | 🟡 | "Mistral 3" launch news |
| mistral-medium-3.5 | 🟡 | model card (docs.mistral.ai) |
| mistral-large, mistral-small-3/3.1/3.2, mistral-small-4 | 🟡 | model cards / blog |

### NVIDIA (Nemotron)
| model | doc | covered by |
|---|---|---|
| nemotron-3-nano, nemotron-3-super, nemotron-nano-9b-v2, nemotron-ultra | ✅ | Nemotron technical reports |

### Amazon (Nova)
| model | doc | covered by |
|---|---|---|
| nova-lite, nova-micro | ✅ | Amazon Nova family report (same source as nova-pro) |
| nova-2-lite, nova-2-omni, nova-2-pro | ✅ | Amazon Nova 2 technical report (Dec 2025) |

### Microsoft (Phi)
| model | doc | covered by |
|---|---|---|
| phi-3.5-moe | ✅ | Phi-3.5 paper |
| phi-4, phi-4-mini, phi-4-reasoning-plus | ✅ | Phi-4 technical report |

### Moonshot (Kimi)
| model | doc | covered by |
|---|---|---|
| kimi-k1.5 | ✅ | Kimi K1.5 paper |
| kimi-k2-instruct | ✅ | = Kimi K2 report (already ingested as kimi-k2) |
| kimi-k2-thinking, kimi-k2.5, kimi-k2.6 | ✅ | Kimi-K2 / K2.5 GitHub tech reports |

### MiniMax
| model | doc | covered by |
|---|---|---|
| minimax-m1 | ✅ | MiniMax-M1 paper |
| minimax-m2, minimax-m2.1, minimax-m2.5 | ✅ | MiniMax-M2 series paper (arXiv 2605.26494) |
| minimax-m3 | ❌ | announced/teased only, no report yet |

### xAI (Grok)
| model | doc | covered by |
|---|---|---|
| grok-2, grok-3, grok-4-fast, grok-4-heavy, grok-4.2, grok-4.3 | 🟡 | xAI model cards (docs.x.ai) — no formal tech reports |

### Upstage (Solar)
| model | doc | covered by |
|---|---|---|
| solar-pro-2, solar-pro-3 | 🟡 | launch blogs only (Solar Open has a paper; Pro 2/3 do not) |

### Others
| model | doc | covered by |
|---|---|---|
| ax-4 | ✅ | SKT A.X 4.0 model card + A.X K1 report |
| ax-3.1-light | 🟡 | SKT A.X model card |
| ernie-5 | ✅ | ERNIE 5.0 technical report (arXiv 2602.04705) |
| exaone-4 | ✅ | LG AI EXAONE 4.0 technical report (exaone 3.x paper already ingested) |
| pixtral-large | 🟡 | Pixtral blog / model card |

## Summary

- **Ingested here:** 27 models.
- **Ingestable (✅ formal doc exists):** ~70 models — strongest candidates for
  `docs/` drop + `npm run ingest:watch`.
- **Lighter source only (🟡 model card / blog):** ~38 models.
- **No public document (❌):** only `qwen3-max`, `qwen3.6-max`, `minimax-m3`.

The single highest-leverage action: re-ingesting the **Llama 3 herd paper already
in `docs/`** would add llama-3-8b / llama-3-70b / llama-3.1-70b rows that are
currently missing as own-subjects.

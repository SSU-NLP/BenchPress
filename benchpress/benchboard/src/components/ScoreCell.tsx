import { useEffect, useRef, useState } from "react";
import type { ScoreRecord, ScoreVariant } from "../lib/types";

function fmt(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  if (Math.abs(n) >= 100) return n.toFixed(1);
  return n.toFixed(2);
}

const SOURCE_KIND_LABEL: Record<string, string> = {
  tech_report: "tech report",
  arena_api: "Arena",
  alpaca_csv: "AlpacaEval",
  github_repo: "GitHub",
  hf_dataset: "HuggingFace",
  aggregator_api: "Aggregator",
  manual: "manual",
};

/** Known provenance by URL host — keyed by host suffix. Lets us name the
 *  actual origin (incl. aggregators that re-host other people's numbers)
 *  instead of a generic "Aggregator"/"GitHub". */
const HOST_LABEL: Array<[string, string]> = [
  ["epoch.ai", "Epoch AI"],
  ["artificialanalysis.ai", "Artificial Analysis"],
  ["scale.com", "Scale SEAL"],
  ["llm-stats.com", "LLM-Stats"],
  ["pricepertoken.com", "PricePerToken"],
  ["wulong.dev", "LMArena"],
  ["lmarena.ai", "LMArena"],
  ["arxiv.org", "arXiv"],
  ["anthropic.com", "Anthropic"],
  ["openai.com", "OpenAI"],
  ["llama.com", "Meta"],
  ["upstage.ai", "Upstage"],
  ["amazon.science", "Amazon"],
  ["huggingface.co", "HuggingFace"],
  ["evalplus.github.io", "EvalPlus"],
  ["githubusercontent.com", "GitHub"],
  ["github.io", "GitHub"],
  ["github.com", "GitHub"],
];

function httpHost(url: string): string {
  if (!/^https?:\/\//i.test(url)) return ""; // local PDF paths etc.
  try {
    return new URL(url).host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

/** Best-effort precise origin name for a score's source. */
function sourceLabel(s: ScoreVariant["source"]): string {
  const host = httpHost(s.url);
  if (host) {
    for (const [suffix, label] of HOST_LABEL) {
      if (host === suffix || host.endsWith("." + suffix)) return label;
    }
  }
  if (s.reported_by) return s.reported_by;
  return SOURCE_KIND_LABEL[s.kind] ?? s.kind;
}

/** Only surface a URL in the hover title when it's a real web link — never a
 *  local filesystem path from a tech-report PDF. */
function sourceTitle(s: ScoreVariant["source"]): string {
  const label = sourceLabel(s);
  const detail = s.ref ? `${label}: ${s.ref}` : label;
  return /^https?:\/\//i.test(s.url) ? `${detail} - ${s.url}` : detail;
}

interface Props {
  record?: ScoreRecord;
}

export default function ScoreCell({ record }: Props) {
  const [pinned, setPinned] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!pinned) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setPinned(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPinned(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [pinned]);

  if (!record) return <span className="text-neutral-400">—</span>;

  const variants = record.variants ?? [];
  const allEntries: Array<{ score: number; config: string; source: ScoreVariant["source"] }> = [
    { score: record.score, config: record.config, source: record.source },
    ...variants,
  ];

  const tooltipVisible = pinned ? "block" : "hidden group-hover:block";
  const tooltipInteractive = pinned ? "" : "pointer-events-none";

  return (
    <span
      ref={ref}
      className="group relative inline-flex items-baseline gap-1"
    >
      <span
        onClick={() => setPinned((p) => !p)}
        className="inline-flex items-baseline gap-1 cursor-pointer select-none"
        title={pinned ? "" : "Click to pin the source breakdown"}
      >
        <span className="font-medium tabular-nums">{fmt(record.score)}</span>
        {variants.length > 0 && (
          <span className="text-neutral-400 text-xs tabular-nums">
            ({variants.map((v) => fmt(v.score)).join(", ")})
          </span>
        )}
      </span>
      <span
        className={`absolute left-0 top-full z-20 mt-1 min-w-[14rem] max-w-md max-h-64 overflow-y-auto rounded border border-neutral-200 bg-white px-2 py-1.5 text-xs shadow-lg ${tooltipVisible} ${tooltipInteractive}`}
      >
        {pinned && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setPinned(false);
            }}
            className="absolute top-0.5 right-1 text-neutral-400 hover:text-neutral-700 leading-none"
            aria-label="Close"
          >
            ×
          </button>
        )}
        <ul className="space-y-0.5">
          {allEntries.map((e, i) => (
            <li key={i} className="leading-tight flex items-baseline gap-2">
              <span className="whitespace-nowrap">
                <span className="font-medium tabular-nums">{fmt(e.score)}</span>
                <span className="text-neutral-500"> · {e.config || "default"}</span>
              </span>
              <span
                className="ml-auto text-xs text-neutral-400 truncate min-w-0 shrink"
                title={sourceTitle(e.source)}
              >
                {sourceLabel(e.source)}
              </span>
            </li>
          ))}
        </ul>
      </span>
    </span>
  );
}

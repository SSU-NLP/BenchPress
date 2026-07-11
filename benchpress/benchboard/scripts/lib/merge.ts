import type {
  ScoreExtra,
  ScoreRecord,
  ScoreVariant,
  SourceKind,
} from "../../src/lib/types.ts";

// Reliability as a representative when no publication date is known. Higher
// wins. `manual` is curated (official launch pages / model cards) so it ranks
// above raw aggregator dumps; aggregator hosts are further pinned to 2 below.
const SOURCE_RANK: Record<SourceKind, number> = {
  tech_report: 5,
  arena_api: 4,
  alpaca_csv: 4,
  manual: 4,
  github_repo: 3,
  hf_dataset: 3,
  aggregator_api: 2,
};

function key(r: { model_id: string; benchmark_id: string }): string {
  return `${r.model_id}::${r.benchmark_id}`;
}

function normalizeConfig(c: string): string {
  return c.trim().toLowerCase().replace(/\s+/g, " ");
}

/** A flattened score entry. `extra` only rides along on entries that were a
 *  record's primary (variants never carry extra), so it can follow the score
 *  if that entry becomes the representative again. */
type Entry = ScoreVariant & { extra?: ScoreExtra };

function entriesOf(r: ScoreRecord): Entry[] {
  return [
    { score: r.score, config: r.config, source: r.source, extra: r.extra },
    ...(r.variants ?? []),
  ];
}

function entryKey(e: ScoreVariant): string {
  return [
    normalizeConfig(e.config),
    e.score,
    e.source.url,
    e.source.ref ?? "",
    e.source.reported_by ?? "",
  ].join("::");
}

/** Hosts that merely re-publish other people's numbers. They get the lowest
 *  reliability so a primary source (system card, official leaderboard) wins the
 *  representative slot when we don't have a real publication date to compare. */
const AGGREGATOR_HOSTS = [
  "epoch.ai",
  "artificialanalysis.ai",
  "scale.com",
  "llm-stats.com",
];

function hostOf(url: string): string {
  if (!/^https?:\/\//i.test(url)) return "";
  try {
    return new URL(url).host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

/**
 * How much we trust a source as the representative when no publication date is
 * available. Higher wins. Aggregator hosts are pinned low regardless of `kind`
 * (e.g. llm-stats.com arrives as a github_repo but is second-hand data).
 */
function reliability(e: Entry): number {
  if (e.extra?.user_correction === true) return 6;
  const host = hostOf(e.source.url);
  if (AGGREGATOR_HOSTS.some((h) => host === h || host.endsWith("." + h))) {
    return 2;
  }
  return SOURCE_RANK[e.source.kind] ?? 0;
}

function publishedOf(e: ScoreVariant): string | undefined {
  return e.source.published;
}

/**
 * Representative/ordering comparator (newest/most-trusted first):
 *   1. If both entries carry a real publication date → newer published wins.
 *   2. Otherwise fall back to source reliability (primary > aggregator).
 *   3. Then `fetched_at` recency, then the higher score.
 */
function byRecency(a: Entry, b: Entry): number {
  const pa = publishedOf(a);
  const pb = publishedOf(b);
  if (pa && pb && pa !== pb) return pa > pb ? -1 : 1;

  const rel = reliability(b) - reliability(a);
  if (rel !== 0) return rel;

  if (a.source.fetched_at !== b.source.fetched_at) {
    return a.source.fetched_at > b.source.fetched_at ? -1 : 1;
  }
  return b.score - a.score;
}

/**
 * Rebuild a record from a flat set of entries for one (model, benchmark):
 * de-duplicate by (config, score), sort newest-first, promote the latest entry
 * to the representative score and keep the rest in variants[] newest-first.
 */
function rebuild(
  model_id: string,
  benchmark_id: string,
  rawEntries: Entry[],
): ScoreRecord {
  const seen = new Set<string>();
  const entries: Entry[] = [];
  for (const e of [...rawEntries].sort(byRecency)) {
    const k = entryKey(e);
    if (seen.has(k)) continue;
    seen.add(k);
    entries.push(e);
  }

  const [rep, ...rest] = entries;
  return {
    model_id,
    benchmark_id,
    score: rep.score,
    config: rep.config,
    source: rep.source,
    ...(rep.extra ? { extra: rep.extra } : {}),
    variants: rest.length
      ? rest.map(({ score, config, source }) => ({ score, config, source }))
      : undefined,
  };
}

/**
 * Merge a record into an existing one for the same (model, benchmark). The
 * latest entry becomes the representative; the rest become newest-first
 * variants.
 */
export function mergeOne(existing: ScoreRecord, incoming: ScoreRecord): ScoreRecord {
  return rebuild(existing.model_id, existing.benchmark_id, [
    ...entriesOf(existing),
    ...entriesOf(incoming),
  ]);
}

/**
 * Fold a batch of new records into an existing list, applying mergeOne for
 * every (model_id, benchmark_id) collision. Every surviving record is run
 * through rebuild() so even un-merged records get latest-as-representative /
 * newest-first variant ordering.
 */
export function mergeScores(
  existing: ScoreRecord[],
  incoming: ScoreRecord[],
): ScoreRecord[] {
  const map = new Map<string, ScoreRecord>();
  for (const r of existing) map.set(key(r), r);
  for (const r of incoming) {
    const k = key(r);
    const prev = map.get(k);
    map.set(k, prev ? mergeOne(prev, r) : r);
  }
  return [...map.values()]
    .map((r) => rebuild(r.model_id, r.benchmark_id, entriesOf(r)))
    .sort((a, b) => {
      if (a.benchmark_id !== b.benchmark_id) {
        return a.benchmark_id.localeCompare(b.benchmark_id);
      }
      return b.score - a.score;
    });
}

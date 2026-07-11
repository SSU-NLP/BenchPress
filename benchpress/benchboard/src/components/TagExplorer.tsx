import { useMemo, useState } from "react";
import type {
  ExplorerData,
  ExplorerScore,
  SchemeId,
} from "../lib/loadData";
import {
  SCHEME_ABILITY,
  SCHEMES,
  SURVEY_TREE,
  surveyDescendantLeaves,
  tagById,
  type TaxNode,
} from "../lib/tags";
import { withBase } from "../lib/url";

interface Props {
  data: ExplorerData;
  topK?: number;
}

function fmt(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  if (Math.abs(n) >= 100) return n.toFixed(1);
  return n.toFixed(2);
}

export default function TagExplorer({ data, topK = 15 }: Props) {
  const [scheme, setScheme] = useState<SchemeId>("ability");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [matchAll, setMatchAll] = useState(true);
  const [activeBench, setActiveBench] = useState<string | null>(null);
  const [limit, setLimit] = useState(topK);

  const schemeDef = SCHEMES[scheme];
  const { tagCounts, benchTags } = data.schemes[scheme];

  // Switching scheme clears the other scheme's selection (picking one resets the other).
  // Default match mode differs: flat abilities default to AND (intersect skills),
  // the taxonomy tree defaults to OR (union of categories / subtrees).
  function switchScheme(id: SchemeId) {
    if (id === scheme) return;
    setScheme(id);
    setSelected(new Set());
    setActiveBench(null);
    setLimit(topK);
    setMatchAll(id === "ability");
  }

  function toggleTag(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setActiveBench(null);
  }

  // Survey tree: selecting an internal node toggles its whole subtree of leaves.
  function toggleSubtree(nodeId: string) {
    const leaves = surveyDescendantLeaves(nodeId).filter(
      (l) => (tagCounts[l] ?? 0) > 0,
    );
    setSelected((prev) => {
      const next = new Set(prev);
      const allOn = leaves.length > 0 && leaves.every((l) => next.has(l));
      for (const l of leaves) {
        if (allOn) next.delete(l);
        else next.add(l);
      }
      return next;
    });
    setActiveBench(null);
  }

  // Count of benchmarks under a node (≥1 leaf in its subtree), deduped.
  const nodeCounts = useMemo(() => {
    const out: Record<string, number> = {};
    const leavesOf = (id: string) => new Set(surveyDescendantLeaves(id));
    const allNodeIds = (nodes: TaxNode[]): string[] =>
      nodes.flatMap((n) => [n.id, ...(n.children ? allNodeIds(n.children) : [])]);
    for (const id of allNodeIds(SURVEY_TREE)) {
      const leaves = leavesOf(id);
      let c = 0;
      for (const b of data.benchmarks) {
        const tags = benchTags[b.id] ?? [];
        if (tags.some((t) => leaves.has(t))) c++;
      }
      out[id] = c;
    }
    return out;
  }, [data.benchmarks, benchTags]);

  const matched = useMemo(() => {
    const sel = [...selected];
    const list = data.benchmarks.filter((b) => {
      const tags = benchTags[b.id] ?? [];
      if (sel.length === 0) return true;
      return matchAll
        ? sel.every((t) => tags.includes(t))
        : sel.some((t) => tags.includes(t));
    });
    return list.sort((a, b) => {
      const at = benchTags[a.id] ?? [];
      const bt = benchTags[b.id] ?? [];
      const am = selected.size ? at.filter((t) => selected.has(t)).length : 0;
      const bm = selected.size ? bt.filter((t) => selected.has(t)).length : 0;
      if (bm !== am) return bm - am;
      return b.modelCount - a.modelCount;
    });
  }, [data.benchmarks, benchTags, selected, matchAll]);

  const shown = matched.slice(0, limit);
  const activeScores: ExplorerScore[] | null = activeBench
    ? data.scores[activeBench] ?? []
    : null;
  const activeBenchObj = activeBench
    ? data.benchmarks.find((b) => b.id === activeBench) ?? null
    : null;

  return (
    <div className="space-y-4">
      {/* Scheme toggle — pick one tagging method, the other clears. */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-neutral-500 font-medium uppercase tracking-wide">
          Tagging method
        </span>
        {(Object.keys(SCHEMES) as SchemeId[]).map((id) => {
          const s = SCHEMES[id];
          const on = scheme === id;
          return (
            <button
              key={id}
              onClick={() => switchScheme(id)}
              title={s.blurb}
              className={
                "rounded px-3 py-1 border " +
                (on
                  ? "border-black bg-black text-white"
                  : "border-neutral-300 text-neutral-600 hover:border-neutral-400")
              }
            >
              {s.label}
            </button>
          );
        })}
        <a
          href={schemeDef.url}
          target="_blank"
          rel="noopener"
          className="text-neutral-400 hover:text-neutral-700 underline"
        >
          {schemeDef.paper} ↗
        </a>
      </div>
      <p className="text-xs text-neutral-400 -mt-2">{schemeDef.blurb}</p>

      <div className="grid grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)_minmax(0,1fr)] gap-4">
        {/* ── Section 1: tags ───────────────────────────────────────── */}
        <aside className="lg:border-r lg:border-neutral-200 lg:pr-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Tags
            </h2>
            {selected.size > 0 && (
              <button
                onClick={() => {
                  setSelected(new Set());
                  setActiveBench(null);
                }}
                className="text-xs text-neutral-400 hover:text-neutral-700"
              >
                clear ({selected.size})
              </button>
            )}
          </div>

          {selected.size > 1 && (
            <div className="mb-3 flex gap-1 text-xs">
              {(["all", "any"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setMatchAll(mode === "all")}
                  className={
                    "rounded px-2 py-0.5 border " +
                    ((mode === "all") === matchAll
                      ? "border-black bg-black text-white"
                      : "border-neutral-300 text-neutral-500 hover:border-neutral-400")
                  }
                >
                  {mode === "all" ? "Match all (AND)" : "Match any (OR)"}
                </button>
              ))}
            </div>
          )}

          {scheme === "ability"
            ? SCHEME_ABILITY.groups.map((g) => (
                <div key={g.id} className="mb-4">
                  <div className="text-xs uppercase tracking-wider text-neutral-400 mb-1">
                    {g.label}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {g.tags.map((t) => {
                      const n = tagCounts[t.id] ?? 0;
                      const on = selected.has(t.id);
                      return (
                        <button
                          key={t.id}
                          onClick={() => toggleTag(t.id)}
                          disabled={n === 0}
                          title={t.description}
                          className={
                            "rounded px-2 py-0.5 border text-xs " +
                            (on
                              ? "border-black bg-black text-white"
                              : n === 0
                                ? "border-neutral-200 text-neutral-300 cursor-not-allowed"
                                : "border-neutral-300 text-neutral-700 hover:border-neutral-400")
                          }
                        >
                          {t.label}
                          <span className={on ? "ml-1 opacity-60" : "ml-1 text-neutral-400"}>
                            {n}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))
            : SURVEY_TREE.map((top) => (
                <SurveyBranch
                  key={top.id}
                  node={top}
                  depth={0}
                  selected={selected}
                  nodeCounts={nodeCounts}
                  leafCounts={tagCounts}
                  onToggleLeaf={toggleTag}
                  onToggleSubtree={toggleSubtree}
                />
              ))}
        </aside>

        {/* ── Section 2: matching benchmarks ────────────────────────── */}
        <section className="lg:border-r lg:border-neutral-200 lg:pr-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Benchmarks
            </h2>
            <span className="text-xs text-neutral-400 tabular-nums">
              {matched.length} matching
            </span>
          </div>

          {shown.length === 0 ? (
            <p className="text-sm text-neutral-400 mt-4">
              No benchmarks match the selected tags. Try removing a tag or switching to “Match any (OR)”.
            </p>
          ) : (
            <ul className="space-y-1">
              {shown.map((b) => {
                const tags = benchTags[b.id] ?? [];
                return (
                  <li key={b.id}>
                    <button
                      onClick={() => setActiveBench(b.id)}
                      className={
                        "w-full text-left rounded border px-3 py-2 transition-colors " +
                        (activeBench === b.id
                          ? "border-black bg-neutral-50"
                          : "border-neutral-200 hover:border-neutral-400")
                      }
                    >
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-sm font-medium truncate">{b.name}</span>
                        <span className="text-xs text-neutral-400 tabular-nums shrink-0">
                          {b.modelCount} models
                        </span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {tags.map((t) => (
                          <span
                            key={t}
                            className={
                              "text-xs rounded px-1 py-0.5 " +
                              (selected.has(t)
                                ? "bg-black text-white"
                                : "bg-neutral-100 text-neutral-500")
                            }
                          >
                            {tagById(t)?.label ?? t}
                          </span>
                        ))}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          {matched.length > limit && (
            <button
              onClick={() => setLimit((l) => l + topK)}
              className="mt-3 text-xs text-neutral-500 underline hover:text-neutral-800"
            >
              Show {matched.length - limit} more →
            </button>
          )}
        </section>

        {/* ── Section 3: model scores for selected benchmark ────────── */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Scores
            </h2>
            {activeBenchObj && (
              <a
                href={withBase(`/benchmarks/${activeBenchObj.id}`)}
                className="text-xs text-neutral-400 hover:text-neutral-700 underline"
              >
                detail ↗
              </a>
            )}
          </div>

          {!activeBench ? (
            <p className="text-sm text-neutral-400 mt-4">
              Pick a benchmark from the middle column to see its per-model scores here.
            </p>
          ) : activeScores && activeScores.length > 0 ? (
            <>
              <div className="text-sm font-medium mb-2">{activeBenchObj?.name}</div>
              <ol className="space-y-0.5">
                {activeScores.map((s, i) => (
                  <li
                    key={i}
                    className="flex items-baseline gap-2 border-b border-neutral-100 py-1"
                  >
                    <span className="w-5 text-right text-xs text-neutral-400 tabular-nums">
                      {i + 1}
                    </span>
                    <span className="flex-1 min-w-0 truncate text-sm">
                      {s.model}
                      <span className="ml-1 text-xs text-neutral-400">
                        {s.vendor}
                      </span>
                    </span>
                    {s.config && s.config !== "default" && (
                      <span className="text-xs text-neutral-400 truncate max-w-[40%]">
                        {s.config}
                      </span>
                    )}
                    <span className="font-medium tabular-nums text-sm">
                      {fmt(s.score)}
                    </span>
                  </li>
                ))}
              </ol>
            </>
          ) : (
            <p className="text-sm text-neutral-400 mt-4">
              No model scores recorded for this benchmark yet.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}

/** One node of the survey taxonomy tree. Internal nodes are a clickable header
 *  (toggles whole subtree); leaves render as a selectable chip. */
function SurveyBranch({
  node,
  depth,
  selected,
  nodeCounts,
  leafCounts,
  onToggleLeaf,
  onToggleSubtree,
}: {
  node: TaxNode;
  depth: number;
  selected: Set<string>;
  nodeCounts: Record<string, number>;
  leafCounts: Record<string, number>;
  onToggleLeaf: (id: string) => void;
  onToggleSubtree: (id: string) => void;
}) {
  const isLeaf = !node.children;

  if (isLeaf) {
    const n = leafCounts[node.id] ?? 0;
    const on = selected.has(node.id);
    return (
      <button
        onClick={() => onToggleLeaf(node.id)}
        disabled={n === 0}
        title={node.description}
        className={
          "rounded px-2 py-0.5 border text-xs " +
          (on
            ? "border-black bg-black text-white"
            : n === 0
              ? "border-neutral-200 text-neutral-300 cursor-not-allowed"
              : "border-neutral-300 text-neutral-700 hover:border-neutral-400")
        }
      >
        {node.label}
        <span className={on ? "ml-1 opacity-60" : "ml-1 text-neutral-400"}>{n}</span>
      </button>
    );
  }

  // Internal node: header toggles the subtree; children indented below.
  const count = nodeCounts[node.id] ?? 0;
  const headerCls =
    depth === 0
      ? "text-xs font-semibold uppercase tracking-wide text-neutral-600"
      : "text-xs font-medium text-neutral-500";
  return (
    <div className={depth === 0 ? "mb-3" : "mb-2 ml-2 pl-2 border-l border-neutral-100"}>
      <button
        onClick={() => onToggleSubtree(node.id)}
        disabled={count === 0}
        title={node.description ?? `Select all under ${node.label}`}
        className={
          "mb-1 hover:text-black disabled:text-neutral-300 disabled:cursor-not-allowed " +
          headerCls
        }
      >
        {node.label}
        <span className="ml-1 text-neutral-400 font-normal">{count}</span>
      </button>
      <div className="flex flex-wrap gap-1">
        {node.children!.map((c) =>
          c.children ? (
            <div key={c.id} className="w-full">
              <SurveyBranch
                node={c}
                depth={depth + 1}
                selected={selected}
                nodeCounts={nodeCounts}
                leafCounts={leafCounts}
                onToggleLeaf={onToggleLeaf}
                onToggleSubtree={onToggleSubtree}
              />
            </div>
          ) : (
            <SurveyBranch
              key={c.id}
              node={c}
              depth={depth + 1}
              selected={selected}
              nodeCounts={nodeCounts}
              leafCounts={leafCounts}
              onToggleLeaf={onToggleLeaf}
              onToggleSubtree={onToggleSubtree}
            />
          ),
        )}
      </div>
    </div>
  );
}

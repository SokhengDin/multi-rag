import { SearchX } from "lucide-react";
import { scoreLabel, type Hit, type SearchResult } from "../lib/api";
import { ResultCard } from "./ResultCard";

export interface ResultColumn {
  label: string;
  result?: SearchResult;
  error?: string;
}

interface Props {
  columns: ResultColumn[];
  query: string;
  onImage: (src: string) => void;
  onView: (hit: Hit) => void;
  onPage: (hit: Hit) => void;
}

export function Results({ columns, query, onImage, onView, onPage }: Props) {
  const compare = columns.length > 1;

  return (
    <div className={compare ? "grid gap-4 lg:grid-cols-3" : ""}>
      {columns.map((column) => (
        <section key={column.label} className="min-w-0">
          <div className="mb-2 flex items-baseline justify-between px-1 text-sm">
            <span className="font-medium text-slate-700">{column.label}</span>
            {column.result && (
              <span className="text-xs text-slate-400">{column.result.hits.length} results · {Math.round(column.result.took_ms)} ms</span>
            )}
          </div>

          {column.error ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{column.error}</div>
          ) : column.result && column.result.hits.length === 0 ? (
            <div className="flex flex-col items-center rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-400">
              <SearchX className="mb-2 size-5" />
              No results. Try another method, fewer filters, or wait for ingestion to finish.
            </div>
          ) : (
            <div className="space-y-3">
              {column.result?.hits.map((hit, i) => (
                <ResultCard
                  key={hit.chunk_id}
                  hit={hit}
                  rank={i + 1}
                  query={query}
                  scoreLabel={scoreLabel(column.result!)}
                  compact={compare}
                  onImage={onImage}
                  onView={onView}
                  onPage={onPage}
                />
              ))}
            </div>
          )}
        </section>
      ))}
    </div>
  );
}

import { useState } from "react";
import { ExternalLink, PanelRightOpen, ScanSearch } from "lucide-react";
import { assetUrl, fileUrl, isPdf, type Hit } from "../lib/api";
import { highlight } from "../lib/highlight";
import { PageWithBoxes } from "./PageWithBoxes";
import { Badge, ModalityBadge } from "./ui";

interface Props {
  hit: Hit;
  rank: number;
  query: string;
  scoreLabel: string;
  compact?: boolean;
  onImage: (src: string) => void;
  onView: (hit: Hit) => void;
  onPage: (hit: Hit) => void;
}

export function ResultCard({ hit, rank, query, scoreLabel, compact, onImage, onView, onPage }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const onPdfPage = isPdf(hit.filename) && hit.page != null;
  const ranks = [
    hit.dense_rank && `dense #${hit.dense_rank}`,
    hit.fulltext_rank && `keyword #${hit.fulltext_rank}`,
    hit.sparse_rank && `sparse #${hit.sparse_rank}`,
  ].filter(Boolean);

  return (
    <article className="rounded-xl border border-slate-200 bg-white p-4 transition hover:border-slate-300 hover:shadow-sm">
      <header className="flex flex-wrap items-center gap-1.5 text-sm">
        <span className="font-mono text-xs text-slate-400">#{rank}</span>
        <span className="min-w-0 truncate font-medium text-slate-800" title={hit.filename}>{hit.filename}</span>
        {hit.page != null && <Badge>page {hit.page}</Badge>}
        <ModalityBadge modality={hit.modality} />
        {hit.parser && !compact && <Badge>{hit.parser}</Badge>}
        <span className="ml-auto whitespace-nowrap text-xs text-slate-400">
          {scoreLabel} <span className="font-mono font-semibold text-slate-800">{hit.score.toFixed(compact ? 3 : 4)}</span>
        </span>
      </header>

      {hit.heading_path?.length ? <p className="mt-1 truncate text-xs text-slate-400">{hit.heading_path.join(" › ")}</p> : null}
      {hit.json_path && <p className="mt-1 font-mono text-xs text-slate-400">{hit.json_path}</p>}

      <div className={`mt-3 flex gap-4 ${compact ? "flex-col" : "flex-col sm:flex-row"}`}>
        {onPdfPage ? (
          <button onClick={() => onPage(hit)} title="Show on page" className={`shrink-0 overflow-hidden rounded-lg border border-slate-200 transition hover:border-amber-400 ${compact ? "" : "sm:w-44"}`}>
            <PageWithBoxes documentId={hit.document_id} page={hit.page!} boxes={hit.boxes} dpi={60} />
          </button>
        ) : hit.image_url && !imageFailed && (
          <button onClick={() => onImage(assetUrl(hit.image_url!))} className={`shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 ${compact ? "" : "sm:w-56"}`}>
            <img src={assetUrl(hit.image_url)} loading="lazy" onError={() => setImageFailed(true)} className="max-h-56 w-full object-contain" />
          </button>
        )}
        <div className="min-w-0 flex-1">
          <p className={`whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700 ${expanded ? "" : compact ? "line-clamp-4" : "line-clamp-6"} ${hit.modality === "table" ? "font-mono text-xs" : ""}`}>
            {hit.content?.trim() ? highlight(hit.content, query) : <span className="italic text-slate-400">No text, matched by the image.</span>}
          </p>

          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
            <button onClick={() => setExpanded((v) => !v)} className="text-indigo-600 hover:underline">{expanded ? "Show less" : "Show more"}</button>
            {onPdfPage && (
              <button onClick={() => onPage(hit)} className="inline-flex items-center gap-1 text-indigo-600 hover:underline">
                <ScanSearch className="size-3" /> Show on page
              </button>
            )}
            {hit.image_url && onPdfPage && (
              <button onClick={() => onImage(assetUrl(hit.image_url!))} className="text-indigo-600 hover:underline">View image</button>
            )}
            <button onClick={() => onView(hit)} className="inline-flex items-center gap-1 text-indigo-600 hover:underline">
              <PanelRightOpen className="size-3" /> View in document
            </button>
            <a href={fileUrl(hit.document_id, hit.filename, hit.page)} target="_blank" className="inline-flex items-center gap-1 text-indigo-600 hover:underline">
              <ExternalLink className="size-3" /> {onPdfPage ? `Open page ${hit.page}` : "Open file"}
            </a>
            {ranks.length > 0 && <span className="text-slate-400">{ranks.join(" · ")}</span>}
          </div>
        </div>
      </div>
    </article>
  );
}

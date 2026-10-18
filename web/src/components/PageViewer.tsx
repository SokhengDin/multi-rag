import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, ExternalLink, X } from "lucide-react";
import { fileUrl, type Hit } from "../lib/api";
import { PageWithBoxes } from "./PageWithBoxes";
import { ModalityBadge } from "./ui";

export function PageViewer({ hit, onClose }: { hit: Hit; onClose: () => void }) {
  const [page, setPage] = useState(hit.page ?? 1);
  const onHitPage = page === hit.page;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") setPage((p) => Math.max(1, p - 1));
      if (e.key === "ArrowRight") setPage((p) => p + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div onClick={onClose} className="fixed inset-0 z-50 flex flex-col bg-slate-900/85">
      <header onClick={(e) => e.stopPropagation()} className="flex items-center gap-3 bg-white px-4 py-2.5 text-sm shadow">
        <span className="truncate font-medium">{hit.filename}</span>
        <ModalityBadge modality={hit.modality} />
        <div className="ml-auto flex items-center gap-1">
          <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1} className="rounded p-1 hover:bg-slate-100 disabled:opacity-30"><ChevronLeft className="size-4" /></button>
          <span className="w-20 text-center tabular-nums">page {page}</span>
          <button onClick={() => setPage((p) => p + 1)} className="rounded p-1 hover:bg-slate-100"><ChevronRight className="size-4" /></button>
        </div>
        {!onHitPage && <button onClick={() => setPage(hit.page ?? 1)} className="text-indigo-600 hover:underline">Back to hit</button>}
        <a href={fileUrl(hit.document_id, hit.filename, page)} target="_blank" className="inline-flex items-center gap-1 text-indigo-600 hover:underline">
          <ExternalLink className="size-3.5" /> Open PDF
        </a>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-700"><X className="size-5" /></button>
      </header>

      <div className="flex-1 overflow-auto p-6">
        <div onClick={(e) => e.stopPropagation()} className="mx-auto w-full max-w-4xl">
          <PageWithBoxes key={page} documentId={hit.document_id} page={page} boxes={onHitPage ? hit.boxes : null} dpi={150} className="overflow-hidden rounded-lg shadow-2xl" />
          {onHitPage && !hit.boxes?.length && (
            <p className="mt-3 text-center text-sm text-slate-300">No location saved for this chunk. Re-ingest the document to get boxes.</p>
          )}
        </div>
      </div>
    </div>
  );
}

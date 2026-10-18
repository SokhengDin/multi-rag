import { useEffect, useMemo, useState } from "react";
import { ExternalLink, Loader2, RotateCw, Trash2, X } from "lucide-react";
import { api, assetUrl, fileUrl, PARSERS, type Chunk, type Document, type Job, type Parser } from "../lib/api";
import { useToast } from "./Toast";
import { Badge, ModalityBadge, StatusBadge } from "./ui";

interface Props {
  doc: Document;
  job?: Job;
  focusChunkId?: string;
  onClose: () => void;
  onChanged: () => void;
  onImage: (src: string) => void;
}

export function DocumentDrawer({ doc, job, focusChunkId, onClose, onChanged, onImage }: Props) {
  const [chunks, setChunks] = useState<Chunk[] | null>(null);
  const [parser, setParser] = useState<Parser>(job?.parser ?? "vlm");
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  useEffect(() => {
    setChunks(null);
    api.chunks(doc.id).then(setChunks).catch((e) => toast(e.message, "error"));
  }, [doc.id, doc.num_chunks, toast]);

  useEffect(() => {
    if (chunks && focusChunkId) document.getElementById(`chunk-${focusChunkId}`)?.scrollIntoView({ block: "center" });
  }, [chunks, focusChunkId]);

  const pages = useMemo(() => {
    const groups = new Map<string, Chunk[]>();
    for (const c of chunks ?? []) {
      const key = c.page != null ? `Page ${c.page}` : "No page";
      groups.set(key, [...(groups.get(key) ?? []), c]);
    }
    return [...groups];
  }, [chunks]);

  async function reingest() {
    setBusy(true);
    try {
      await api.reingest(doc.id, parser);
      toast(`Re-ingesting with ${parser}`);
      onChanged();
    } catch (e) {
      toast((e as Error).message, "error");
    }
    setBusy(false);
  }

  async function remove() {
    if (!confirm(`Delete ${doc.filename}? Its chunks, jobs and stored files are removed permanently.`)) return;
    try {
      await api.remove(doc.id);
      toast(`Deleted ${doc.filename}`);
      onChanged();
      onClose();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  }

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-900/30" onClick={onClose}>
      <aside onClick={(e) => e.stopPropagation()} className="flex h-full w-full max-w-2xl flex-col bg-white shadow-2xl">
        <header className="border-b border-slate-200 p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="truncate font-semibold">{doc.filename}</h2>
              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
                <StatusBadge status={doc.status} />
                <span>{doc.num_chunks} chunks</span>
                {job && <span>· {job.parser}</span>}
                <span>· {(doc.size_bytes / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            </div>
            <button onClick={onClose} className="text-slate-400 hover:text-slate-700"><X className="size-5" /></button>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <a href={fileUrl(doc.id, doc.filename)} target="_blank" className="inline-flex items-center gap-1 rounded-lg border border-slate-300 px-2.5 py-1 hover:bg-slate-50">
              <ExternalLink className="size-3.5" /> Open file
            </a>
            <div className="inline-flex items-center rounded-lg border border-slate-300">
              <select value={parser} onChange={(e) => setParser(e.target.value as Parser)} title={PARSERS[parser]} className="rounded-l-lg bg-transparent py-1 pl-2.5 pr-1 outline-none">
                {(Object.keys(PARSERS) as Parser[]).map((p) => <option key={p}>{p}</option>)}
              </select>
              <button onClick={reingest} disabled={busy} className="inline-flex items-center gap-1 border-l border-slate-300 px-2.5 py-1 hover:bg-slate-50 disabled:opacity-50">
                {busy ? <Loader2 className="size-3.5 animate-spin" /> : <RotateCw className="size-3.5" />} Re-ingest
              </button>
            </div>
            <button onClick={remove} className="ml-auto inline-flex items-center gap-1 rounded-lg border border-rose-200 px-2.5 py-1 text-rose-600 hover:bg-rose-50">
              <Trash2 className="size-3.5" /> Delete
            </button>
          </div>
          {doc.error && <p className="mt-2 rounded-lg bg-rose-50 p-2 text-xs text-rose-700">{doc.error}</p>}
        </header>

        <div className="flex-1 overflow-y-auto p-4">
          {chunks === null ? (
            <div className="flex justify-center py-12"><Loader2 className="size-5 animate-spin text-slate-400" /></div>
          ) : chunks.length === 0 ? (
            <p className="py-12 text-center text-sm text-slate-400">No chunks yet. They appear once ingestion finishes.</p>
          ) : (
            pages.map(([page, items]) => (
              <section key={page} className="mb-6">
                <h3 className="sticky top-0 z-10 -mx-4 mb-2 bg-white/95 px-4 py-1 text-xs font-semibold uppercase tracking-wide text-slate-400 backdrop-blur">{page}</h3>
                <div className="space-y-2">
                  {items.map((c) => (
                    <div
                      key={c.id}
                      id={`chunk-${c.id}`}
                      className={`rounded-lg border p-3 ${c.id === focusChunkId ? "border-indigo-400 bg-indigo-50/50 ring-2 ring-indigo-100" : "border-slate-200"}`}
                    >
                      <div className="mb-2 flex items-center gap-1.5 text-xs text-slate-400">
                        <span className="font-mono">#{c.chunk_index}</span>
                        <ModalityBadge modality={c.modality} />
                        {c.token_count != null && <span>{c.token_count} tokens</span>}
                        {c.heading_path?.length ? <Badge>{c.heading_path.join(" › ")}</Badge> : null}
                      </div>
                      <div className="flex gap-3">
                        {c.image_url && (
                          <button onClick={() => onImage(assetUrl(c.image_url!))} className="shrink-0">
                            <img src={assetUrl(c.image_url)} loading="lazy" className="max-h-40 w-40 rounded border border-slate-200 bg-slate-50 object-contain" />
                          </button>
                        )}
                        <p className={`min-w-0 flex-1 whitespace-pre-wrap break-words text-sm text-slate-700 ${c.modality === "table" ? "font-mono text-xs" : ""}`}>
                          {c.content || <span className="italic text-slate-400">No text, embedded from the image.</span>}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}

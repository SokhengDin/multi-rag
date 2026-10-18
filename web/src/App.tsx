import { useCallback, useEffect, useState } from "react";
import { BookOpen, Layers } from "lucide-react";
import { api, API_URL, type Document, type Filters, type Hit, type Job, type SearchMode } from "./lib/api";
import { DocumentDrawer } from "./components/DocumentDrawer";
import { DocumentList } from "./components/DocumentList";
import { ImageModal } from "./components/ImageModal";
import { PageViewer } from "./components/PageViewer";
import { Results, type ResultColumn } from "./components/Results";
import { SearchPanel, type SearchSpec } from "./components/SearchPanel";
import { useToast } from "./components/Toast";
import { UploadPanel } from "./components/UploadPanel";

const COMPARE: { label: string; mode: SearchMode; rerank: boolean }[] = [
  { label: "Dense", mode: "dense", rerank: false },
  { label: "Hybrid", mode: "hybrid", rerank: false },
  { label: "Hybrid + Rerank", mode: "hybrid", rerank: true },
];

export default function App() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [columns, setColumns] = useState<ResultColumn[]>([]);
  const [lastQuery, setLastQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [openDoc, setOpenDoc] = useState<{ id: string; focusChunkId?: string } | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [pageHit, setPageHit] = useState<Hit | null>(null);
  const toast = useToast();

  const refresh = useCallback(async () => {
    try {
      const [docList, jobList] = await Promise.all([api.documents(), api.jobs()]);
      const latest: Record<string, Job> = {};
      for (const job of jobList) latest[job.document_id] ??= job;
      setDocs(docList);
      setJobs(latest);
      setSelected((s) => new Set([...s].filter((id) => docList.some((d) => d.id === id))));
    } catch (e) {
      toast(`Cannot reach the API at ${API_URL}: ${(e as Error).message}`, "error");
    }
  }, [toast]);

  useEffect(() => { refresh(); }, [refresh]);

  const busy = docs.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [busy, refresh]);

  async function search(spec: SearchSpec) {
    const filters: Filters = {};
    if (selected.size) filters.document_ids = [...selected];
    if (spec.modalities.length) filters.modalities = spec.modalities;
    if (spec.parser) filters.parser = spec.parser;

    setSearching(true);
    setLastQuery(spec.query);

    if (spec.kind === "image") {
      try {
        const result = await api.searchImage(spec.image!, spec.query, spec.topK, spec.candidates, filters);
        setColumns([{ label: spec.query ? `Image + "${spec.query}"` : "Image", result }]);
      } catch (e) {
        setColumns([{ label: "Image", error: (e as Error).message }]);
      }
      setSearching(false);
      return;
    }

    const configs = spec.compare ? COMPARE : [{ label: `${spec.mode === "dense" ? "Dense" : "Hybrid"}${spec.rerank ? " + Rerank" : ""}`, mode: spec.mode, rerank: spec.rerank }];
    const settled = await Promise.allSettled(
      configs.map((c) => api.search({
        query: spec.query, mode: c.mode, rerank: c.rerank,
        top_k: spec.topK, candidates: spec.candidates, rerank_top: spec.rerankTop, filters,
      })),
    );
    setColumns(configs.map((c, i) => {
      const s = settled[i];
      return s.status === "fulfilled" ? { label: c.label, result: s.value } : { label: c.label, error: (s.reason as Error).message };
    }));
    setSearching(false);
  }

  const toggle = (id: string) => setSelected((s) => {
    const next = new Set(s);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const selectedNames = docs.filter((d) => selected.has(d.id)).map((d) => d.filename);
  const scope = selectedNames.length ? `Searching in ${selectedNames.join(", ")}` : `Searching all ${docs.length} documents`;
  const drawerDoc = openDoc && docs.find((d) => d.id === openDoc.id);

  return (
    <div className="flex h-screen flex-col bg-slate-50 text-slate-800 antialiased">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
        <div className="flex items-center gap-2">
          <div className="flex size-7 items-center justify-center rounded-lg bg-indigo-600 text-white"><Layers className="size-4" /></div>
          <h1 className="font-semibold">Multi RAG</h1>
          <span className="hidden text-sm text-slate-400 sm:inline">Multimodal search on Postgres + pgvector</span>
        </div>
        <a href={`${API_URL}/docs`} target="_blank" className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800">
          <BookOpen className="size-4" /> API docs
        </a>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto p-5 lg:flex-row lg:overflow-hidden">
        <aside className="flex shrink-0 flex-col gap-5 lg:w-80">
          <UploadPanel onUploaded={refresh} />
          <DocumentList
            docs={docs}
            jobs={jobs}
            selected={selected}
            onToggle={toggle}
            onClearSelection={() => setSelected(new Set())}
            onOpen={(doc) => setOpenDoc({ id: doc.id })}
            onRefresh={refresh}
          />
        </aside>

        <main className="min-w-0 flex-1 space-y-5 lg:overflow-y-auto lg:pr-1">
          <SearchPanel busy={searching} scope={scope} onSearch={search} />
          {columns.length > 0 ? (
            <Results
              columns={columns}
              query={lastQuery}
              onImage={setImage}
              onView={(hit: Hit) => setOpenDoc({ id: hit.document_id, focusChunkId: hit.chunk_id })}
              onPage={setPageHit}
            />
          ) : (
            <p className="px-1 pt-8 text-center text-sm text-slate-400">
              Upload a document, then search its text, tables and charts.
            </p>
          )}
        </main>
      </div>

      {drawerDoc && (
        <DocumentDrawer
          doc={drawerDoc}
          job={jobs[drawerDoc.id]}
          focusChunkId={openDoc?.focusChunkId}
          onClose={() => setOpenDoc(null)}
          onChanged={refresh}
          onImage={setImage}
        />
      )}
      {pageHit && <PageViewer hit={pageHit} onClose={() => setPageHit(null)} />}
      <ImageModal src={image} onClose={() => setImage(null)} />
    </div>
  );
}

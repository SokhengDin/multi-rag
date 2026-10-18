import { useRef, useState, type DragEvent } from "react";
import { FileUp, Loader2 } from "lucide-react";
import { api, PARSERS, type Parser } from "../lib/api";
import { useToast } from "./Toast";
import { Card } from "./ui";

const ACCEPT = ".pdf,.pptx,.md,.markdown,.txt,.json";

export function UploadPanel({ onUploaded }: { onUploaded: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [parser, setParser] = useState<Parser>("vlm");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    setFiles([...e.dataTransfer.files]);
  };

  async function upload() {
    setBusy(true);
    let queued = 0;
    for (const file of files) {
      try {
        await api.upload(file, parser);
        queued++;
      } catch (err) {
        toast(`${file.name}: ${(err as Error).message}`, "error");
      }
    }
    setBusy(false);
    setFiles([]);
    if (input.current) input.current.value = "";
    if (queued) toast(`${queued} file${queued > 1 ? "s" : ""} queued for ingestion`);
    onUploaded();
  }

  return (
    <Card className="p-4">
      <h2 className="mb-3 text-sm font-semibold">Upload</h2>

      <label
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-5 text-center transition ${dragging ? "border-indigo-400 bg-indigo-50" : "border-slate-300 hover:border-indigo-300 hover:bg-slate-50"}`}
      >
        <FileUp className="mb-1.5 size-5 text-slate-400" />
        {files.length ? (
          <span className="text-sm font-medium text-slate-700">{files.map((f) => f.name).join(", ")}</span>
        ) : (
          <>
            <span className="text-sm text-slate-600">Drop files or click to choose</span>
            <span className="mt-0.5 text-xs text-slate-400">PDF, PPTX, Markdown, TXT, JSON</span>
          </>
        )}
        <input ref={input} type="file" multiple accept={ACCEPT} className="hidden" onChange={(e) => setFiles([...(e.target.files ?? [])])} />
      </label>

      <div className="mt-3 grid grid-cols-4 gap-1 rounded-lg bg-slate-100 p-0.5 text-xs">
        {(Object.keys(PARSERS) as Parser[]).map((p) => (
          <button key={p} type="button" onClick={() => setParser(p)} className={`rounded-md py-1 font-medium transition ${parser === p ? "bg-white shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
            {p}
          </button>
        ))}
      </div>
      <p className="mt-1.5 min-h-8 text-xs text-slate-500">{PARSERS[parser]}</p>

      <button
        onClick={upload}
        disabled={!files.length || busy}
        className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {busy && <Loader2 className="size-4 animate-spin" />}
        {busy ? "Uploading…" : "Upload & ingest"}
      </button>
    </Card>
  );
}

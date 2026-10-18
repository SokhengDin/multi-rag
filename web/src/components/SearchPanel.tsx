import { useEffect, useState } from "react";
import { ImageIcon, Loader2, Search, SlidersHorizontal } from "lucide-react";
import { PARSERS, type Modality, type Parser, type SearchMode } from "../lib/api";
import { Card, inputClass, Segmented } from "./ui";

export interface SearchSpec {
  kind: "text" | "image";
  query: string;
  image?: File;
  mode: SearchMode;
  rerank: boolean;
  compare: boolean;
  topK: number;
  candidates: number;
  rerankTop: number;
  modalities: Modality[];
  parser?: Parser;
}

const MODE_HINT: Record<SearchMode, string> = {
  dense: "Meaning-based vector search. Finds charts and images by what they show.",
  hybrid: "Vectors + keywords fused with RRF. Good for exact names, terms and numbers.",
};

export function SearchPanel({ busy, scope, onSearch }: { busy: boolean; scope: string; onSearch: (spec: SearchSpec) => void }) {
  const [kind, setKind] = useState<"text" | "image">("text");
  const [query, setQuery] = useState("");
  const [image, setImage] = useState<File>();
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [rerank, setRerank] = useState(false);
  const [compare, setCompare] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [topK, setTopK] = useState(10);
  const [candidates, setCandidates] = useState(100);
  const [rerankTop, setRerankTop] = useState(30);
  const [modalities, setModalities] = useState<Modality[]>([]);
  const [parser, setParser] = useState<Parser | "">("");

  const canSearch = kind === "text" ? query.trim().length > 0 : !!image;

  const [preview, setPreview] = useState<string>();
  useEffect(() => {
    if (!image) return setPreview(undefined);
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSearch || busy) return;
    onSearch({
      kind, query: query.trim(), image, mode, rerank, compare: kind === "text" && compare,
      topK, candidates, rerankTop, modalities, parser: parser || undefined,
    });
  }

  const toggleModality = (m: Modality) => setModalities((all) => (all.includes(m) ? all.filter((x) => x !== m) : [...all, m]));

  return (
    <Card className="p-4">
      <form onSubmit={submit} className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <Segmented
            value={kind}
            onChange={setKind}
            options={[
              { value: "text", label: <span className="flex items-center gap-1.5"><Search className="size-3.5" />Text</span> },
              { value: "image", label: <span className="flex items-center gap-1.5"><ImageIcon className="size-3.5" />Image</span> },
            ]}
          />
          <button type="button" onClick={() => setShowOptions((v) => !v)} className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-sm ${showOptions ? "bg-slate-100 text-slate-900" : "text-slate-500 hover:text-slate-800"}`}>
            <SlidersHorizontal className="size-3.5" /> Options
          </button>
        </div>

        {kind === "text" ? (
          <div className="flex gap-2">
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. xgb accuracy 0.85" className={`${inputClass} py-2.5 text-base`} autoFocus />
            <SubmitButton busy={busy} disabled={!canSearch} />
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <label className="flex h-20 w-28 shrink-0 cursor-pointer items-center justify-center overflow-hidden rounded-lg border-2 border-dashed border-slate-300 text-xs text-slate-500 hover:border-indigo-300">
              {preview ? <img src={preview} className="h-full w-full object-contain" /> : "Choose image"}
              <input type="file" accept="image/*" className="hidden" onChange={(e) => setImage(e.target.files?.[0])} />
            </label>
            <div className="flex flex-1 gap-2">
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Optional text, e.g. revenue" className={`${inputClass} py-2.5 text-base`} />
              <SubmitButton busy={busy} disabled={!canSearch} />
            </div>
          </div>
        )}

        {kind === "text" ? (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
            <div className={compare ? "pointer-events-none opacity-40" : ""}>
              <Segmented value={mode} onChange={setMode} options={[{ value: "dense", label: "Dense" }, { value: "hybrid", label: "Hybrid" }]} />
            </div>
            <Toggle checked={rerank && !compare} disabled={compare} onChange={setRerank} label="Rerank" />
            <Toggle checked={compare} onChange={setCompare} label="Compare methods" />
          </div>
        ) : null}

        <p className="text-xs text-slate-500">
          {kind === "image"
            ? "Finds charts, tables, pages or text similar to the image (dense vectors)."
            : compare
              ? "Runs Dense, Hybrid and Hybrid + Rerank side by side on the same query."
              : MODE_HINT[mode] + (rerank ? " A cross-encoder then re-scores the top results." : "")}
        </p>

        {showOptions && (
          <div className="grid gap-3 rounded-lg bg-slate-50 p-3 sm:grid-cols-2">
            <div className="grid grid-cols-3 gap-2">
              <NumberField label="Top K" value={topK} onChange={setTopK} min={1} max={100} />
              <NumberField label="Candidates" value={candidates} onChange={setCandidates} min={1} max={1000} />
              <NumberField label="Rerank top" value={rerankTop} onChange={setRerankTop} min={1} max={200} />
            </div>
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <span className="w-14 text-slate-500">Type</span>
                {(["text", "table", "image"] as Modality[]).map((m) => (
                  <Chip key={m} active={modalities.includes(m)} onClick={() => toggleModality(m)}>{m}</Chip>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <span className="w-14 text-slate-500">Parser</span>
                <Chip active={parser === ""} onClick={() => setParser("")}>any</Chip>
                {(Object.keys(PARSERS) as Parser[]).map((p) => (
                  <Chip key={p} active={parser === p} onClick={() => setParser(p)}>{p}</Chip>
                ))}
              </div>
            </div>
          </div>
        )}

        <p className="text-xs text-slate-400">{scope}</p>
      </form>
    </Card>
  );
}

function SubmitButton({ busy, disabled }: { busy: boolean; disabled: boolean }) {
  return (
    <button disabled={disabled || busy} className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-indigo-600 px-5 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40">
      {busy ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
      Search
    </button>
  );
}

function Toggle({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <label className={`inline-flex cursor-pointer items-center gap-2 ${disabled ? "opacity-40" : ""}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full transition ${checked ? "bg-indigo-600" : "bg-slate-300"}`}
      >
        <span className={`absolute top-0.5 size-4 rounded-full bg-white shadow transition-all ${checked ? "left-4.5" : "left-0.5"}`} />
      </button>
      <span className="text-slate-700">{label}</span>
    </label>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} className={`rounded-full px-2.5 py-0.5 ring-1 ring-inset transition ${active ? "bg-indigo-600 text-white ring-indigo-600" : "bg-white text-slate-600 ring-slate-300 hover:ring-slate-400"}`}>
      {children}
    </button>
  );
}

function NumberField({ label, value, onChange, min, max }: { label: string; value: number; onChange: (v: number) => void; min: number; max: number }) {
  return (
    <label className="text-xs text-slate-500">
      {label}
      <input type="number" value={value} min={min} max={max} onChange={(e) => onChange(Math.min(max, Math.max(min, Number(e.target.value) || min)))} className={`${inputClass} mt-1 py-1.5`} />
    </label>
  );
}

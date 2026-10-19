import type { ReactNode } from "react";
import { isBusy, type Ingestion, type JobStatus, type Modality } from "../lib/api";

const MODALITY_STYLE: Record<Modality, string> = {
  text: "bg-sky-50 text-sky-700 ring-sky-200",
  table: "bg-violet-50 text-violet-700 ring-violet-200",
  image: "bg-pink-50 text-pink-700 ring-pink-200",
};

const STATUS_STYLE: Record<JobStatus, string> = {
  done: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  queued: "bg-slate-50 text-slate-600 ring-slate-200",
  running: "bg-amber-50 text-amber-700 ring-amber-200",
  failed: "bg-rose-50 text-rose-700 ring-rose-200",
};

export function Badge({ children, className = "bg-slate-50 text-slate-600 ring-slate-200" }: { children: ReactNode; className?: string }) {
  return <span className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset ${className}`}>{children}</span>;
}

export const ModalityBadge = ({ modality }: { modality: Modality }) => <Badge className={MODALITY_STYLE[modality]}>{modality}</Badge>;

/** e.g. "vlm · 94" when done, "vlm · running", "vlm · failed" */
export function IngestionBadge({ ingestion }: { ingestion: Ingestion }) {
  const label = ingestion.status === "done" ? ingestion.num_chunks : ingestion.status;
  return (
    <span title={ingestion.error ?? `${ingestion.parser}: ${ingestion.status}, ${ingestion.num_chunks} chunks searchable`}>
      <Badge className={STATUS_STYLE[ingestion.status]}>
        {isBusy(ingestion) && <span className="size-1.5 animate-pulse rounded-full bg-amber-500" />}
        {ingestion.parser} · {label}
      </Badge>
    </span>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-lg bg-slate-100 p-0.5 text-sm">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1 font-medium transition ${value === o.value ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-slate-200 bg-white ${className}`}>{children}</div>;
}

export const inputClass = "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100";

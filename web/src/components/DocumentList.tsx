import { FileText, RefreshCw } from "lucide-react";
import type { Document, Job } from "../lib/api";
import { Card, StatusBadge } from "./ui";

interface Props {
  docs: Document[];
  jobs: Record<string, Job>;
  selected: Set<string>;
  onToggle: (id: string) => void;
  onClearSelection: () => void;
  onOpen: (doc: Document) => void;
  onRefresh: () => void;
}

export function DocumentList({ docs, jobs, selected, onToggle, onClearSelection, onOpen, onRefresh }: Props) {
  return (
    <Card className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <h2 className="text-sm font-semibold">Documents <span className="font-normal text-slate-400">{docs.length}</span></h2>
        <div className="flex items-center gap-3 text-xs">
          {selected.size > 0 && <button onClick={onClearSelection} className="text-indigo-600 hover:underline">Clear {selected.size}</button>}
          <button onClick={onRefresh} className="text-slate-400 hover:text-slate-700" title="Refresh"><RefreshCw className="size-3.5" /></button>
        </div>
      </div>

      {docs.length === 0 ? (
        <p className="p-8 text-center text-sm text-slate-400">No documents yet</p>
      ) : (
        <ul className="min-h-0 flex-1 divide-y divide-slate-100 overflow-y-auto">
          {docs.map((doc) => (
            <li key={doc.id} className="group flex items-start gap-3 px-4 py-3 hover:bg-slate-50">
              <input
                type="checkbox"
                checked={selected.has(doc.id)}
                onChange={() => onToggle(doc.id)}
                title="Search only in the ticked documents"
                className="mt-1 size-4 rounded accent-indigo-600"
              />
              <button onClick={() => onOpen(doc)} className="min-w-0 flex-1 text-left">
                <p className="flex items-center gap-1.5 truncate text-sm font-medium text-slate-800 group-hover:text-indigo-700">
                  <FileText className="size-3.5 shrink-0 text-slate-400" />
                  <span className="truncate">{doc.filename}</span>
                </p>
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
                  <StatusBadge status={doc.status} />
                  <span>{doc.num_chunks} chunks</span>
                  {jobs[doc.id] && <span>· {jobs[doc.id].parser}</span>}
                </div>
                {doc.error && <p className="mt-1 line-clamp-2 text-xs text-rose-600">{doc.error}</p>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

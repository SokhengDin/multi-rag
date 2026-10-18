import { useState } from "react";
import { pageImageUrl, type Box } from "../lib/api";

interface Props {
  documentId: string;
  page: number;
  boxes?: Box[] | null;
  dpi?: number;
  className?: string;
}

export function PageWithBoxes({ documentId, page, boxes, dpi, className = "" }: Props) {
  const [state, setState] = useState<"loading" | "ready" | "failed">("loading");

  if (state === "failed") {
    return <div className={`flex items-center justify-center bg-slate-50 p-4 text-xs text-slate-400 ${className}`}>Page unavailable</div>;
  }

  return (
    <div className={`relative ${className}`}>
      <img
        src={pageImageUrl(documentId, page, dpi)}
        loading="lazy"
        onLoad={() => setState("ready")}
        onError={() => setState("failed")}
        className={`block w-full bg-white ${state === "loading" ? "min-h-40 animate-pulse bg-slate-100" : ""}`}
      />
      {state === "ready" && boxes?.map(([x0, y0, x1, y1], i) => (
        <div
          key={i}
          className="pointer-events-none absolute rounded-[2px] border-2 border-amber-500 bg-amber-300/25"
          style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` }}
        />
      ))}
    </div>
  );
}

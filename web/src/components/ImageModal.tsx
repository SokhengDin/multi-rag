import { useEffect } from "react";

export function ImageModal({ src, onClose }: { src: string | null; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!src) return null;
  return (
    <div onClick={onClose} className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 p-6">
      <img src={src} className="max-h-full max-w-full rounded-lg bg-white shadow-2xl" />
    </div>
  );
}

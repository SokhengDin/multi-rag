import type { ReactNode } from "react";

const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function highlight(text: string, query: string): ReactNode {
  const terms = [...new Set(query.match(/[\p{L}\p{N}][\p{L}\p{N}.%_-]*/gu) ?? [])].filter((t) => t.length > 1);
  if (!terms.length) return text;

  const pattern = new RegExp(`(${terms.map(escapeRegex).join("|")})`, "giu");
  return text.split(pattern).map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : part));
}

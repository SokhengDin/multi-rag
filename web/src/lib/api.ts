export const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type Modality = "text" | "table" | "image";
export type Parser = "fast" | "docling" | "vlm" | "page";
export type SearchMode = "dense" | "hybrid";
export type DocumentStatus = "pending" | "processing" | "ready" | "failed";
export type JobStatus = "queued" | "running" | "done" | "failed";
/** [x0, y0, x1, y1] as fractions of the page, origin top-left */
export type Box = [number, number, number, number];

export interface Document {
  id: string;
  filename: string;
  mime_type: string;
  source_type: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  num_chunks: number;
  created_at: string;
}

export interface Job {
  id: string;
  document_id: string;
  status: JobStatus;
  parser: Parser;
  attempts: number;
  error: string | null;
  created_at: string;
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_index: number;
  modality: Modality;
  content: string;
  token_count: number | null;
  page: number | null;
  heading_path: string[] | null;
  json_path: string | null;
  embedding_model: string;
  metadata: Record<string, unknown>;
  image_url: string | null;
}

export interface Hit {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  modality: Modality;
  content: string;
  page: number | null;
  heading_path: string[] | null;
  json_path: string | null;
  parser: Parser | null;
  boxes: Box[] | null;
  image_url: string | null;
  score: number;
  dense_rank: number | null;
  fulltext_rank: number | null;
  sparse_rank: number | null;
  rerank_score: number | null;
}

export interface SearchResult {
  query: string;
  mode: SearchMode;
  reranked: boolean;
  took_ms: number;
  hits: Hit[];
}

export interface Filters {
  document_ids?: string[];
  modalities?: Modality[];
  parser?: Parser;
}

export interface TextSearch {
  query: string;
  mode: SearchMode;
  rerank: boolean;
  top_k: number;
  candidates: number;
  rerank_top: number;
  filters: Filters;
}

export const PARSERS: Record<Parser, string> = {
  fast: "Text only. Fastest. Charts and images are skipped.",
  docling: "Layout-aware: headings, and tables as Markdown.",
  vlm: "A VLM describes every chart and figure. Best for values inside charts. Slow.",
  page: "Every page embedded as an image. No VLM.",
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const details = body?.data?.errors?.map((e: { field: string; message: string }) => `${e.field}: ${e.message}`).join("; ");
    throw new Error(details || body?.message || response.statusText);
  }
  return body.data as T;
}

const json = (method: string, data: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

export const api = {
  documents: () => request<Document[]>("/documents?limit=200"),
  jobs: () => request<Job[]>("/jobs?limit=200"),
  chunks: (id: string) => request<Chunk[]>(`/documents/${id}/chunks?limit=500`),

  upload: (file: File, parser: Parser) => {
    const form = new FormData();
    form.append("file", file);
    form.append("parser", parser);
    return request<Job>("/documents", { method: "POST", body: form });
  },
  reingest: (id: string, parser: Parser) => request<Job>(`/documents/${id}/ingest`, json("POST", { parser })),
  remove: (id: string) => request<null>(`/documents/${id}`, { method: "DELETE" }),

  search: (body: TextSearch) => request<SearchResult>("/search", json("POST", body)),
  searchImage: (file: File, query: string, topK: number, candidates: number, filters: Filters) => {
    const form = new FormData();
    form.append("file", file);
    if (query) form.append("query", query);
    form.append("top_k", String(topK));
    form.append("candidates", String(candidates));
    form.append("filters", JSON.stringify(filters));
    return request<SearchResult>("/search/image", { method: "POST", body: form });
  },
};

export const fileUrl = (documentId: string, filename: string, page?: number | null) =>
  `${API_URL}/documents/${documentId}/file${filename.toLowerCase().endsWith(".pdf") && page ? `#page=${page}` : ""}`;

/** Image paths from the API are relative to it */
export const assetUrl = (path: string) => (/^https?:\/\//.test(path) ? path : `${API_URL}${path}`);

export const isPdf =(filename: string) => filename.toLowerCase().endsWith(".pdf");

export const pageImageUrl = (documentId: string, page: number, dpi = 110) => `${API_URL}/documents/${documentId}/pages/${page}?dpi=${dpi}`;

export function scoreLabel(result: Pick<SearchResult, "mode" | "reranked">) {
  if (result.reranked) return "rerank";
  return result.mode === "dense" ? "cosine" : "RRF";
}

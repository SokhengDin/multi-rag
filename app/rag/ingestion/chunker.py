from collections.abc import Iterator
from dataclasses import dataclass, fields
from itertools import groupby

from transformers import PreTrainedTokenizerBase

from app.db.model import Modality
from app.rag.ingestion.parsers import Element


@dataclass
class Chunk(Element):
    token_count : int | None = None


def chunk_elements(
    elements  : list[Element],
    tokenizer : PreTrainedTokenizerBase,
    size      : int,
    overlap   : int,
) -> list[Chunk]:
    """Merge consecutive text in the same section, then split it into windows of `size` tokens.
    Tables and images stay whole, one chunk each."""
    if overlap >= size:
        raise ValueError(f"CHUNK_OVERLAP ({overlap}) must be smaller than CHUNK_SIZE ({size})")

    chunks: list[Chunk] = []

    for _, group in groupby(elements, key=_section_key):
        group = list(group)

        if group[0].modality != Modality.TEXT:
            chunks.extend(Chunk(**_fields(e), token_count=count_tokens(tokenizer, e.content)) for e in group)
            continue

        parts = [e.content.strip() for e in group]
        text  = "\n\n".join(parts)
        spans = []
        cursor = 0
        for part, element in zip(parts, group):
            spans.append((cursor, cursor + len(part), element.boxes or []))
            cursor += len(part) + 2

        for content, token_count, start, end in split_tokens(text, tokenizer, size, overlap):
            boxes = [box for s, e, element_boxes in spans if s < end and e > start for box in element_boxes]
            chunks.append(Chunk(**{**_fields(group[0]), "content": content, "boxes": boxes or None}, token_count=token_count))

    return chunks


def split_tokens(text: str, tokenizer: PreTrainedTokenizerBase, size: int, overlap: int) -> Iterator[tuple[str, int, int, int]]:
    """Windows are cut on token offsets, so each chunk is an exact slice of the original text.
    Yields (content, token_count, start_char, end_char)."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]

    if len(offsets) <= size:
        yield text, len(offsets), 0, len(text)
        return

    for start in range(0, len(offsets), size - overlap):
        window = offsets[start:start + size]
        yield text[window[0][0]:window[-1][1]].strip(), len(window), window[0][0], window[-1][1]

        if start + size >= len(offsets):
            break


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def _section_key(element: Element) -> tuple:
    return element.modality, element.page, tuple(element.heading_path or ()), element.json_path


def _fields(element: Element) -> dict:
    """Shallow, unlike asdict(), so PIL images are passed through, not deep-copied."""
    return {field.name: getattr(element, field.name) for field in fields(Element)}

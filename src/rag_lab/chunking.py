"""Token-bounded source slices, preserving headings and original character offsets."""
from functools import lru_cache
from bisect import bisect_left
import re

from rag_lab.config import PROJECT_ROOT

TOKENIZER_REPO = "Qwen/Qwen3-Embedding-0.6B"
# Public Git commit identifying the vocabulary data, not an access credential.
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CHUNKER_VERSION = "qwen-tokens-v1"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64


def heading_only(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return bool(lines) and all(re.fullmatch(r"#{1,6}[ \t]+.+|={2,6}[ \t]*.*?[ \t]*={2,6}", line) for line in lines)


@lru_cache(maxsize=1)
def load_tokenizer():
    # Only a pinned JSON vocabulary is downloaded; no repository code is loaded.
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    try:
        path = hf_hub_download(
            repo_id=TOKENIZER_REPO, filename="tokenizer.json", revision=QWEN_REVISION,
            token=False, cache_dir=PROJECT_ROOT / ".cache/huggingface",
        )
        tokenizer = Tokenizer.from_file(path)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        return tokenizer
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot load the pinned Qwen tokenizer. Check network access or its local cache.") from error


def sections(content: str) -> list[tuple[int, int, list[str]]]:
    """Recognize Markdown ATX and Wikipedia headings, ignoring fenced code blocks."""
    result, hierarchy = [], []
    start = position = 0
    fence = None
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        markdown = re.match(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", line.rstrip("\r\n")) if fence is None else None
        wiki = re.match(r"^(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*$", line.rstrip("\r\n")) if fence is None else None
        if markdown or wiki:
            match = markdown or wiki
            level = len(match.group(1)) if markdown else len(match.group(1)) - 1
            if position > start and content[start:position].strip():
                if not heading_only(content[start:position]):
                    result.append((start, position, [title for _, title in hierarchy]))
                    start = position
                # Empty parent headings stay with the next substantive section.
            else:
                start = position
            hierarchy = [(depth, title) for depth, title in hierarchy if depth < level]
            hierarchy.append((level, match.group(2).strip()))
        position += len(line)
    if content[start:].strip():
        if result and heading_only(content[start:]):
            previous_start, _, previous_headings = result[-1]
            result[-1] = (previous_start, len(content), previous_headings)
        else:
            result.append((start, len(content), [title for _, title in hierarchy]))
    return result


def token_chunks(content: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP, *, tokenizer=None) -> list[dict]:
    """Prefer paragraph/sentence boundaries while enforcing the tokenizer limit.

    Overlap is a target token count; intact Unicode character boundaries may shift
    it slightly. It never crosses a substantive section boundary. Heading-only
    blocks are attached to adjacent content.
    Offsets are Python character offsets, not byte offsets. Every content value is
    exactly content[start_char:end_char], including internal whitespace.
    """
    if not 1 <= chunk_size <= 8192 or not 0 <= overlap < chunk_size:
        raise ValueError("Use chunk size 1–8192 and overlap between 0 and chunk size - 1.")
    if not content.strip():
        return []
    tokenizer = tokenizer or load_tokenizer()
    result = []
    for section_start, section_end, heading_path in sections(content):
        section = content[section_start:section_end]
        encoding = tokenizer.encode(section, add_special_tokens=False)
        offsets = encoding.offsets
        start = 0
        boundaries = [match.end() for match in re.finditer(r"\r?\n[ \t]*\r?\n|[.!?。！？](?:[\"'”’)]*)\s+", section)]
        while start < len(offsets):
            end = min(len(offsets), start + chunk_size)
            # Choose a nearby natural boundary, not an arbitrary word count.
            if end < len(offsets):
                floor = start + max(1, int(chunk_size * 0.7))
                for candidate in range(end, floor - 1, -1):
                    boundary = bisect_left(boundaries, offsets[candidate - 1][1])
                    next_character = offsets[candidate][0] if candidate < len(offsets) else len(section)
                    if boundary < len(boundaries) and boundaries[boundary] <= next_character:
                        end = candidate
                        break
            start_char = offsets[start][0]
            end_char = offsets[end - 1][1]
            fragment = section[start_char:end_char]
            count = len(tokenizer.encode(fragment, add_special_tokens=False).ids)
            # A Unicode character can span several tokens with identical offsets.
            # Retokenize slices and shrink by whole characters rather than decode
            # partial byte tokens into replacement characters.
            while count > chunk_size and end > start:
                previous_end_char = end_char
                while end > start and offsets[end - 1][1] >= previous_end_char:
                    end -= 1
                if end <= start:
                    raise ValueError("Chunk size is too small to encode a source character intact.")
                end_char = offsets[end - 1][1]
                fragment = section[start_char:end_char]
                count = len(tokenizer.encode(fragment, add_special_tokens=False).ids)
            if fragment.strip():
                result.append({"content": fragment, "start_char": section_start + start_char,
                               "end_char": section_start + end_char, "token_count": count,
                               "heading_path": list(heading_path)})
            if end >= len(offsets):
                break
            next_start = max(start + 1, end - overlap)
            while next_start < len(offsets) and offsets[next_start][0] <= start_char:
                next_start += 1
            # With zero overlap, never repeat a character split across tokens.
            if overlap == 0:
                while next_start < len(offsets) and offsets[next_start][0] < end_char:
                    next_start += 1
            start = next_start
    return result

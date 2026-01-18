"""
Parse Utility Functions

This module provides utility functions for parsing LaTeX papers and managing
the parsing pipeline. Functions are designed to be called from notebooks
to keep the notebook code clean and maintainable.
"""

import hashlib
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional progress bar
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------#
# Fast .bib parsing helpers (no latex_parser_tree dependency)
# ---------------------------------------------------------------------------#

TITLE_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "of",
    "in",
    "for",
    "on",
    "to",
    "with",
    "by",
    "at",
    "from",
    "as",
}
AUTHOR_STOP_TOKENS = {"dr", "prof", "mr", "ms", "mrs", "phd", "jr", "sr", "ii", "iii"}

_BIB_ENTRY_START_RE = re.compile(r"@\w+\s*\{", re.I)
_BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,]+)\s*,", re.I)
_BIB_STRING_RE = re.compile(r"^\s*@string\s*\{", re.I)
_INLINE_COMMENT_RE = re.compile(r"(?<!\\)%.*")

_PUNCT_NO_ID = "".join(ch for ch in string.punctuation if ch not in {":", "/"})
_PUNCT_TABLE = str.maketrans({ch: "" for ch in _PUNCT_NO_ID})


def strip_inline_comment(line: str) -> str:
    return _INLINE_COMMENT_RE.sub("", line)


def iter_bib_entries(path: Path) -> Iterable[str]:
    """Stream .bib entries, removing @string and % comment lines for speed."""
    in_entry = False
    depth = 0
    buf: List[str] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            stripped = raw.lstrip()
            if stripped.startswith("%"):
                continue
            if _BIB_STRING_RE.match(stripped):
                continue
            line = strip_inline_comment(raw)
            if not in_entry:
                if _BIB_ENTRY_START_RE.search(line):
                    in_entry = True
                    buf = [line]
                    depth = line.count("{") - line.count("}")
                    if depth <= 0 and "}" in line:
                        yield "".join(buf)
                        in_entry = False
                        buf = []
                        depth = 0
            else:
                buf.append(line)
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    yield "".join(buf)
                    in_entry = False
                    buf = []
                    depth = 0


def _consume_brace_value(text: str, start: int) -> Tuple[str, int]:
    """Consume a {...} value starting at '{' (supports nested braces)."""
    depth = 0
    i = start + 1
    val_start = i
    while i < len(text):
        ch = text[i]
        if ch == "{" and text[i - 1] != "\\":
            depth += 1
        elif ch == "}" and text[i - 1] != "\\":
            if depth == 0:
                return text[val_start:i], i + 1
            depth -= 1
        i += 1
    return text[val_start:], len(text)


def _consume_quote_value(text: str, start: int) -> Tuple[str, int]:
    """Consume a "..." value starting at quote."""
    i = start + 1
    val_start = i
    while i < len(text):
        ch = text[i]
        if ch == '"' and text[i - 1] != "\\":
            return text[val_start:i], i + 1
        i += 1
    return text[val_start:], len(text)


def _parse_bib_fields(entry: str) -> Dict[str, str]:
    """Parse common BibTeX fields with a lightweight scanner."""
    fields: Dict[str, str] = {}
    m = _BIB_KEY_RE.search(entry)
    if not m:
        return fields
    i = m.end()
    body = entry[i:]
    pos = 0
    while pos < len(body):
        m2 = re.search(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", body[pos:])
        if not m2:
            break
        field = m2.group(1).lower()
        j = pos + m2.end()
        while j < len(body) and body[j].isspace():
            j += 1
        if j >= len(body):
            break
        if body[j] == "{":
            val, j = _consume_brace_value(body, j)
        elif body[j] == '"':
            val, j = _consume_quote_value(body, j)
        else:
            k = j
            while k < len(body) and body[k] not in ",\n":
                k += 1
            val = body[j:k].strip()
            j = k
        fields[field] = " ".join(val.split())
        comma = body.find(",", j)
        if comma == -1:
            break
        pos = comma + 1
    return fields


def _split_authors_full(auth_str: str) -> List[str]:
    if not auth_str:
        return []
    parts = re.split(r"\band\b|,", auth_str, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def _unwrap_latex_commands(text: str, max_passes: int = 5) -> str:
    """Unwrap simple LaTeX commands like \cmd{...} or \cmd[opt]{...} -> ..."""
    if not text:
        return text or ""
    pattern = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^]]*\])?\{([^{}]*)\}")
    for _ in range(max_passes):
        new_text = pattern.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text
    return text


def _cleanup_hierarchy_text(text: str, keep_env: bool = False) -> str:
    """Light cleanup for hierarchy sentences (strip LaTeX wrappers, keep content).
    
    Parse order (important!):
    1. PROTECT INLINE MATH MODE FIRST ($...$) - content inside is preserved
    2. Normalize line breaks (newlines -> space)
    3. Remove spacing commands (\[.3cm], etc.)
    4. Remove begin/end environments
    5. UNWRAP COMMANDS FIRST (\cite{Katz84} -> Katz84, \command{content} -> content)
    6. Remove remaining backslashes (\\, \, etc.)
    7. Remove braces and normalize spaces
    """
    text = text or ""
    
    def _clean(body: str) -> str:
        body = body.replace("\r", " ").replace("\n", " ")
        body = body.replace(r"\n", " ")
        body = re.sub(r"\\\[[^\]]*\]", " ", body)  # remove spacing like \[.3cm]
        body = re.sub(r"\[\.\d+cm\]", " ", body)  # remove spacing like [.3cm] (without backslash)
        if not keep_env:
            body = re.sub(r"\\begin\{[^}]+\}", " ", body)
            body = re.sub(r"\\end\{[^}]+\}", " ", body)
        # STEP 5: UNWRAP COMMANDS FIRST (before removing backslashes!)
        # unwrap citations and references: \cite{Katz84} -> Katz84
        body = re.sub(r"\\cite\w*\{([^}]+)\}", r"\1", body)
        body = re.sub(r"\\ref\w*\{([^}]+)\}", r"\1", body)
        # unwrap other commands: \command{content} -> content
        body = _unwrap_latex_commands(body)
        # STEP 6: Remove remaining backslashes (after unwrapping)
        body = re.sub(r"\\\\+", " ", body)  # remove LaTeX line breaks (\\))
        body = re.sub(r"\\[^a-zA-Z@]", " ", body)  # remove backslash followed by non-letter (like \,\, \\ etc.)
        body = re.sub(r"\\([A-Za-z@]+)", r"\1", body)  # drop leading backslashes before letters
        # STEP 7: Clean up braces and spaces
        body = re.sub(r"[{}]", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        return body
    
    # STEP 1: Protect inline math mode ($...$) first, then clean, then restore
    # Only protect $...$ pattern (inline math), not $$...$$ or other math modes
    math_pattern = r"\$(?:\\.|[^\$\\])+\$"
    stored: List[str] = []
    
    def _repl(match: re.Match) -> str:
        idx = len(stored)
        stored.append(match.group(0))
        return f"__MATH{idx}__"
    
    tmp = re.sub(math_pattern, _repl, text, flags=re.S)
    tmp = _clean(tmp)
    
    for i, orig in enumerate(stored):
        tmp = tmp.replace(f"__MATH{i}__", orig)
    return tmp


def normalize_ref_text(text: str, remove_stop: bool = False) -> str:
    """Lowercase, strip LaTeX-ish noise, remove most punctuation."""
    text = text or ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace(r"\n", " ")
    text = re.sub(r"\\\\+", " ", text)
    # Unwrap simple LaTeX commands like \cite{...}, \ref{...} -> ...
    text = _unwrap_latex_commands(text)
    text = re.sub(r"\\[a-zA-Z@]+", " ", text)  # drop commands
    text = re.sub(r"[{}]", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stop:
        text = " ".join(w.strip(":/") for w in text.split() if w.strip(":/"))
    return text


def _filter_title_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    return [t for t in tokens if t and len(t) > 1 and t not in TITLE_STOPWORDS]


def _author_tokens_from_norm(authors_norm: List[str]) -> List[str]:
    tokens: List[str] = []
    for a in authors_norm:
        parts = [t for t in re.split(r"[:/\s]+", a or "") if t]
        if not parts:
            continue
        has_long = any(len(t) > 2 for t in parts)
        for t in parts:
            if len(t) <= 1:
                continue
            if t in AUTHOR_STOP_TOKENS:
                continue
            if has_long and len(t) <= 2:
                continue
            tokens.append(t)
    return tokens


def _author_tokens_all(authors_norm: List[str]) -> List[str]:
    """All author tokens (including initials) for removal from title tokens."""
    tokens: List[str] = []
    for a in authors_norm:
        parts = [t for t in re.split(r"[:/\s]+", a or "") if t]
        for t in parts:
            if t in AUTHOR_STOP_TOKENS:
                continue
            tokens.append(t)
    return tokens


def _strip_author_tokens(tokens: List[str], authors_norm: List[str]) -> List[str]:
    if not tokens or not authors_norm:
        return tokens
    author_token_set = set(_author_tokens_all(authors_norm))
    if not author_token_set:
        return tokens
    return [t for t in tokens if t not in author_token_set]


def _derive_note_title_tokens(note_norm: str, authors_norm: List[str]) -> List[str]:
    if not note_norm:
        return []
    candidate = re.split(r"\b(19|20)\d{2}\b", note_norm, maxsplit=1)[0]
    candidate = re.split(r"\.\s+", candidate, maxsplit=1)[0]
    candidate = candidate.strip()
    tokens = [t for t in candidate.split() if t]
    if not tokens:
        tokens = [t for t in note_norm.split() if t]
    tokens = _filter_title_tokens(tokens)
    tokens = _strip_author_tokens(tokens, authors_norm)
    return tokens


def parse_bib_entries(path: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not path.exists():
        return entries
    for entry in iter_bib_entries(path):
        m = _BIB_KEY_RE.search(entry)
        if not m:
            continue
        key = m.group(1).strip()
        fields = _parse_bib_fields(entry)
        fields["key"] = key
        fields["raw"] = " ".join(entry.split())
        entries.append(fields)
    return entries


# ---------------------------------------------------------------------------#
# Normalized outputs (used for JSONL aggregation)
# ---------------------------------------------------------------------------#


def save_normalized_bibitems(paper_dir: Path, bib_items: Dict[str, str]) -> None:
    """No-op: normalized outputs are disabled."""
    return None


def build_normalized_bibitems(paper_dir: Path, bib_items: Dict[str, str]) -> List[Dict]:
    """Return normalized bibitems (used for JSONL aggregation)."""
    rows: List[Dict] = []
    if not bib_items:
        return rows

    def _derive_note_title_tokens(note_norm: str, authors_norm: List[str]) -> (List[str], str):
        if not note_norm:
            return [], "note"
        candidate = note_norm
        candidate = re.split(r"\\em", candidate, maxsplit=1)[0]
        candidate = re.split(r"\b(19|20)\d{2}\b", candidate, maxsplit=1)[0]
        candidate = re.split(r"\.\s+", candidate, maxsplit=1)[0]
        candidate = candidate.strip()
        tokens = [t for t in candidate.split() if t]
        if not tokens:
            tokens = [t for t in note_norm.split() if t]
        tokens = _filter_title_tokens(tokens)
        tokens = _strip_author_tokens(tokens, authors_norm)
        return tokens, "note"

    for key, body in sorted(bib_items.items()):
        fields = legacy_parse_bibitem_fields(body, key)
        authors_norm = [normalize_ref_text(a, remove_stop=False) for a in fields.get("authors_full", [])]
        note_norm = normalize_ref_text(body, remove_stop=True)
        title_raw = fields.get("title", "") or ""
        title_clean = normalize_ref_text(title_raw, remove_stop=False)
        title_clean = re.sub(r"[:/]", " ", title_clean)
        title_tokens = [t for t in title_clean.split() if t]
        tokens_source = "title"
        if title_tokens:
            title_tokens = _filter_title_tokens(title_tokens)
            title_tokens = _strip_author_tokens(title_tokens, authors_norm)
        else:
            title_tokens, tokens_source = _derive_note_title_tokens(note_norm, authors_norm)
        author_tokens: List[str] = []
        for a in authors_norm:
            a_clean = re.sub(r"[:/]", " ", a or "")
            author_tokens.extend(t for t in a_clean.split() if len(t) > 1)
        rows.append(
            {
                "paper_id": paper_dir.name,
                "key": key,
                "note": body,
                "author": fields["author"],
                "year": fields["year"],
                "arxiv": fields["arxiv"],
                "authors_norm": authors_norm,
                "note_norm": note_norm,
                "title_tokens": title_tokens,
                "author_tokens": author_tokens,
                "tokens_source": tokens_source,
            }
        )
    return rows


def _parse_bib_fields(body: str) -> Dict[str, str]:
    """Parse BibTeX fields from an entry body (supports braces, quotes, or bare values)."""
    fields: Dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n or body[i] == "}":
            break
        name_start = i
        while i < n and (body[i].isalnum() or body[i] in "_-"):
            i += 1
        name = body[name_start:i].strip().lower()
        if not name:
            i += 1
            continue
        while i < n and body[i].isspace():
            i += 1
        if i >= n or body[i] != "=":
            i += 1
            continue
        i += 1
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            break
        if body[i] == "{":
            i += 1
            val_start = i
            depth = 1
            while i < n and depth > 0:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            value = body[val_start:i - 1]
        elif body[i] == "\"":
            i += 1
            val_start = i
            while i < n:
                if body[i] == "\"" and body[i - 1] != "\\":
                    break
                i += 1
            value = body[val_start:i]
            if i < n and body[i] == "\"":
                i += 1
        else:
            val_start = i
            while i < n and body[i] not in ",\r\n":
                i += 1
            value = body[val_start:i].strip()
        fields[name] = value.strip()
        while i < n and body[i] not in ",":
            i += 1
        if i < n and body[i] == ",":
            i += 1
    return fields


def _parse_refs_bib_entries(content: str) -> List[Dict[str, Optional[str]]]:
    """Parse refs.bib content into structured fields."""
    entries: List[Dict[str, Optional[str]]] = []

    parts = re.split(r"@(?=\w+\{)", content)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        block = "@" + part
        key_match = re.match(r"@(\w+)\{([^,]+),", block, re.S)
        if not key_match:
            continue
        entry_type = key_match.group(1).strip().lower()
        key = key_match.group(2).strip()
        body = block[key_match.end():]
        parsed_fields = _parse_bib_fields(body)
        note_full = parsed_fields.get("note", "")
        fields: Dict[str, Optional[str]] = {
            "key": key,
            "entry_type": entry_type,
            "title": "",
            "author": "",
            "year": "",
            "doi": "",
            "arxiv": "",
            "pages": "",
            "raw": note_full,
        }
        for fkey, fval in parsed_fields.items():
            if fkey == "archiveprefix":
                continue
            if fkey == "eprint":
                fields["arxiv"] = fval
            elif fkey in fields:
                fields[fkey] = fval
        author_field = fields.get("author", "") or ""
        if "\n" in author_field:
            author_field = author_field.splitlines()[0].strip(" .,;")
        if len(author_field) < 5 or "\\" in author_field:
            raw_lines = (fields.get("raw", "") or "").splitlines()
            raw_line = raw_lines[0] if raw_lines else ""
            if raw_line:
                author_guess = re.split(r",", raw_line, maxsplit=1)[0]
                author_field = author_guess.strip(" .,;")
        fields["author"] = author_field
        fields["authors_full"] = _split_authors_full(author_field)
        entries.append(fields)
    return entries


def save_normalized_bibitems_from_refs(paper_dir: Path, refs_bib_path: Path) -> None:
    """No-op: normalized outputs are disabled."""
    return None


def build_normalized_bibitems_from_refs(paper_dir: Path, refs_bib_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not refs_bib_path.exists():
        return rows
    try:
        content = refs_bib_path.read_text(encoding="utf-8")
    except OSError:
        return rows
    parsed = _parse_refs_bib_entries(content)

    def _derive_note_title_tokens(note_norm: str, authors_norm: List[str]) -> (List[str], str):
        if not note_norm:
            return [], "note"
        candidate = note_norm
        candidate = re.split(r"\\em", candidate, maxsplit=1)[0]
        candidate = re.split(r"\b(19|20)\d{2}\b", candidate, maxsplit=1)[0]
        candidate = re.split(r"\.\s+", candidate, maxsplit=1)[0]
        candidate = candidate.strip()
        tokens = [t for t in candidate.split() if t]
        if not tokens:
            tokens = [t for t in note_norm.split() if t]
        tokens = _filter_title_tokens(tokens)
        tokens = _strip_author_tokens(tokens, authors_norm)
        return tokens, "note"

    for entry in parsed:
        entry_type = (entry.get("entry_type") or "").lower()
        if entry_type and entry_type not in {"article", "misc", "inproceedings"}:
            continue
        authors_norm = [normalize_ref_text(a, remove_stop=False) for a in entry.get("authors_full", []) or []]
        note_val = entry.get("raw", "") or entry.get("note", "") or ""
        note_norm = normalize_ref_text(note_val, remove_stop=True)
        title_raw = entry.get("title", "") or ""
        title_clean = normalize_ref_text(title_raw, remove_stop=False)
        title_clean = re.sub(r"[:/]", " ", title_clean)
        title_tokens = [t for t in title_clean.split() if t]
        tokens_source = "title"
        if title_tokens:
            title_tokens = _filter_title_tokens(title_tokens)
            title_tokens = _strip_author_tokens(title_tokens, authors_norm)
        else:
            title_tokens, tokens_source = _derive_note_title_tokens(note_norm, authors_norm)
        author_tokens: List[str] = []
        for a in authors_norm:
            a_clean = re.sub(r"[:/]", " ", a or "")
            author_tokens.extend(t for t in a_clean.split() if len(t) > 1)
        rows.append(
            {
                "paper_id": paper_dir.name,
                "key": entry.get("key", ""),
                "note": note_val,
                "author": entry.get("author", "") or "",
                "year": entry.get("year", "") or "",
                "arxiv": entry.get("arxiv", "") or "",
                "authors_norm": authors_norm,
                "note_norm": note_norm,
                "title_tokens": title_tokens,
                "author_tokens": author_tokens,
                "tokens_source": tokens_source,
            }
        )
    return rows


def _extract_year_from_date(date_str: str) -> str:
    if not date_str:
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", date_str)
    return m.group(0) if m else ""


def save_normalized_references(paper_dir: Path) -> None:
    """No-op: normalized outputs are disabled."""
    return None


def build_normalized_references(paper_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    ref_path = paper_dir / "references.json"
    if not ref_path.exists():
        return rows
    try:
        data = json.loads(ref_path.read_text(encoding="utf-8"))
    except Exception:
        return rows
    for rid, meta in data.items():
        title = meta.get("paper_title", "") or ""
        authors_raw: List[str] = meta.get("authors", []) or []
        authors_norm = [normalize_ref_text(a, remove_stop=False) for a in authors_raw]
        title_clean = normalize_ref_text(title, remove_stop=False)
        title_clean = re.sub(r"[:/]", " ", title_clean)
        title_tokens = [t for t in title_clean.split() if t]
        author_tokens: List[str] = []
        for a in authors_norm:
            a_clean = re.sub(r"[:/]", " ", a or "")
            author_tokens.extend(t for t in a_clean.split() if len(t) > 1)
        year = _extract_year_from_date(meta.get("submission_date", "") or "")
        note_source = f"{' '.join(authors_norm)} {normalize_ref_text(title, remove_stop=False)}"
        note_norm = normalize_ref_text(note_source, remove_stop=True)
        rows.append(
            {
                "paper_id": paper_dir.name,
                "id": rid,
                "title": title,
                "authors": authors_raw,
                "authors_norm": authors_norm,
                "title_tokens": title_tokens,
                "author_tokens": author_tokens,
                "year": year,
                "arxiv": rid,
                "note_norm": note_norm,
            }
        )
    return rows


# ---------------------------------------------------------------------------#
# Legacy parsing logic (ported from latex_parser_tree.py)
# ---------------------------------------------------------------------------#

LEGACY_BIB_RE = re.compile(
    r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
    re.S | re.IGNORECASE,
)
LEGACY_BIB_KEY_RE = re.compile(r"@\w+\{([^,]+),", re.I)


def legacy_strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def legacy_strip_reference_blocks(text: str) -> str:
    text = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        "",
        text,
        flags=re.S | re.IGNORECASE,
    )
    text = re.split(r"\\end\{document\}", text, flags=re.IGNORECASE)[0]
    return text


def legacy_extract_bibitems(text: str):
    items = []
    for m in LEGACY_BIB_RE.finditer(text):
        key = m.group(1).strip()
        body = legacy_normalize_spaces(m.group(2))
        if key:
            items.append((key, body))
    return items


def legacy_merge_bibitem_fields(base: Dict[str, Optional[str]], other: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    merged = dict(base)
    for field in ("title", "author", "year", "doi", "arxiv", "pages"):
        if not merged.get(field) and other.get(field):
            merged[field] = other[field]
    for field in ("raw", "note"):
        base_val = merged.get(field) or ""
        other_val = other.get(field) or ""
        if len(other_val) > len(base_val):
            merged[field] = other_val
    base_authors = merged.get("authors_full") or []
    other_authors = other.get("authors_full") or []
    if base_authors or other_authors:
        merged["authors_full"] = list(dict.fromkeys(base_authors + other_authors))
    return merged


def legacy_remap_cite_keys(text: str, key_map: Dict[str, str]) -> str:
    if not key_map:
        return text

    def _replace(match: re.Match) -> str:
        raw_keys = match.group(1)
        parts = [p.strip() for p in raw_keys.split(",") if p.strip()]
        if not parts:
            return match.group(0)
        mapped = [key_map.get(p, p) for p in parts]
        return match.group(0).replace(raw_keys, ", ".join(mapped))

    return re.sub(r"\\cite\w*\{([^}]+)\}", _replace, text)


def _legacy_parse_authors(auth_str: str) -> List[str]:
    if not auth_str:
        return []
    parts = re.split(r"\band\b|,", auth_str, flags=re.I)
    last_names: List[str] = []
    for p in parts:
        name = p.strip()
        if not name:
            continue
        tokens = name.split()
        if tokens:
            last_names.append(tokens[-1])
    return last_names


def _legacy_split_authors_full(auth_str: str) -> List[str]:
    if not auth_str:
        return []
    parts = re.split(r"\band\b|,", auth_str, flags=re.I)
    names: List[str] = []
    for p in parts:
        name = p.strip()
        if name:
            names.append(name)
    return names


def _legacy_fallback_author_from_key(key: str) -> str:
    if not key:
        return ""
    m = re.match(r"[A-Za-z]+", key)
    cand = m.group(0) if m else key
    cand = cand.replace("_", " ")
    cand = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cand)
    return cand.strip()


def legacy_parse_bibitem_fields(body: str, key: str = "", prefer_author_from_raw: bool = False) -> Dict[str, Optional[str]]:
    cleaned = legacy_cleanup_formatting(body.replace("\\newblock", " "))
    cleaned = legacy_normalize_spaces(cleaned)

    doi_match = re.search(r"\b10\.\d{4,9}/\S+\b", cleaned, flags=re.I)
    arxiv_match = re.search(r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", cleaned, flags=re.I)
    year_match = re.search(r"\b(19|20)\d{2}\b", cleaned)

    title = None
    author = None
    m_title = re.search(r'["“](.+?)["”]', cleaned)
    if m_title:
        title = m_title.group(1).strip()
        author = cleaned[: m_title.start()].strip(" ,.;:-")
    else:
        if prefer_author_from_raw:
            head = cleaned.splitlines()[0]
            seg = re.split(r"\band\b|,", head, maxsplit=1, flags=re.I)[0]
            author = seg.strip(" ,.;:-.")
            title = ""
        else:
            title = cleaned
            author = ""

    if not author:
        author = _legacy_fallback_author_from_key(key)

    authors_last = _legacy_parse_authors(author)
    authors_full = _legacy_split_authors_full(author)

    return {
        "title": title or "",
        "author": author or "",
        "year": year_match.group(0) if year_match else "",
        "doi": doi_match.group(0) if doi_match else "",
        "arxiv": arxiv_match.group(1) if arxiv_match else "",
        "authors_last": authors_last,
        "authors_full": authors_full,
        "raw": cleaned,
        "pages": re.search(r"\b(\d{1,4}\s*[–-]{1,2}\s*\d{1,4})\b", cleaned).group(1)
        if re.search(r"\b(\d{1,4}\s*[–-]{1,2}\s*\d{1,4})\b", cleaned)
        else "",
    }


def legacy_format_bib_entry_fields(key: str, fields: Dict[str, Optional[str]]) -> str:
    lines = [f"@misc{{{key},"]
    if fields.get("title"):
        lines.append(f"  title = {{{fields['title']}}},")
    if fields.get("author"):
        lines.append(f"  author = {{{fields['author']}}},")
    if fields.get("year"):
        lines.append(f"  year = {{{fields['year']}}},")
    if fields.get("doi"):
        lines.append(f"  doi = {{{fields['doi']}}},")
    if fields.get("arxiv"):
        lines.append(f"  eprint = {{{fields['arxiv']}}},")
        lines.append("  archivePrefix = {arXiv},")
    if fields.get("pages"):
        lines.append(f"  pages = {{{fields['pages']}}},")
    if fields.get("raw"):
        lines.append(f"  note = {{{fields['raw']}}},")
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}\n")
    return "\n".join(lines)


def legacy_protect_math(text: str, cleaner) -> str:
    math_patterns = [
        r"\$\$[\s\S]*?\$\$",
        r"\\\[[\s\S]*?\\\]",
        r"\\\(.*?\\\)",
        r"\$(?:\\.|[^\$\\])+\$",
        r"\\begin\{(?P<env>align\*?|gather\*?|equation\*?|multline\*?|flalign\*?|alignat\*?|eqnarray\*?|displaymath)\}[\s\S]*?\\end\{(?P=env)\}",
    ]
    stored: List[str] = []

    def _repl(match: re.Match) -> str:
        idx = len(stored)
        stored.append(match.group(0))
        return f"__MATH{idx}__"

    tmp = text
    for pat in math_patterns:
        tmp = re.sub(pat, _repl, tmp, flags=re.S)

    tmp = cleaner(tmp)

    for i, orig in enumerate(stored):
        tmp = tmp.replace(f"__MATH{i}__", orig)
    return tmp


def legacy_cleanup_formatting(text: str) -> str:
    replacements = [
        r"\\centering",
        r"\\raggedright",
        r"\\raggedleft",
        r"\\hfill",
        r"\\linebreak",
        r"\\pagebreak",
        r"\\newpage",
        r"\\clearpage",
        r"\\midrule",
        r"\\toprule",
        r"\\bottomrule",
        r"\\hline",
        r"\\vspace\{[^}]*\}",
        r"\\hspace\{[^}]*\}",
        r"\[[htpb!]+\]",
        r"\\noindent",
        r"\\\\+",
    ]

    def _clean(body: str) -> str:
        cleaned = body
        for pat in replacements:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\\s*\n\s*", "\n", cleaned)
        cleaned = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\btextbf\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\emph\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\bemph\{([^}]*)\}", r"\1", cleaned)
        # unwrap citation and reference commands: \cite{Katz84} -> Katz84
        cleaned = re.sub(r"\\cite\w*\{([^}]+)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\ref\w*\{([^}]+)\}", r"\1", cleaned)
        # unwrap other common commands with content: \command{content} -> content
        cleaned = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^]]*\])?\{([^{}]+)\}", r"\1", cleaned)
        cleaned = re.sub(
            r"(?is)keywords:\s*(.+?)\s+msc 2020 subject classifications:",
            r"Keywords: \1. MSC 2020 subject classifications:",
            cleaned,
        )
        cleaned = re.sub(r"(?i)(?<![\.\?!])\s+(keywords:)", r". \1", cleaned)
        return cleaned

    return legacy_protect_math(text, _clean)


def legacy_normalize_spaces(text: str) -> str:
    text = legacy_strip_comments(text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("\n", " ")  # replace remaining newlines with space
    return text.strip()


def legacy_normalize_math(text: str) -> str:
    standardized = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.S)
    standardized = re.sub(r"\\\[(.*?)\\\]", r"\\begin{equation}\1\\end{equation}", standardized, flags=re.S)
    standardized = re.sub(r"\$\$(.*?)\$\$", r"\\begin{equation}\1\\end{equation}", standardized, flags=re.S)
    standardized = re.sub(
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        r"\\begin{equation}\1\\end{equation}",
        standardized,
        flags=re.S,
    )
    standardized = re.sub(
        r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}",
        r"\\begin{equation}\1\\end{equation}",
        standardized,
        flags=re.S,
    )
    standardized = re.sub(
        r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}",
        r"\\begin{equation}\1\\end{equation}",
        standardized,
        flags=re.S,
    )
    return standardized


def legacy_normalize_ref_text(text: str, remove_stop: bool = False) -> str:
    text = legacy_cleanup_formatting(text)
    text = legacy_normalize_spaces(text)
    text = text.replace("\n", " ")
    text = text.replace(r"\n", " ")
    text = re.sub(r"\\\\+", " ", text)
    # Unwrap simple LaTeX commands like \cite{...}, \ref{...} -> ...
    text = _unwrap_latex_commands(text)
    text = re.sub(r"\\[a-zA-Z@]+", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stop:
        text = " ".join(w.strip(":/") for w in text.split() if w.strip(":/"))
    return text


def legacy_clean_figure_text(body: str) -> str:
    cleaned = re.sub(r"\\includegraphics\[.*?\]\{.*?\}", "", body, flags=re.S)
    cap = re.search(r"\\caption\{(.*?)\}", cleaned, flags=re.S)
    if cap:
        cleaned = cap.group(1)
    cleaned = re.sub(r"\\label\{[^}]*\}", "", cleaned)
    cleaned = legacy_cleanup_formatting(cleaned)
    cleaned = legacy_normalize_spaces(cleaned)
    return cleaned


def legacy_clean_equation_text(body: str) -> str:
    cleaned = re.sub(r"\\nonumber\b", "", body)
    cleaned = legacy_cleanup_formatting(cleaned)
    cleaned = legacy_normalize_spaces(cleaned)
    return cleaned


def legacy_split_sentences(text: str) -> List[str]:
    abbreviations = ["e.g.", "i.e.", "etc.", "vs.", "fig.", "figs.", "sec.", "secs.", "eq.", "eqs.", "dr.", "mr.", "ms.", "prof."]
    initials: List[str] = []

    def _init_repl(match: re.Match) -> str:
        idx = len(initials)
        initials.append(match.group(0))
        return f"__INIT{idx}__"

    tmp = re.sub(r"\b[A-Z]\.(?=\s+[A-Z])", _init_repl, text)
    for i, abbr in enumerate(abbreviations):
        tmp = tmp.replace(abbr, f"__ABBR{i}__")
    parts = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z]|__INIT\d+__))", tmp)
    restored = []
    for part in parts:
        for i, val in enumerate(initials):
            part = part.replace(f"__INIT{i}__", val)
        for i, abbr in enumerate(abbreviations):
            part = part.replace(f"__ABBR{i}__", abbr)
        restored.append(part.strip())
    return [p for p in restored if p]


LEGACY_MAIN_CANDIDATES = [
    "main.tex",
    "paper.tex",
    "ms.tex",
    "manuscript.tex",
    "article.tex",
    "root.tex",
]


def legacy_sanitize_path(name: str) -> List[str]:
    parts = Path(name.lstrip("./")).parts
    return [p for p in parts if p not in ("", ".", "..")]


def legacy_resolve_include_path(base_dir: Path, current_dir: Path, raw_path: str) -> Optional[Path]:
    candidate = raw_path.strip()
    if not candidate:
        return None
    if not candidate.endswith(".tex"):
        candidate = f"{candidate}.tex"
    rel_candidate = current_dir / candidate
    abs_candidate = base_dir / candidate
    for cand in (rel_candidate, abs_candidate):
        if cand.exists():
            return cand.resolve()
    return None


def legacy_normalize_include_token(token: str) -> str:
    token = token.strip().strip("{}")
    if not token:
        return ""
    token = token.replace("\\", "/")
    if token.endswith(".tex"):
        token = token[:-4]
    parts = [p for p in token.split("/") if p and p not in (".", "..")]
    return "/".join(parts)


def legacy_parse_includeonly(raw: str) -> Optional[set]:
    m = re.search(r"\\includeonly\{([^}]*)\}", raw, flags=re.I)
    if not m:
        return None
    raw_list = m.group(1)
    items = [legacy_normalize_include_token(x) for x in raw_list.split(",")]
    items = [x for x in items if x]
    return set(items) if items else None


def legacy_is_include_allowed(raw_path: str, include_only: Optional[set]) -> bool:
    if include_only is None:
        return True
    norm = legacy_normalize_include_token(raw_path)
    if not norm:
        return False
    if norm in include_only:
        return True
    base = norm.split("/")[-1]
    return base in include_only


def legacy_inline_includes(path: Path, base_dir: Path, visited: set, include_only: Optional[set] = None) -> str:
    if path in visited:
        return ""
    visited.add(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = legacy_strip_comments(raw)
    if include_only is None:
        include_only = legacy_parse_includeonly(raw)
    raw = re.sub(r"\\includeonly\{[^}]*\}", "", raw, flags=re.I)

    def _replace_basic(match: re.Match) -> str:
        cmd = (match.group(1) or "").lower()
        inc_path = match.group(2)
        if cmd == "include" and not legacy_is_include_allowed(inc_path, include_only):
            return ""
        resolved = legacy_resolve_include_path(base_dir, path.parent, inc_path)
        if resolved:
            return legacy_inline_includes(resolved, base_dir, visited, include_only)
        return ""

    def _replace_import(match: re.Match) -> str:
        imp_dir = match.group(1)
        imp_file = match.group(2)
        imp_path = str(Path(imp_dir) / imp_file)
        resolved = legacy_resolve_include_path(base_dir, path.parent, imp_path)
        if resolved:
            return legacy_inline_includes(resolved, base_dir, visited, include_only)
        return ""

    combined = re.sub(r"\\(input|include|subfile)\{([^}]+)\}", _replace_basic, raw)
    combined = re.sub(r"\\import\{([^}]+)\}\{([^}]+)\}", _replace_import, combined)
    return combined


def legacy_find_main_tex(tex_dir: Path) -> Optional[Path]:
    tex_files = list(tex_dir.rglob("*.tex"))
    if not tex_files:
        return None

    prioritized = [tex_dir / name for name in LEGACY_MAIN_CANDIDATES if (tex_dir / name).exists()]
    if prioritized:
        return prioritized[0]

    for tf in tex_files:
        try:
            head = tf.read_text(encoding="utf-8", errors="ignore")[:2000]
            if re.search(r"\\(input|include|subfile)\{", head) or re.search(r"\\import\{[^}]+\}\{[^}]+\}", head):
                return tf
        except OSError:
            continue

    for tf in tex_files:
        try:
            head = tf.read_text(encoding="utf-8", errors="ignore")[:2000]
            if "\\documentclass" in head:
                return tf
        except OSError:
            continue
    return None


class LegacyIdGen:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.sent = 0
        self.eq = 0
        self.para = 0
        self.sec = 0
        self.subsec = 0
        self.subsubsec = 0
        self.sec_stack: List[str] = []
        self.fig = 0
        self.list = 0
        self.item = 0
        self.cid_map: Dict[tuple[str, str], str] = {}
        self.cid_count = 0

    def root(self) -> str:
        return f"{self.paper_id}::root"

    @staticmethod
    def _h(text: str, length: int = 6) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]

    def _next_with_dedup(self, prefix: str, counter_attr: str, norm_key: str) -> str:
        norm = legacy_normalize_spaces(legacy_cleanup_formatting(legacy_normalize_math(norm_key))).lower()
        key = (prefix, norm)
        if key in self.cid_map:
            return self.cid_map[key]
        setattr(self, counter_attr, getattr(self, counter_attr) + 1)
        idx = getattr(self, counter_attr)
        cid = f"{self.paper_id}::{prefix}_{idx:03d}::{self._h(norm)}"
        self.cid_map[key] = cid
        return cid

    def next_sentence(self, text: str) -> str:
        return self._next_with_dedup("sent", "sent", text)

    def next_equation(self, text: str) -> str:
        return self._next_with_dedup("eq", "eq", text)

    def next_figure(self, text: str) -> str:
        return self._next_with_dedup("fig", "fig", text)

    def next_list(self, env: str, body: str) -> str:
        return self._next_with_dedup("list", "list", f"{env}:{body}")

    def next_item(self, text: str) -> str:
        return self._next_with_dedup("item", "item", text)

    def next_section(self, level: int, title: str) -> str:
        sid, _ = self.push_section(level, title)
        return sid

    def next_paragraph(self) -> str:
        self.para += 1
        return f"{self.paper_id}::para_{self.para:03d}::{self._h(str(self.para))}"

    def push_section(self, level: int, title: str):
        if level == 1:
            prefix = "sec"
        else:
            prefix = f"{'sub' * (level - 1)}sec"
        if not hasattr(self, prefix):
            setattr(self, prefix, 0)
        sec_id = self._next_with_dedup(prefix, prefix, f"L{level}:{title}")
        while len(self.sec_stack) < level - 1:
            self.sec_stack.append(self.root())
        self.sec_stack = self.sec_stack[: level - 1]
        parent_id = self.root() if level == 1 else self.sec_stack[level - 2]
        self.sec_stack.append(sec_id)
        return sec_id, parent_id

    def push_paragraph(self, level: int, title: str):
        para_id = self._next_with_dedup("para", "para", f"L{level}:{title}")
        while len(self.sec_stack) < level - 1:
            self.sec_stack.append(self.root())
        self.sec_stack = self.sec_stack[: level - 1]
        parent_id = self.root() if level == 1 else self.sec_stack[level - 2]
        self.sec_stack.append(para_id)
        return para_id, parent_id


LEGACY_SEC_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]*)\}",
    re.I,
)
LEGACY_BLOCK_RE = re.compile(
    r"(\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|alignat\*?|eqnarray\*?|displaymath|figure\*?|table\*?)\}(?P<body>.*?)\\end\{(?P=env)\}"
    r"|\\\[(?P<disp1>.*?)\\\]"
    r"|\$\$(?P<disp2>.*?)\$\$)",
    re.S | re.I,
)
LEGACY_LIST_RE = re.compile(
    r"\\begin\{(?P<env>itemize|enumerate|description)\}(?P<body>.*?)\\end\{(?P=env)\}",
    re.S | re.I,
)
LEGACY_PARA_RE = re.compile(r"\\(paragraph|subparagraph)\*?\{([^}]*)\}", re.I)


def legacy_parse_tex(text: str, paper_id: str, ids: LegacyIdGen):
    elements: Dict[str, str] = {}
    hierarchy: Dict[str, List[str]] = {}

    title_match = re.search(r"\\title\{([^}]*)\}", text, flags=re.I | re.S)
    title_text = legacy_normalize_spaces(title_match.group(1)) if title_match else ""

    parts = re.split(r"\\begin\{document\}", text, flags=re.I, maxsplit=1)
    if len(parts) > 1:
        text = parts[1]

    abstract_text = ""
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.S | re.I)
    if abs_match:
        abstract_text = legacy_normalize_math(legacy_cleanup_formatting(abs_match.group(1)))
        text = text[:abs_match.start()] + text[abs_match.end():]

    text = re.sub(r"\\maketitle", "", text, flags=re.I)
    text = re.sub(r"\\author\{[^}]*\}", "", text, flags=re.S | re.I)
    text = re.sub(r"\\date\{[^}]*\}", "", text, flags=re.S | re.I)
    text = re.sub(r"\\thanks\{[^}]*\}", "", text, flags=re.S | re.I)
    text = re.sub(r"\\label\{[^}]*\}", "", text, flags=re.S | re.I)
    text = legacy_cleanup_formatting(text)
    text = legacy_normalize_math(text)

    first_sec = LEGACY_SEC_RE.search(text)
    if first_sec:
        text = text[first_sec.start():]

    root_id = ids.root()
    elements[root_id] = "Document"
    hierarchy[root_id] = []

    if title_text:
        title_id = ids.next_sentence(title_text)
        elements[title_id] = title_text
        legacy_append_child(hierarchy, root_id, title_id)

    matches = list(LEGACY_SEC_RE.finditer(text))

    abstract_id = None
    if abstract_text:
        abstract_id, abs_parent = ids.push_section(1, "Abstract")
        elements[abstract_id] = "Abstract"
        legacy_append_child(hierarchy, abs_parent, abstract_id)

    if not matches:
        legacy_add_body(text, ids, elements, hierarchy, abstract_id or root_id)
        if abstract_id:
            legacy_add_body(abstract_text, ids, elements, hierarchy, abstract_id)
        return elements, hierarchy

    first_start = matches[0].start()
    if first_start > 0:
        legacy_add_body(text[:first_start], ids, elements, hierarchy, abstract_id or root_id)

    if abstract_id and abstract_text:
        legacy_add_body(abstract_text, ids, elements, hierarchy, abstract_id)

    for idx, m in enumerate(matches):
        cmd, title = m.group(1), legacy_normalize_spaces(m.group(2))
        if re.search(r"\breferences?\b|bibliography", title, flags=re.I):
            continue
        level = {
            "part": 1,
            "chapter": 1,
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "paragraph": 4,
            "subparagraph": 5,
        }[cmd.lower()]
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]

        if cmd.lower() in ("paragraph", "subparagraph"):
            node_id, parent_id = ids.push_paragraph(level, title)
        else:
            node_id, parent_id = ids.push_section(level, title)
        elements[node_id] = title
        legacy_append_child(hierarchy, parent_id, node_id)
        legacy_add_body(body, ids, elements, hierarchy, node_id)

    return elements, hierarchy


def legacy_append_child(h: Dict[str, List[str]], parent: str, child: str):
    if parent not in h:
        h[parent] = []
    h[parent].append(child)


def legacy_hierarchy_to_child_parent(tree: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for parent, children in tree.items():
        for child in children:
            out[child] = parent
    return out


def legacy_order_child_parent(child_parent: Dict[str, str], root_id: str) -> Dict[str, str]:
    parent_children: Dict[str, List[str]] = defaultdict(list)
    for child, parent in child_parent.items():
        parent_children[parent].append(child)
    for v in parent_children.values():
        v.sort()

    ordered: List[tuple[str, str]] = []
    seen: set = set()

    queue: List[str] = [root_id]
    while queue:
        parent = queue.pop(0)
        for child in parent_children.get(parent, []):
            ordered.append((child, parent))
            if child not in seen:
                seen.add(child)
                queue.append(child)

    for child, parent in sorted(child_parent.items(), key=lambda kv: (kv[1], kv[0])):
        if child not in seen:
            ordered.append((child, parent))

    return dict(ordered)


def legacy_add_list_items(body: str, ids: LegacyIdGen, elements: Dict[str, str], hierarchy: Dict[str, List[str]], parent: str):
    parts = re.split(r"\\item(?:\s*\[[^\]]*\])?", body)
    for part in parts:
        item_body = part.strip()
        if not item_body:
            continue
        item_id = ids.next_item(item_body)
        clean_text = legacy_normalize_spaces(legacy_cleanup_formatting(item_body))
        elements[item_id] = clean_text
        legacy_append_child(hierarchy, parent, item_id)
        legacy_add_body(item_body, ids, elements, hierarchy, item_id)


def legacy_add_body(body: str, ids: LegacyIdGen, elements: Dict[str, str], hierarchy: Dict[str, List[str]], parent: str):
    def process_segment_no_lists(seg: str, allow_sentences: bool):
        cursor = 0
        for m in LEGACY_BLOCK_RE.finditer(seg):
            if m.start() > cursor and allow_sentences:
                legacy_add_sentences(seg[cursor:m.start()], ids, elements, hierarchy, parent)
            env = (m.group("env") or "").lower()
            btxt = legacy_normalize_spaces(m.group("body") or m.group("disp1") or m.group("disp2") or "")
            if env in ("figure", "figure*", "table", "table*"):
                clean_txt = legacy_clean_figure_text(btxt)
                fid = ids.next_figure(clean_txt)
                elements[fid] = clean_txt
                legacy_append_child(hierarchy, parent, fid)
            else:
                eq_text = legacy_clean_equation_text(btxt)
                eq_text = _cleanup_hierarchy_text(eq_text, keep_env=True)
                eq_text = f"\\begin{{equation}}{eq_text}\\end{{equation}}"
                eqid = ids.next_equation(eq_text)
                elements[eqid] = eq_text
                legacy_append_child(hierarchy, parent, eqid)
            cursor = m.end()
        if cursor < len(seg) and allow_sentences:
            legacy_add_sentences(seg[cursor:], ids, elements, hierarchy, parent)

    def process_segment(seg: str, allow_sentences: bool):
        cursor = 0
        for m in LEGACY_LIST_RE.finditer(seg):
            if m.start() > cursor:
                process_segment_no_lists(seg[cursor:m.start()], allow_sentences)
            list_env = (m.group("env") or "").lower()
            list_body = m.group("body") or ""
            list_id = ids.next_list(list_env, list_body)
            elements[list_id] = list_env
            legacy_append_child(hierarchy, parent, list_id)
            legacy_add_list_items(list_body, ids, elements, hierarchy, list_id)
            cursor = m.end()
        if cursor < len(seg):
            process_segment_no_lists(seg[cursor:], allow_sentences)

    paras = list(LEGACY_PARA_RE.finditer(body))
    if not paras:
        process_segment(body, allow_sentences=True)
        return

    cursor = 0
    for i, pm in enumerate(paras):
        if pm.start() > cursor:
            process_segment(body[cursor:pm.start()], allow_sentences=True)
        seg_end = paras[i + 1].start() if i + 1 < len(paras) else len(body)
        para_body = body[pm.end():seg_end]
        process_segment(para_body, allow_sentences=True)
        cursor = seg_end
    if cursor < len(body):
        process_segment(body[cursor:], allow_sentences=True)


def legacy_add_sentences(text: str, ids: LegacyIdGen, elements: Dict[str, str], hierarchy: Dict[str, List[str]], parent: str):
    text = _cleanup_hierarchy_text(text)
    for sent in legacy_split_sentences(legacy_normalize_spaces(text)):
        if not sent:
            continue
        sid = ids.next_sentence(sent)
        elements[sid] = sent
        legacy_append_child(hierarchy, parent, sid)


def legacy_run_for_paper(
    paper_dir: Path,
    jsonl_bib: Optional[Path] = None,
    jsonl_refs: Optional[Path] = None,
):
    paper_id = paper_dir.name
    tex_dir = paper_dir / "tex"
    versions: Dict[str, Dict[str, str]] = {}
    elements_all: Dict[str, str] = {}
    bib_items: Dict[str, str] = {}
    refs_entries: List[str] = []
    existing_bib_keys: set = set()
    bibitem_key_by_norm: Dict[str, str] = {}
    bibitem_fields_by_key: Dict[str, Dict[str, Optional[str]]] = {}
    bib_key_map: Dict[str, str] = {}
    ids = LegacyIdGen(paper_id)
    for idx, vdir in enumerate(sorted(tex_dir.iterdir()), 1):
        if not vdir.is_dir():
            continue
        main_tex = legacy_find_main_tex(vdir)
        if not main_tex:
            versions[str(idx)] = {}
            continue
        raw_full = legacy_inline_includes(main_tex, vdir, set(), None)

        for bib_path in vdir.rglob("*.bib"):
            try:
                content = bib_path.read_text(encoding="utf-8", errors="ignore").strip()
                refs_entries.append(content)
                for k in LEGACY_BIB_KEY_RE.findall(content):
                    existing_bib_keys.add(k.strip())
            except OSError:
                pass

        version_refs = legacy_extract_bibitems(raw_full)
        for key, body in version_refs:
            fields = legacy_parse_bibitem_fields(body, key, prefer_author_from_raw=True)
            norm_body = legacy_normalize_ref_text(body, remove_stop=False)
            canonical_key = bibitem_key_by_norm.get(norm_body)
            if canonical_key:
                bib_key_map[key] = canonical_key
                bibitem_fields_by_key[canonical_key] = legacy_merge_bibitem_fields(
                    bibitem_fields_by_key[canonical_key], fields
                )
            else:
                bibitem_key_by_norm[norm_body] = key
                bib_key_map[key] = key
                bibitem_fields_by_key[key] = fields

        raw_full = legacy_remap_cite_keys(raw_full, bib_key_map)

        raw = legacy_strip_reference_blocks(raw_full)
        elems, tree = legacy_parse_tex(raw, paper_id, ids)
        tree = legacy_hierarchy_to_child_parent(tree)
        tree = legacy_order_child_parent(tree, ids.root())
        elements_all.update(elems)
        versions[str(idx)] = tree

    out = {
        "elements": elements_all,
        "hierarchy": versions,
    }
    out_path = paper_dir / "hierarchy.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    bibitem_fields_filtered = {
        key: fields for key, fields in bibitem_fields_by_key.items() if key not in existing_bib_keys
    }
    bib_lines = []
    if refs_entries:
        bib_lines.append("\n\n".join(refs_entries))
    for key in sorted(bibitem_fields_filtered.keys()):
        bib_lines.append(legacy_format_bib_entry_fields(key, bibitem_fields_filtered[key]))
    refs_bib_path = paper_dir / "refs.bib"
    refs_bib_path.write_text("\n".join(bib_lines), encoding="utf-8")

    if (paper_dir / "refs.bib").exists():
        bib_rows = build_normalized_bibitems_from_refs(paper_dir, paper_dir / "refs.bib")
    else:
        bib_rows = build_normalized_bibitems(paper_dir, bib_items)
    ref_rows = build_normalized_references(paper_dir)

    if jsonl_bib:
        _append_jsonl(jsonl_bib, bib_rows)
    if jsonl_refs:
        _append_jsonl(jsonl_refs, ref_rows)


def build_refs_bib(
    paper_dir: Path,
    output_path: Optional[Path] = None,
    clean: bool = True,
    overwrite: bool = False,
) -> Optional[Path]:
    """Create a merged refs.bib for a paper from tex/*.bib files."""
    if output_path is None:
        output_path = paper_dir / "refs.bib"
    if output_path.exists() and not overwrite:
        return output_path

    tex_dir = paper_dir / "tex"
    if not tex_dir.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return output_path

    bib_paths = sorted(tex_dir.rglob("*.bib"))
    if not bib_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        first = True
        for bp in bib_paths:
            if bp.resolve() == output_path.resolve():
                continue
            if clean:
                for entry in iter_bib_entries(bp):
                    entry = entry.strip()
                    if not entry:
                        continue
                    if not first:
                        out.write("\n")
                    out.write(entry + "\n")
                    first = False
            else:
                text = bp.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                if not first:
                    out.write("\n")
                out.write(text + "\n")
                first = False
    return output_path


def generate_hierarchy_json(paper_dir: Path, overwrite: bool = False) -> Optional[Path]:
    """Generate hierarchy.json for a paper using latex_parser_tree logic."""
    out_path = paper_dir / "hierarchy.json"
    if out_path.exists() and not overwrite:
        return out_path
    tex_dir = paper_dir / "tex"
    if not tex_dir.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"elements": {}, "hierarchy": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
    try:
        legacy_run_for_paper(paper_dir)
        return out_path if out_path.exists() else None
    except Exception:
        return None


def build_bib_rows(paper_dir: Path) -> List[Dict]:
    refs_bib = paper_dir / "refs.bib"
    if refs_bib.exists():
        return build_normalized_bibitems_from_refs(paper_dir, refs_bib)

    # Fallback: attempt to build refs.bib from tex/*.bib and normalize from it.
    built = build_refs_bib(paper_dir, output_path=refs_bib, clean=True, overwrite=False)
    if built and built.exists():
        return build_normalized_bibitems_from_refs(paper_dir, built)

    return []


def build_reference_rows(paper_dir: Path) -> List[Dict]:
    return build_normalized_references(paper_dir)


def _append_jsonl(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def setup_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Setup and return paths for parsing pipeline.
    
    Args:
        root: Root directory of the project. If None, resolves from current directory.
        
    Returns:
        Dictionary containing:
        - 'root': Project root directory
        - 'paper_root': Directory containing paper folders
        - 'agg_dir': Aggregated output directory
        - 'bib_path': Path to bibitems.jsonl
        - 'ref_path': Path to references.jsonl
    """
    if root is None:
        root = Path("..").resolve()
    else:
        root = Path(root).resolve()
    
    paper_root = root / "23120334"
    agg_dir = root / "aggregated"
    agg_dir.mkdir(exist_ok=True)
    
    bib_path = agg_dir / "bibitems.jsonl"
    ref_path = agg_dir / "references.jsonl"
    
    return {
        'root': root,
        'paper_root': paper_root,
        'agg_dir': agg_dir,
        'bib_path': bib_path,
        'ref_path': ref_path
    }


def cleanup_output_files(bib_path: Path, ref_path: Path) -> None:
    """
    Remove existing output files if they exist.
    
    Args:
        bib_path: Path to bibitems.jsonl
        ref_path: Path to references.jsonl
    """
    if bib_path.exists():
        bib_path.unlink()
    if ref_path.exists():
        ref_path.unlink()


def get_paper_list(
    paper_root: Path,
    run_all: bool = True,
    paper_id: Optional[str] = None,
    start: Optional[str] = None,
    num: Optional[int] = None
) -> List[Path]:
    """
    Get list of papers to process.
    
    Args:
        paper_root: Root directory containing paper folders
        run_all: If True, process all papers; if False, process only paper_id
        paper_id: Single paper ID to process (used when run_all=False)
        start: Starting paper ID for filtering (used when run_all=True)
        num: Maximum number of papers to process (used when run_all=True)
        
    Returns:
        List of paper directory paths
    """
    if run_all:
        papers = sorted([
            p for p in paper_root.iterdir()
            if p.is_dir() and (p / "tex").exists()
        ])
        if start:
            papers = [p for p in papers if p.name >= start]
        if num is not None:
            papers = papers[:num]
    else:
        if paper_id is None:
            raise ValueError("paper_id must be provided when run_all=False")
        papers = [paper_root / paper_id]
    
    return papers


def run_parser(
    papers: List[Path],
    bib_path: Path,
    ref_path: Path,
    show_progress: bool = True,
    use_legacy: Optional[bool] = None
) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Run parser on a list of papers.
    
    Args:
        papers: List of paper directory paths
        bib_path: Path to output bibitems.jsonl
        ref_path: Path to output references.jsonl
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (processed_count, skipped_list)
        skipped_list contains tuples of (paper_id, reason)
    """
    processed = 0
    skipped = []
    
    if use_legacy is None:
        use_legacy = True
    iterator = tqdm(papers, desc="Parsing papers") if (show_progress and TQDM_AVAILABLE) else papers
    
    total = len(papers)
    index_width = max(2, len(str(total)))
    for i, pdir in enumerate(iterator, 1):
        tex_dir = pdir / "tex"
        refs_bib = pdir / "refs.bib"
        
        if not pdir.exists():
            continue
        
        if not show_progress or not TQDM_AVAILABLE:
            print(f"[{i:{index_width}d}/{total}] {pdir.name:<12} | parsing")

        try:
            if use_legacy and tex_dir.exists():
                legacy_run_for_paper(pdir, jsonl_bib=bib_path, jsonl_refs=ref_path)
            else:
                bib_rows = build_bib_rows(pdir)
                ref_rows = build_reference_rows(pdir)
                _append_jsonl(bib_path, bib_rows)
                _append_jsonl(ref_path, ref_rows)
            processed += 1
        except Exception as e:
            if not show_progress or not TQDM_AVAILABLE:
                print(f"[{i:{index_width}d}/{total}] {pdir.name:<12} | error: {e}")
            skipped.append((pdir.name, f"error: {str(e)}"))

    return processed, skipped


def postprocess_token_fields(path: Path) -> None:
    """Rewrite JSONL to remove stopwords from title_tokens and clean author_tokens."""
    if not path.exists():
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            authors_norm = obj.get("authors_norm") or []
            if isinstance(authors_norm, str):
                authors_norm = [authors_norm]
            authors_norm = [
                normalize_ref_text(str(a), remove_stop=False)
                for a in authors_norm
                if a and str(a).strip()
            ]
            author_tokens = _author_tokens_from_norm(authors_norm)
            if author_tokens:
                obj["author_tokens"] = author_tokens

            title_tokens = obj.get("title_tokens") or []
            if title_tokens:
                cleaned = _filter_title_tokens(title_tokens)
                cleaned = _strip_author_tokens(cleaned, authors_norm)
                obj["title_tokens"] = cleaned

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def compute_statistics(
    bib_path: Path,
    ref_path: Path
) -> Dict:
    """
    Compute statistics from aggregated JSONL files.
    
    Args:
        bib_path: Path to bibitems.jsonl
        ref_path: Path to references.jsonl
        
    Returns:
        Dictionary containing:
        - 'total_papers': Total number of unique papers
        - 'total_bibitems': Total number of bibitems
        - 'total_references': Total number of references
        - 'bib_counts': Counter of bibitems per paper
        - 'ref_counts': Counter of references per paper
        - 'zero_bib': List of papers with no bibitems
        - 'zero_ref': List of papers with no references
    """
    bib_counts = Counter()
    ref_counts = Counter()
    
    if bib_path.exists():
        with bib_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("paper_id")
                    if pid:
                        bib_counts[pid] += 1
                except json.JSONDecodeError:
                    continue
    
    if ref_path.exists():
        with ref_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("paper_id")
                    if pid:
                        ref_counts[pid] += 1
                except json.JSONDecodeError:
                    continue
    
    all_papers = sorted(set(bib_counts) | set(ref_counts))
    zero_bib = [p for p in all_papers if bib_counts[p] == 0]
    zero_ref = [p for p in all_papers if ref_counts[p] == 0]
    
    return {
        'total_papers': len(all_papers),
        'total_bibitems': sum(bib_counts.values()),
        'total_references': sum(ref_counts.values()),
        'bib_counts': bib_counts,
        'ref_counts': ref_counts,
        'zero_bib': zero_bib,
        'zero_ref': zero_ref
    }


def print_statistics(stats: Dict, top_n: int = 5) -> None:
    """
    Print formatted statistics.
    
    Args:
        stats: Statistics dictionary from compute_statistics()
        top_n: Number of top papers to show
    """
    print("=" * 60)
    print("PARSING STATISTICS")
    print("=" * 60)
    print(f"Total papers seen:        {stats['total_papers']}")
    print(f"Total bibitems:           {stats['total_bibitems']:,}")
    print(f"Total references:         {stats['total_references']:,}")
    print(f"Papers with no bibitems:  {len(stats['zero_bib'])}")
    print(f"Papers with no references: {len(stats['zero_ref'])}")
    
    if stats['zero_bib']:
        print(f"\nExample papers missing bibitems: {stats['zero_bib'][:top_n]}")
    if stats['zero_ref']:
        print(f"Example papers missing references: {stats['zero_ref'][:top_n]}")
    
    print(f"\nTop {top_n} papers by bibitems count:")
    for pid, count in stats['bib_counts'].most_common(top_n):
        print(f"  {pid}: {count:,}")
    
    print(f"\nTop {top_n} papers by references count:")
    for pid, count in stats['ref_counts'].most_common(top_n):
        print(f"  {pid}: {count:,}")
    print("=" * 60)


def quick_check(bib_path: Path, ref_path: Path, num_samples: int = 3) -> Dict:
    """
    Quick check of aggregated files.
    
    Args:
        bib_path: Path to bibitems.jsonl
        ref_path: Path to references.jsonl
        num_samples: Number of sample lines to show
        
    Returns:
        Dictionary containing:
        - 'bib_count': Number of lines in bibitems.jsonl
        - 'ref_count': Number of lines in references.jsonl
        - 'bib_samples': Sample lines from bibitems.jsonl
        - 'ref_samples': Sample lines from references.jsonl
    """
    bib_count = sum(1 for _ in bib_path.open()) if bib_path.exists() else 0
    ref_count = sum(1 for _ in ref_path.open()) if ref_path.exists() else 0
    
    bib_samples = []
    if bib_count and bib_path.exists():
        with bib_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                bib_samples.append(line.strip())
    
    ref_samples = []
    if ref_count and ref_path.exists():
        with ref_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                ref_samples.append(line.strip())
    
    return {
        'bib_count': bib_count,
        'ref_count': ref_count,
        'bib_samples': bib_samples,
        'ref_samples': ref_samples
    }


def print_quick_check(check_result: Dict) -> None:
    """
    Print formatted quick check results.
    
    Args:
        check_result: Result dictionary from quick_check()
    """
    print("=" * 60)
    print("QUICK CHECK")
    print("=" * 60)
    print(f"bibitems.jsonl lines:     {check_result['bib_count']:,}")
    print(f"references.jsonl lines:   {check_result['ref_count']:,}")
    
    if check_result['bib_samples']:
        print("\nSample bibitems:")
        for i, sample in enumerate(check_result['bib_samples'], 1):
            print(f"  [{i}] {sample[:100]}..." if len(sample) > 100 else f"  [{i}] {sample}")
    
    if check_result['ref_samples']:
        print("\nSample references:")
        for i, sample in enumerate(check_result['ref_samples'], 1):
            print(f"  [{i}] {sample[:100]}..." if len(sample) > 100 else f"  [{i}] {sample}")
    print("=" * 60)


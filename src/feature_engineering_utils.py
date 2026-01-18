"""
Feature Engineering Utilities

This module provides functions for computing features used in citation matching,
including text similarity metrics, year parsing, and text normalization.
"""

import re
from typing import Dict, List, Optional, Set


def parse_year_int(val: str) -> Optional[int]:
    """Extract a 4-digit year as int; return None if not found/invalid."""
    if not val:
        return None
    m = re.search(r"\d{4}", str(val))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def levenshtein_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0,1]."""
    if a == b:
        return 1.0 if a else 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    dist = dp[lb]
    return 1.0 - dist / max(la, lb)


def jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def token_overlap(tokens_a: List[str], tokens_b: List[str]) -> int:
    """Count overlapping tokens between two token lists (no IDF)."""
    return len(set(tokens_a) & set(tokens_b))


def token_overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute overlap ratio using max length (not Jaccard)."""
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(set(tokens_a) & set(tokens_b))
    denom = max(len(tokens_a), len(tokens_b))
    return inter / denom if denom else 0.0


def _char_ngrams(text: str, n: int) -> Set[str]:
    """Build a set of character n-grams from normalized text."""
    if not text:
        return set()
    cleaned = re.sub(r"\s+", "", text.lower())
    if len(cleaned) < n:
        return set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def char_ngram_overlap(text_a: str, text_b: str, n: int) -> float:
    """Jaccard overlap for character n-grams."""
    a = _char_ngrams(text_a, n)
    b = _char_ngrams(text_b, n)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def digit_overlap(text_a: str, text_b: str) -> int:
    """Count overlapping numeric tokens between two texts."""
    nums_a = set(re.findall(r"\d+", text_a))
    nums_b = set(re.findall(r"\d+", text_b))
    return len(nums_a & nums_b)


def norm_text(entry: Dict) -> str:
    """Extract normalized text from entry, preferring title, fallback to note_norm/note."""
    txt = entry.get("title") or entry.get("note_norm") or entry.get("note") or ""
    return txt.strip()


def tokens_from_entry(entry: Dict) -> List[str]:
    """Extract tokens from entry, preferring title_tokens, fallback to splitting norm_text."""
    tokens = entry.get("title_tokens") or []
    if tokens:
        return tokens
    txt = norm_text(entry)
    return txt.split() if txt else []


def _author_list(entry: Dict) -> List[str]:
    """Extract normalized author list from entry."""
    authors = entry.get("authors_norm") or entry.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    return [str(a).strip().lower() for a in authors if a]


def author_overlap(authors_a: List[str], authors_b: List[str]) -> float:
    """Jaccard overlap between author full names."""
    set_a, set_b = set(authors_a), set(authors_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def author_lastname_match(authors_a: List[str], authors_b: List[str]) -> float:
    """Binary match if any author last name overlaps."""
    def last_name(a: str) -> str:
        parts = a.split()
        return parts[-1] if parts else ""

    set_a = {last_name(a) for a in authors_a if a}
    set_b = {last_name(b) for b in authors_b if b}
    if not set_a or not set_b:
        return 0.0
    return 1.0 if (set_a & set_b) else 0.0


def author_firstname_match(authors_a: List[str], authors_b: List[str]) -> float:
    """Binary match if any author first name overlaps."""
    def first_name(a: str) -> str:
        parts = a.split()
        return parts[0] if parts else ""

    set_a = {first_name(a) for a in authors_a if a}
    set_b = {first_name(b) for b in authors_b if b}
    if not set_a or not set_b:
        return 0.0
    return 1.0 if (set_a & set_b) else 0.0


def compute_features(bib: Dict, ref: Dict) -> Dict:
    """
    Compute all features for a bibitem-reference pair.
    
    Args:
        bib: Bibliography item dictionary
        ref: Reference dictionary
        
    Returns:
        Dictionary containing computed features:
        - levenshtein: Levenshtein similarity
        - jaccard: Jaccard similarity
        - author_overlap: Jaccard overlap of author full names
        - author_lastname_match: Binary match if any last name overlaps
        - author_firstname_match: Binary match if any first name overlaps
        - year_match: 1 if years match, 0 otherwise
        - year_diff: Absolute difference in years, or 100 if missing
        - source_year: Year from bibitem
        - cand_year: Year from reference
    """
    b_txt = norm_text(bib)
    r_txt = norm_text(ref)
    b_tokens = tokens_from_entry(bib)
    r_tokens = tokens_from_entry(ref)
    b_authors = _author_list(bib)
    r_authors = _author_list(ref)
    b_year = parse_year_int(bib.get("year", ""))
    r_year = parse_year_int(ref.get("year", ""))
    
    return {
        "levenshtein": levenshtein_sim(b_txt, r_txt),
        "jaccard": jaccard(b_tokens, r_tokens),
        "token_overlap": token_overlap(b_tokens, r_tokens),
        "token_overlap_ratio": token_overlap_ratio(b_tokens, r_tokens),
        "char_ngram_3": char_ngram_overlap(b_txt, r_txt, 3),
        "char_ngram_4": char_ngram_overlap(b_txt, r_txt, 4),
        "char_ngram_5": char_ngram_overlap(b_txt, r_txt, 5),
        "author_overlap": author_overlap(b_authors, r_authors),
        "author_lastname_match": author_lastname_match(b_authors, r_authors),
        "year_match": 1 if (b_year is not None and r_year is not None and b_year == r_year) else 0,
        "year_diff": abs(b_year - r_year) if (b_year is not None and r_year is not None) else 100,
        "source_year": b_year,
        "cand_year": r_year,
    }


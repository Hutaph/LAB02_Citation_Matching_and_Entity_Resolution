"""
Matching Utilities

This module provides functions for matching bibliography items with references,
data loading, manual candidate handling, and data splitting.
"""

import json
import random
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Optional progress bar
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

from feature_engineering_utils import compute_features, norm_text, tokens_from_entry, parse_year_int, author_overlap, author_lastname_match

# Constants for text normalization
_PUNCT_NO_ID = "".join(ch for ch in string.punctuation if ch not in {":", "/"})
_PUNCT_TABLE = str.maketrans({ch: "" for ch in _PUNCT_NO_ID})


# Text normalization functions (copied from latex_parser_tree to avoid import)
def strip_comments(text: str) -> str:
    """Strip LaTeX comments."""
    return re.sub(r"(?<!\\)%.*", "", text)


def normalize_spaces(text: str) -> str:
    """Normalize whitespace in text."""
    text = strip_comments(text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("\n", " ")  # replace remaining newlines with space
    return text.strip()


def protect_math(text: str, cleaner) -> str:
    """Protect math blocks with placeholders during cleaning, then restore."""
    math_patterns = [
        r"\$\$[\s\S]*?\$\$",  # $$ ... $$
        r"\\\[[\s\S]*?\\\]",  # \[ ... \]
        r"\\\(.*?\\\)",  # \( ... \)
        r"\$(?:\\.|[^\$\\])+\$",  # $ ... $
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


def cleanup_formatting(text: str) -> str:
    """Clean LaTeX formatting from text."""
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
        r"\\\\+",  # remove LaTeX line breaks \\ -> space
        r"\\\[\.\d+cm\]",  # remove spacing like \[.3cm] (with backslash)
        r"\[\.\d+cm\]",  # remove spacing like [.3cm] (without backslash)
    ]

    def _clean(body: str) -> str:
        cleaned = body
        for pat in replacements:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        # collapse escaped linebreak + newline
        cleaned = re.sub(r"\\\s*\n\s*", "\n", cleaned)
        # unwrap common formatting (with or without backslash)
        cleaned = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\btextbf\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\emph\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\bemph\{([^}]*)\}", r"\1", cleaned)
        # unwrap citation and reference commands: \cite{Katz84} -> Katz84
        cleaned = re.sub(r"\\cite\w*\{([^}]+)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\ref\w*\{([^}]+)\}", r"\1", cleaned)
        # unwrap other common commands with content: \command{content} -> content
        cleaned = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^]]*\])?\{([^{}]+)\}", r"\1", cleaned)
        # keywords / MSC lines often split by newline; stitch with period between
        cleaned = re.sub(
            r"(?is)keywords:\s*(.+?)\s+msc 2020 subject classifications:",
            r"Keywords: \1. MSC 2020 subject classifications:",
            cleaned,
        )
        # force keywords/msc to start new sentence if glued
        cleaned = re.sub(r"(?i)(?<![\.\?!])\s+(keywords:)", r". \1", cleaned)
        return cleaned

    return protect_math(text, _clean)


def normalize_ref_text(text: str, remove_stop: bool = False) -> str:
    """Lowercase, strip LaTeX-ish noise, remove most punctuation, optional stopwords."""
    text = cleanup_formatting(text)
    text = normalize_spaces(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\\[a-zA-Z@]+", " ", text)  # drop remaining commands
    text = re.sub(r"[{}]", " ", text)
    # unify common unicode punctuation before accent strip
    text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    text = text.lower()
    # strip accents to ASCII
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stop:
        text = " ".join(w.strip(":/") for w in text.split() if w.strip(":/"))
    return text


def norm_arxiv(val: str) -> str:
    """Normalize arXiv ID by removing non-word characters and lowercasing."""
    if not val:
        return ""
    return re.sub(r"\W+", "", str(val)).lower()


def load_grouped(path: Path) -> Dict[str, List[Dict]]:
    """Load JSONL and group rows by paper_id."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    if not path.exists():
        return grouped
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("paper_id")
            if not pid:
                continue
            grouped[pid].append(obj)
    return grouped


def load_grouped_subset(path: Path, target_ids: Set[str]) -> Dict[str, List[Dict]]:
    """Load JSONL and group rows by paper_id, filtering by target_ids."""
    out: Dict[str, List[Dict]] = defaultdict(list)
    if not path.exists() or not target_ids:
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("paper_id")
            if pid not in target_ids:
                continue
            out[pid].append(obj)
    return out


def load_manual_candidates(manual_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load manual candidate mappings from JSON file.
    
    Args:
        manual_path: Path to manual_candidates.json
        
    Returns:
        Dictionary mapping paper_id -> {bib_key: cand_id}
    """
    if not manual_path.exists():
        return {}
    try:
        return json.loads(manual_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to load manual_candidates.json: {e}")
        return {}


def get_target_paper_ids(
    bib_path: Path,
    manual_map: Dict[str, Dict[str, str]],
    start: Optional[str] = None,
    num: Optional[int] = None
) -> Tuple[Set[str], List[str]]:
    """
    Get target paper IDs based on manual candidates and filtering criteria.
    
    Args:
        bib_path: Path to bibitems.jsonl
        manual_map: Manual candidate mappings
        start: Starting paper ID filter
        num: Maximum number of papers
        
    Returns:
        Tuple of (target_ids_set, manual_ids_list)
    """
    manual_ids = sorted(set(manual_map.keys()))
    target_ids = set(manual_ids)
    
    # Stream paper IDs from bibitems.jsonl
    all_ids_stream = []
    if bib_path.exists():
        with bib_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = obj.get("paper_id")
                if pid:
                    all_ids_stream.append(pid)
    
    all_ids_stream = sorted(set(all_ids_stream))
    if start:
        all_ids_stream = [p for p in all_ids_stream if p >= start]
    if num is not None:
        all_ids_stream = all_ids_stream[:num]
    
    target_ids.update(all_ids_stream)
    return target_ids, manual_ids


def process_manual_candidates(
    manual_map: Dict[str, Dict[str, str]],
    bib_by_pid: Dict[str, List[Dict]],
    ref_by_pid: Dict[str, List[Dict]],
    output_file
) -> Tuple[int, int]:
    """
    Process manual candidates and write to output file.
    
    Args:
        manual_map: Manual candidate mappings
        bib_by_pid: Bibliography items grouped by paper_id
        ref_by_pid: References grouped by paper_id
        output_file: File handle to write output
        
    Returns:
        Tuple of (matched_count, written_count)
    """
    matched = 0
    written = 0
    
    for pid, bib_map in manual_map.items():
        bibitems = bib_by_pid.get(pid, [])
        refs = ref_by_pid.get(pid, [])
        if not bibitems or not refs:
            continue
        
        # Build normalized arXiv map for references
        ref_norm_map = {}
        for ref in refs:
            ref_norm_map.setdefault(
                norm_arxiv(ref.get("arxiv") or ref.get("id") or ""), []
            ).append(ref)
        
        # Process each manual mapping
        for bib_key, cand_id in bib_map.items():
            bib = next((b for b in bibitems if b.get("key") == bib_key), None)
            if not bib:
                continue
            
            # Find matching reference
            ref = None
            rn = norm_arxiv(cand_id)
            for r in ref_norm_map.get(rn, []):
                rid = r.get("id") or r.get("arxiv") or ""
                if norm_arxiv(rid) == rn:
                    ref = r
                    break
            
            if not ref:
                continue
            
            # Compute features
            features = compute_features(bib, ref)
            
            # Write row
            row = {
                "paper_id": pid,
                "bib_key": bib_key,
                "cand_id": cand_id,
                "score": 1.0,
                **features,
                "label": 1,
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        
        matched += len(bib_map)
    
    return matched, written


def build_text(entry: Dict) -> str:
    """
    Build text representation for TF-IDF matching from entry dictionary.
    Combines title_tokens, author_tokens, authors_norm, title, and note_norm.
    """
    # Combine multiple signals to increase recall:
    # - title_tokens
    # - author_tokens
    # - authors_norm (full names, keeps prefixes/initials)
    # - title cleaned without stopword removal
    # - combined title+authors (no stopword removal)
    # - note_norm
    title_tokens = entry.get("title_tokens", []) or []
    author_tokens = entry.get("author_tokens", []) or []
    authors_norm = entry.get("authors_norm", []) or []

    title_full = normalize_ref_text(entry.get("title", "") or "", remove_stop=False)
    combined_full = normalize_ref_text(
        f"{entry.get('title', '')} {' '.join(entry.get('authors', []) or [])}",
        remove_stop=False,
    )
    note_norm = entry.get("note_norm", "") or ""

    parts = []
    if title_tokens:
        parts.append(" ".join(title_tokens))
    if author_tokens:
        parts.append(" ".join(author_tokens))
    if authors_norm:
        parts.append(" ".join(authors_norm))
    if title_full:
        parts.append(title_full)
    if combined_full:
        parts.append(combined_full)
    if note_norm:
        parts.append(note_norm)

    year = str(entry.get("year", "") or "")
    if year:
        parts.append(year)

    return " ".join(parts).strip()


def compute_matches(
    bibitems: List[Dict],
    refs: List[Dict],
    threshold: float = 0.5,
    collect_all_pairs: bool = False,
    vectorizer=None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute TF-IDF matches between bibitems and references.
    
    Args:
        bibitems: List of bibliography item dictionaries
        refs: List of reference dictionaries
        threshold: Score threshold for positive matches
        collect_all_pairs: If True, collect all pairs regardless of threshold
        vectorizer: Optional pre-fitted TfidfVectorizer
        
    Returns:
        Tuple of (matches, pair_candidates)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise ImportError("scikit-learn is required for TF-IDF matching") from e
    
    if not bibitems or not refs:
        return [], []

    bib_texts = [build_text(b) for b in bibitems]
    ref_texts = [build_text(r) for r in refs]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), stop_words=None)
        try:
            tfidf = vectorizer.fit_transform(bib_texts + ref_texts)
        except ValueError:
            return [], []
    else:
        try:
            tfidf = vectorizer.transform(bib_texts + ref_texts)
        except ValueError:
            return [], []

    b_mat = tfidf[: len(bib_texts)]
    r_mat = tfidf[len(bib_texts) :]
    sim = cosine_similarity(b_mat, r_mat)

    pair_candidates = []
    for i, bib in enumerate(bibitems):
        bib_key = bib.get("key", "")
        bib_arxiv = norm_arxiv(bib.get("arxiv", ""))
        b_year = str(bib.get("year", "") or "")

        row_candidates = []
        for j, ref in enumerate(refs):
            ref_id = ref.get("id") or ""
            ref_arxiv = norm_arxiv(ref.get("arxiv", ""))

            base_score = float(sim[i, j])
            title_tokens_b = set((bib.get("title_tokens") or []))
            title_tokens_r = set((ref.get("title_tokens") or []))
            title_jacc = (
                len(title_tokens_b & title_tokens_r) / len(title_tokens_b | title_tokens_r)
                if title_tokens_b and title_tokens_r
                else 0.0
            )

            if bib_arxiv and ref_arxiv and bib_arxiv == ref_arxiv:
                combined = 1.0
                reason = "arxiv_exact"
            else:
                r_year = str(ref.get("year", "") or "")
                b_year_int = parse_year_int(b_year)
                r_year_int = parse_year_int(r_year)
                year_match = 1.0 if b_year_int is not None and b_year_int == r_year_int else 0.0
                year_close = (
                    1.0
                    if b_year_int is not None
                    and r_year_int is not None
                    and abs(b_year_int - r_year_int) <= 1
                    else 0.0
                )
                auth_list_b = bib.get("authors_norm", []) or []
                auth_list_r = ref.get("authors_norm", []) or []
                auth_overlap_score = author_overlap(auth_list_b, auth_list_r)
                auth_last = author_lastname_match(auth_list_b, auth_list_r)

                combined_raw = (
                    base_score
                    + 0.2 * auth_overlap_score
                    + 0.1 * year_match
                    + 0.05 * year_close
                    + (0.1 if title_jacc >= 0.6 else 0.0)
                    + (0.2 if auth_last >= 1.0 else 0.0)
                )
                capped = combined_raw > 1.0
                combined = min(1.0, combined_raw)
                reason = (
                    f"tfidf_enriched(base={round(base_score,4)},auth={round(auth_overlap_score,3)},"
                    f"auth_last={round(auth_last,3)},year={year_match},"
                    f"title_cos=na,title_jacc={round(title_jacc,3)})"
                )
                if capped:
                    reason += "|cap1"

            if collect_all_pairs or combined >= threshold:
                row_candidates.append(
                    {
                        "bib_key": bib_key,
                        "arxiv_id": ref_id,
                        "score": round(combined, 4),
                        "reason": reason,
                    }
                )

        row_candidates.sort(key=lambda x: x["score"], reverse=True)
        pair_candidates.extend(row_candidates)

    # One-to-one assignment: greedy by score descending
    used_bib = set()
    used_ref = set()
    results = []
    for p in sorted(pair_candidates, key=lambda x: -x["score"]):
        if not p.get("arxiv_id"):
            continue
        if p["score"] < threshold:
            continue
        if p["bib_key"] in used_bib or p["arxiv_id"] in used_ref:
            continue
        used_bib.add(p["bib_key"])
        used_ref.add(p["arxiv_id"])
        results.append(p)

    return results, pair_candidates


def run_matching_and_feature_extraction(
    bib_by_pid: Dict[str, List[Dict]],
    ref_by_pid: Dict[str, List[Dict]],
    paper_ids: List[str],
    output_path: Path,
    match_threshold: float = 0.5,
    neg_per_pos: int = 10,
    hard_negatives: bool = True,
    keep_all_negatives: bool = False,
    max_refs: Optional[int] = None,
    max_bibs: Optional[int] = None,
    random_seed: int = 23120334,
    show_progress: bool = True,
    append: bool = False,
    tfidf_scope: str = "paper"
) -> Tuple[int, int]:
    """
    Run TF-IDF matching and feature extraction for papers.
    
    Args:
        bib_by_pid: Bibliography items grouped by paper_id
        ref_by_pid: References grouped by paper_id
        paper_ids: List of paper IDs to process
        output_path: Path to output JSONL file
        match_threshold: Threshold for positive matches
        neg_per_pos: Number of negative samples per positive
        hard_negatives: If True, sample highest-score negatives (hard negatives)
        keep_all_negatives: If True, keep all negatives (no sampling)
        max_refs: Maximum references per paper (None for no limit)
        max_bibs: Maximum bibitems per paper (None for no limit)
        random_seed: Random seed for sampling
        show_progress: Whether to show progress bar
        tfidf_scope: TF-IDF fit scope ("paper" or "split")
        
    Returns:
        Tuple of (matched_count, written_count)
    """
    random.seed(random_seed)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    out = output_path.open(mode, encoding="utf-8")
    
    matched = 0
    written = 0
    
    tfidf_vectorizer = None
    if tfidf_scope not in {"paper", "split"}:
        raise ValueError("tfidf_scope must be 'paper' or 'split'")
    if tfidf_scope == "split":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("scikit-learn is required for TF-IDF matching") from e
        corpus = []
        for pid in paper_ids:
            bibitems = bib_by_pid.get(pid, [])
            refs = ref_by_pid.get(pid, [])
            if not bibitems or not refs:
                continue
            corpus.extend(build_text(b) for b in bibitems)
            corpus.extend(build_text(r) for r in refs)
        tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), stop_words=None)
        try:
            tfidf_vectorizer.fit(corpus)
        except ValueError:
            tfidf_vectorizer = None

    iterator = tqdm(paper_ids, desc="Papers") if (show_progress and TQDM_AVAILABLE) else paper_ids
    
    for pid in iterator:
        bibitems = bib_by_pid.get(pid, [])
        refs = ref_by_pid.get(pid, [])
        if not bibitems or not refs:
            continue
        
        # Limit size if needed
        if max_refs is not None and len(refs) > max_refs:
            refs = random.sample(refs, max_refs)
        if max_bibs is not None and len(bibitems) > max_bibs:
            bibitems = random.sample(bibitems, max_bibs)
        
        # TF-IDF Matching
        matches, pair_candidates = compute_matches(
            bibitems,
            refs,
            threshold=match_threshold,
            collect_all_pairs=True,
            vectorizer=tfidf_vectorizer
        )
        
        # Build reference map
        ref_by_id = {}
        for ref in refs:
            rid = ref.get("id") or ref.get("arxiv") or ""
            if rid:
                ref_by_id[rid] = ref
        
        # Process each bibitem
        for bib in bibitems:
            bib_key = bib.get("key", "")
            if not bib_key:
                continue
            
            # Get candidates for this bib_key
            bib_candidates = [c for c in pair_candidates if c.get("bib_key") == bib_key]
            
            # Positives: score >= threshold
            positives = [c for c in bib_candidates if c.get("score", 0.0) >= match_threshold]
            for pos in positives:
                cand_id = pos.get("arxiv_id")
                ref = ref_by_id.get(cand_id)
                if not ref:
                    continue
                
                features = compute_features(bib, ref)
                row = {
                    "paper_id": pid,
                    "bib_key": bib_key,
                    "cand_id": cand_id,
                    "score": pos.get("score", 1.0),
                    **features,
                    "label": 1,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            matched += len(positives)
            
            # Negatives: score < threshold
            negatives = [c for c in bib_candidates if c.get("score", 0.0) < match_threshold]
            if negatives:
                if hard_negatives:
                    # Sort by score descending (hard negatives)
                    negatives.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                else:
                    random.shuffle(negatives)

                if keep_all_negatives:
                    neg_sample = negatives
                else:
                    neg_sample = negatives[:min(neg_per_pos, len(negatives))]
                
                for neg in neg_sample:
                    cand_id = neg.get("arxiv_id")
                    ref = ref_by_id.get(cand_id)
                    if not ref:
                        continue
                    
                    features = compute_features(bib, ref)
                    row = {
                        "paper_id": pid,
                        "bib_key": bib_key,
                        "cand_id": cand_id,
                        "score": neg.get("score", 0.0),
                        **features,
                        "label": 0,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
    
    out.close()
    return matched, written

def _display_path(path_value: Path) -> str:
    """Display shortened path using logic similar to display_path in notebook."""
    try:
        path_resolved = Path(path_value).resolve()
        # Use parent as base (assuming path_value is ROOT / name, so parent is ROOT)
        parent = path_resolved.parent
        if parent:
            relative = path_resolved.relative_to(parent)
            return str(Path(parent.name) / relative)
        return path_resolved.name
    except (ValueError, AttributeError):
        # Fallback to just the name
        return path_value.name if hasattr(path_value, 'name') else str(path_value)


def split_data(
    matches_path: Path,
    manual_map: Dict[str, Dict[str, str]],
    bib_by_pid: Dict[str, List[Dict]],
    ref_by_pid: Dict[str, List[Dict]],
    test_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
    split_dir: Path = None,
    pred_dir: Path = None,
    target_test_pct: float = 0.125,
    target_val_pct: float = 0.125
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Split data into train/val/test sets and generate pred.json files.
    
    Args:
        matches_path: Path to matches_fe.jsonl
        manual_map: Manual candidate mappings
        bib_by_pid: Bibliography items grouped by paper_id
        ref_by_pid: References grouped by paper_id
        test_ids: Predefined test paper IDs (None for auto-selection)
        val_ids: Predefined validation paper IDs (None for auto-selection)
        split_dir: Directory for split JSONL files
        pred_dir: Directory for pred.json files
        target_test_pct: Target percentage for test set (default 12.5%)
        target_val_pct: Target percentage for validation set (default 12.5%)
        
    Returns:
        Tuple of (split_map: paper_id -> partition, split_counts: partition -> count)
    """
    # Load data and count rows
    data = {}
    paper_set = set()
    row_counts = Counter()
    bad_lines = 0
    all_rows = []
    
    with matches_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            pid = obj.get("paper_id")
            bib = obj.get("bib_key")
            cand = obj.get("cand_id")
            if not pid or not bib or not cand:
                continue
            
            row_counts[pid] += 1
            paper_set.add(pid)
            data.setdefault(pid, {}).setdefault(bib, []).append(obj)
            all_rows.append(obj)
    
    if bad_lines:
        print(f"Warning: skipped {bad_lines} bad lines in {matches_path}")
    
    # Include manual IDs
    manual_ids = sorted(set(manual_map.keys()))
    paper_ids = sorted(paper_set | set(manual_ids))
    
    # Auto-select test/val if not provided
    manual_set = set(manual_ids)
    auto_ids = [p for p in paper_ids if p not in manual_set]
    
    if not test_ids or not val_ids:
        total_rows = sum(row_counts.values())
        target_test_rows = int(total_rows * target_test_pct)
        target_val_rows = int(total_rows * target_val_pct)
        
        manual_sorted = sorted(
            [pid for pid in manual_set if pid in row_counts],
            key=lambda x: row_counts[x], reverse=True
        )
        auto_sorted = sorted(
            [pid for pid in auto_ids if pid in row_counts],
            key=lambda x: row_counts[x], reverse=True
        )
        
        # Select test papers
        if not test_ids:
            test_papers = []
            test_rows = 0
            if manual_sorted:
                test_papers.append(manual_sorted[0])
                test_rows += row_counts[manual_sorted[0]]
            
            best_auto = None
            best_diff = float('inf')
            for pid in auto_sorted:
                if pid in test_papers:
                    continue
                candidate_rows = test_rows + row_counts[pid]
                diff = abs(candidate_rows - target_test_rows)
                if diff < best_diff:
                    best_diff = diff
                    best_auto = pid
                    if candidate_rows >= target_test_rows * 0.8:
                        break
            if best_auto:
                test_papers.append(best_auto)
            elif not test_papers and auto_sorted:
                test_papers.append(auto_sorted[0])
            test_ids = test_papers
        
        # Select val papers
        if not val_ids:
            val_papers = []
            val_rows = 0
            if len(manual_sorted) > 1 and manual_sorted[1] not in test_ids:
                val_papers.append(manual_sorted[1])
                val_rows += row_counts[manual_sorted[1]]
            elif manual_sorted and manual_sorted[0] not in test_ids:
                val_papers.append(manual_sorted[0])
                val_rows += row_counts[manual_sorted[0]]
            
            best_auto = None
            best_diff = float('inf')
            for pid in auto_sorted:
                if pid in test_ids or pid in val_papers:
                    continue
                candidate_rows = val_rows + row_counts[pid]
                diff = abs(candidate_rows - target_val_rows)
                if diff < best_diff:
                    best_diff = diff
                    best_auto = pid
                    if candidate_rows >= target_val_rows * 0.8:
                        break
            if best_auto:
                val_papers.append(best_auto)
            elif not val_papers and auto_sorted:
                for pid in auto_sorted:
                    if pid not in test_ids:
                        val_papers.append(pid)
                        break
            val_ids = val_papers
        
        # Log statistics
        test_total = sum(row_counts[p] for p in test_ids if p in row_counts)
        val_total = sum(row_counts[p] for p in val_ids if p in row_counts)
        train_total = total_rows - test_total - val_total
        print(f"\nSplit rows target: Test={target_test_rows:,} ({target_test_rows/total_rows*100:.1f}%), "
              f"Val={target_val_rows:,} ({target_val_rows/total_rows*100:.1f}%)")
        print(f"Split rows actual: Test={test_total:,} ({test_total/total_rows*100:.1f}%), "
              f"Val={val_total:,} ({val_total/total_rows*100:.1f}%), "
              f"Train={train_total:,} ({train_total/total_rows*100:.1f}%)")
        print(f"Test papers: {test_ids}, Val papers: {val_ids}")
    
    # Create split map
    split_map = {}
    for pid in paper_ids:
        if pid in test_ids:
            split_map[pid] = "test"
        elif pid in val_ids:
            split_map[pid] = "val"
        else:
            split_map[pid] = "train"
    
    # Write split JSONL files
    if split_dir:
        split_dir.mkdir(parents=True, exist_ok=True)
        split_paths = {
            "train": split_dir / "train.jsonl",
            "val": split_dir / "val.jsonl",
            "test": split_dir / "test.jsonl",
        }
        split_out = {k: p.open("w", encoding="utf-8") for k, p in split_paths.items()}
        split_counts = Counter()
        
        for obj in all_rows:
            pid = obj.get("paper_id")
            part = split_map.get(pid)
            if not part:
                continue
            split_out[part].write(json.dumps(obj, ensure_ascii=False) + "\n")
            split_counts[part] += 1
        
        for fh in split_out.values():
            fh.close()
        
        # Add manual candidates to split files
        if manual_map:
            for pid, bib_map in manual_map.items():
                part = split_map.get(pid, "train")
                path = split_paths.get(part)
                if not path:
                    continue
                bibitems = bib_by_pid.get(pid, [])
                refs = ref_by_pid.get(pid, [])
                if not bibitems or not refs:
                    continue
                
                ref_norm_map = {}
                for ref in refs:
                    ref_norm_map.setdefault(
                        norm_arxiv(ref.get("arxiv") or ref.get("id") or ""), []
                    ).append(ref)
                
                with path.open("a", encoding="utf-8") as fout:
                    for bib_key, cand_id in bib_map.items():
                        bib = next((b for b in bibitems if b.get("key") == bib_key), None)
                        if not bib:
                            continue
                        
                        ref_obj = None
                        rn = norm_arxiv(cand_id)
                        for r in ref_norm_map.get(rn, []):
                            rid = r.get("id") or r.get("arxiv") or ""
                            if norm_arxiv(rid) == rn:
                                ref_obj = r
                                break
                        
                        if not ref_obj:
                            continue
                        
                        features = compute_features(bib, ref_obj)
                        row = {
                            "paper_id": pid,
                            "bib_key": bib_key,
                            "cand_id": cand_id,
                            "score": 1.0,
                            **features,
                            "label": 1,
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        split_counts[part] += 1
        
        print("Split jsonl counts:", dict(split_counts))
    else:
        split_counts = Counter()
    
    # Generate pred.json files
    if pred_dir:
        def rank_candidates(cands):
            return sorted(
                cands,
                key=lambda x: (
                    -(x.get("score") or 0.0),
                    -(x.get("levenshtein") or 0.0),
                    -(x.get("jaccard") or 0.0),
                ),
            )
        
        written_pred = 0
        for pid, bib_map in data.items():
            partition = split_map.get(pid, "train")
            gt = {}
            pred = {}
            
            for bib_key, cands in bib_map.items():
                pos = [c for c in cands if c.get("label") == 1]
                if pos:
                    best = rank_candidates(pos)[0]
                    gt[bib_key] = best.get("cand_id")
                else:
                    continue
                pred[bib_key] = []
            
            out_obj = {
                "partition": partition,
                "groundtruth": gt,
                "prediction": pred,
            }
            
            paper_dir = pred_dir / pid
            paper_dir.mkdir(parents=True, exist_ok=True)
            with (paper_dir / "pred.json").open("w", encoding="utf-8") as fout:
                json.dump(out_obj, fout, ensure_ascii=False, indent=2)
            written_pred += 1
        
        # Display shortened path using display_path logic
        pred_dir_display = _display_path(pred_dir)
        print(f"pred.json written: {written_pred} files under {pred_dir_display}")
    
    return split_map, split_counts


def compute_statistics(matches_path: Path, num_samples: int = 5) -> Dict:
    """
    Compute statistics from matches_fe.jsonl.
    
    Args:
        matches_path: Path to matches_fe.jsonl
        num_samples: Number of sample rows to return
        
    Returns:
        Dictionary containing label counts and sample rows
    """
    labels = Counter()
    sample_rows = []
    
    if matches_path.exists():
        with matches_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    labels[obj.get("label", 0)] += 1
                    if i < num_samples:
                        sample_rows.append(obj)
                except json.JSONDecodeError:
                    continue
    
    return {
        "labels": dict(labels),
        "samples": sample_rows,
    }


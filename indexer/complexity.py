import re


def _tokenize(text: str):
    return re.findall(r"[A-Za-z_0-9]+", text)


def semantic_complexity_score(text: str) -> float:
    """Light-weight heuristic: ratio of unique tokens to total tokens.

    • 0.0 → highly repetitive/homogeneous (e.g. minified CSS)
    • ≈0.05–0.14 → moderately varied (mixed markup & code)
    • ≥0.15 → highly varied/complex (large code files)
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    uniq = len(set(tokens))
    return round(uniq / len(tokens), 4)


def choose_chunk_size(text: str) -> int:
    """Map semantic complexity to a reasonable chunk size (in chars)."""
    scs = semantic_complexity_score(text)

    if scs < 0.05:
        base = 1200
    elif scs < 0.15:
        base = 600
    else:
        base = 300

    # Always return at least the base size; do not shrink small inputs below base
    return max(base, len(text)) 
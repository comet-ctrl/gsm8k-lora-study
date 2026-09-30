"""Conservative exact-number scoring. No expression evaluation."""

import re
from fractions import Fraction


def boxed_content(text: str) -> str | None:
    """Read the final box, including nested braces; reject an unclosed box."""
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        if depth == 0:
            return text[start:end]
    return None


def number(text: str | None) -> Fraction | None:
    if text is None:
        return None
    value = text.strip().replace("−", "-")
    latex = re.fullmatch(r"([+-]?)\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if latex:
        sign, numerator, denominator = latex.groups()
        value = f"{sign}{numerator}/{denominator}"
    # Commas must form proper groups; do not silently turn 1,5 into 15.
    if "," in value:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
            return None
        value = value.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?|[+-]?\d+/[+-]?\d+", value):
        return None
    if len(value) > 128:
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def score(response: str, expected: str) -> dict:
    """Numeric accuracy permits a final-number fallback; strict accuracy needs a box."""
    target = number(expected)
    if target is None:
        raise ValueError("Reference answer is not a supported number")
    content = boxed_content(response)
    has_box = r"\boxed{" in response
    candidate = content
    if not has_box:
        matches = re.findall(r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?(?:%|[eE][+-]?\d+)?", response)
        candidate = matches[-1] if matches else None
    parsed = number(candidate)
    correct = parsed is not None and parsed == target
    return {
        "answer": str(parsed) if parsed is not None else None,
        "correct": correct,
        "boxed": content is not None and parsed is not None,
        "strict_correct": correct and content is not None,
        "invalid": parsed is None,
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    total = len(records)
    return {"total": total, **{
        key: sum(bool(row[key]) for row in records) / total
        for key in ("correct", "strict_correct", "boxed", "invalid")
    }}

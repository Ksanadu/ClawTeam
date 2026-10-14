"""Custom tokenizer (preprocess_func) for BM25: jieba + path/identifier preservation.

Designed for mixed zh/en engineering conversations:
- Preserves file paths, identifiers, version numbers as whole tokens
- Uses jieba for Chinese text segmentation
- Filters Chinese punctuation and single-char noise
- Splits camelCase/snake_case while keeping originals
"""

from __future__ import annotations

import re

import jieba  # type: ignore[import-untyped]

_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:)?(?:[/\\])?(?:[\w.@-]+[/\\])+[\w.@-]+\.\w{1,10}"
)
_FILENAME_PATTERN = re.compile(
    r"\b[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|css|scss|html|vue"
    r"|svelte|json|yaml|yml|toml|md|sql|sh|bat|ps1|env|cfg|ini|txt|xml|csv|proto)\b"
)
# Use lookaround instead of \b so it works at CJK↔ASCII boundaries
_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[_A-Za-z][_A-Za-z0-9]{2,}(?![A-Za-z0-9_])"
)
_VERSION_PATTERN = re.compile(r"\b\d+(?:\.\d+){1,3}\b")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_CJK_PUNCTUATION = set(
    "，。！？；：、""''（）【】《》〈〉…—～·「」『』"
    ",.!?;:()[]<>{}\"'`~@#$%^&*+=|/\\\n\r\t"
)


def _is_noise_token(t: str) -> bool:
    """Filter out punctuation-only and single meaningless chars."""
    if not t or t.isspace():
        return True
    if all(c in _CJK_PUNCTUATION for c in t):
        return True
    if len(t) == 1:
        # Keep single CJK ideographs; jieba often returns single characters for short queries.
        # Otherwise BM25 query tokens could become empty, breaking keyword retrieval.
        if re.match(r"^[\u3400-\u4DBF\u4E00-\u9FFF]$", t):
            return False
        if not t.isalnum():
            return True
    return False


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing, preserving engineering-relevant tokens."""
    if not text:
        return []

    tokens: list[str] = []
    preserved_spans: list[tuple[int, int]] = []

    for pattern in [_PATH_PATTERN, _FILENAME_PATTERN, _VERSION_PATTERN]:
        for m in pattern.finditer(text):
            token = m.group()
            tokens.append(token.lower())
            preserved_spans.append((m.start(), m.end()))
            if "/" in token or "\\" in token:
                parts = re.split(r"[/\\]", token)
                tokens.extend(p.lower() for p in parts if p)

    for m in _IDENTIFIER_PATTERN.finditer(text):
        s, e = m.start(), m.end()
        if any(ps <= s < pe for ps, pe in preserved_spans):
            continue
        ident = m.group()
        tokens.append(ident.lower())

        if "_" in ident:
            tokens.extend(p.lower() for p in ident.split("_") if p)
        elif "-" in ident:
            tokens.extend(p.lower() for p in ident.split("-") if p)

        camel_parts = _CAMEL_SPLIT.split(ident)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts if p)

        preserved_spans.append((s, e))

    remaining = text
    for ps, pe in sorted(preserved_spans, reverse=True):
        remaining = remaining[:ps] + " " + remaining[pe:]

    jieba_tokens = jieba.lcut(remaining)
    for t in jieba_tokens:
        t = t.strip()
        if not _is_noise_token(t):
            tokens.append(t.lower())

    tokens = [t for t in tokens if not _is_noise_token(t)]
    return tokens

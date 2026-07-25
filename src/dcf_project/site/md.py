"""Minimal, dependency-free Markdown -> HTML for daily research notes.

Covers what a note actually uses: #/##/### headings, paragraphs, **bold**, *italic*, `code`,
[links](url), - / 1. lists, > blockquotes, --- rules, ``` fenced code. Input is HTML-escaped
first, so authored notes can never inject markup into the site.
"""
from __future__ import annotations

import html
import re

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)"),
     r'<a href="\2" target="_blank" rel="noopener">\1</a>'),
]


def _inline(s: str) -> str:
    for rx, rep in _INLINE:
        s = rx.sub(rep, s)
    return s


def render(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    in_ul = in_ol = in_quote = False

    def close_blocks():
        nonlocal in_ul, in_ol, in_quote
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False
        if in_quote:
            out.append("</blockquote>")
            in_quote = False

    while i < n:
        raw = lines[i]
        line = html.escape(raw, quote=False)
        stripped = line.strip()
        if stripped.startswith("```"):
            close_blocks()
            code: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(html.escape(lines[i], quote=False))
                i += 1
            out.append("<pre><code>" + "\n".join(code) + "</code></pre>")
            i += 1
            continue
        if not stripped:
            close_blocks()
            i += 1
            continue
        m = re.match(r"(#{1,4})\s+(.*)", stripped)
        if m:
            close_blocks()
            lvl = len(m.group(1)) + 1  # page h1 is the post title; body starts at h2
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
        elif stripped in ("---", "***"):
            close_blocks()
            out.append("<hr>")
        elif stripped.startswith("&gt; ") or stripped == "&gt;":
            if not in_quote:
                close_blocks()
                out.append("<blockquote>")
                in_quote = True
            out.append(f"<p>{_inline(stripped[5:])}</p>")
        elif re.match(r"[-*]\s+", stripped):
            if not in_ul:
                close_blocks()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(re.sub(r'^[-*]\\s+', '', stripped))}</li>")
        elif re.match(r"\d+\.\s+", stripped):
            if not in_ol:
                close_blocks()
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_inline(re.sub(r'^\\d+\\.\\s+', '', stripped))}</li>")
        else:
            close_blocks()
            out.append(f"<p>{_inline(stripped)}</p>")
        i += 1
    close_blocks()
    return "\n".join(out)


def parse_post(text: str) -> tuple[dict, str]:
    """Split `--- key: value ---` frontmatter from the body. Unknown keys pass through."""
    meta: dict[str, str] = {}
    body = text
    if text.lstrip().startswith("---"):
        parts = text.lstrip().split("\n")
        try:
            end = parts[1:].index("---") + 1
        except ValueError:
            return meta, body
        for ln in parts[1:end]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        body = "\n".join(parts[end + 1:])
    return meta, body

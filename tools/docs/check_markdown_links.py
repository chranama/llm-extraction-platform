#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def slugify(text: str) -> str:
    text = text.strip().lower()
    out = []
    prev_dash = False
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
            prev_dash = False
            continue
        if ch.isspace() or ch in {"/", ".", ":"}:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug


def extract_anchors(md_path: Path) -> set[str]:
    anchors: set[str] = set()
    try:
        lines = md_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return anchors
    for line in lines:
        m = HEADING_RE.match(line)
        if not m:
            continue
        anchors.add(slugify(m.group(2)))
    return anchors


def should_ignore(raw: str) -> bool:
    lowered = raw.lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("mailto:")
        or lowered.startswith("tel:")
        or lowered.startswith("#")
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_markdown_links.py <md-file> [<md-file> ...]", file=sys.stderr)
        return 2

    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for file_arg in argv[1:]:
        md_file = Path(file_arg)
        if not md_file.exists():
            errors.append(f"{file_arg}: file missing")
            continue

        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"{md_file}: unable to read ({exc})")
            continue

        for link in LINK_RE.findall(text):
            raw = link.strip()
            if should_ignore(raw):
                continue

            path_part, anchor = (raw.split("#", 1) + [""])[:2] if "#" in raw else (raw, "")
            target = (md_file.parent / path_part).resolve()

            if not target.exists():
                errors.append(f"{md_file}: broken link target '{raw}'")
                continue

            if anchor:
                if target.suffix.lower() != ".md":
                    continue
                anchors = anchor_cache.get(target)
                if anchors is None:
                    anchors = extract_anchors(target)
                    anchor_cache[target] = anchors
                want = slugify(anchor)
                if want and want not in anchors:
                    errors.append(
                        f"{md_file}: missing anchor '#{anchor}' in '{target}'"
                    )

    if errors:
        print("Markdown link check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Markdown link check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

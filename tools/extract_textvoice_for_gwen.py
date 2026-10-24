"""Extract concrete speech strings from the supplied TextVoice Markdown inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"F:\easy_math_1.0.0\docs\TONG_HOP_TEXTVOICE_KHOI_DONG_KHAM_PHA.md")
DEFAULT_OUTPUT = ROOT / "data/gwen_cache"
DYNAMIC = re.compile(r"\$\{|\$[A-Za-z_]")
TECHNICAL_LITERALS = {
    "questionText", "imageQuestion", "choices", "detailType", "questionType",
    "rawQuestion", "hintVoice", "createdAt", "loaimoi", "questionId",
}


def cell_text(value: str) -> str:
    # Undo only presentation escaping, not arbitrary Dart escape sequences.
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = html.unescape(value)
    value = re.sub(r"\\([\\`*_{}\[\]()#+.!|>\-])", r"\1", value)
    return unicodedata.normalize("NFC", value).strip()


def extract(markdown: str) -> dict:
    lesson = filename = ""
    mode = None
    table = False
    entries = {}
    skipped = []
    categories = Counter()
    for document_line, line in enumerate(markdown.splitlines(), 1):
        if line.startswith("### "):
            lesson = line[4:]
            mode = None
        elif line.startswith("#### "):
            filename = line[5:]
            mode = None
        elif line.startswith("##### TextVoice"):
            mode = "voice"
        elif line.startswith("<details>"):
            mode = "supplementary"
        if not line.startswith("|"):
            table = False
            continue
        cells = [part.strip() for part in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) == 4 and cells[:3] == ["Dòng", "Biến / ngữ cảnh", "Loại"]:
            table = True
            continue
        if not table or not cells or not re.match(r"^\d", cells[0]):
            continue
        if len(cells) != 4:
            raise ValueError(f"Unexpected content table at Markdown line {document_line}")
        source_lines, context, kind, raw_text = cells
        text = cell_text(raw_text)
        record = {"lesson": lesson, "file": filename, "source_lines": source_lines,
                  "document_line": document_line, "context": cell_text(context), "kind": kind}
        reason = None
        if mode != "voice":
            reason = "supplementary_labels_or_data"
        elif kind == "Mẫu động" or DYNAMIC.search(text):
            reason = "unresolved_template"
        elif (text in TECHNICAL_LITERALS or context == "isSubtraction"
              or "Debug" in context or text.startswith("Debug JSON")):
            reason = "technical_or_debug_text"
        elif not text:
            reason = "empty"
        if reason:
            categories[reason] += 1
            skipped.append({**record, "text": text, "reason": reason})
            continue
        categories["concrete_voice_rows"] += 1
        entries.setdefault(text, []).append(record)
    return {"texts": list(entries), "sources": entries, "skipped": skipped,
            "counts": dict(categories)}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source = args.source.read_bytes()
    result = extract(source.decode("utf-8-sig"))
    if not result["texts"]:
        parser.error("No concrete TextVoice strings found")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "texts.json", result["texts"])
    pending = [item for item in result["skipped"] if item["reason"] == "unresolved_template"]
    write_json(args.output / "pending_templates.json", pending)
    write_json(args.output / "extraction_report.json", {
        "source": str(args.source.resolve()), "source_sha256": hashlib.sha256(source).hexdigest(),
        "unique_texts": len(result["texts"]), "counts": result["counts"],
        "duplicates_removed": result["counts"].get("concrete_voice_rows", 0) - len(result["texts"]),
        "sources": result["sources"], "excluded": result["skipped"],
    })
    print(f"Extracted {len(result['texts'])} unique texts -> {args.output / 'texts.json'}")
    print(f"Unresolved dynamic rows: {len(pending)} (not sent to TTS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

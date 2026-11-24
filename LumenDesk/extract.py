"""Extract originals into one provenance-preserving JSON file per document."""

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from docx import Document
from docx.table import Table
from openpyxl import load_workbook
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
DOCUMENTS = ROOT / "documents"
EXTRACTED = ROOT / "prepared" / "original"
EXTRACTOR_VERSION = 2


def segment(location, value):
    return {"location": location, "original": value.strip()}


def extract_text(path):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if path.suffix == ".txt":
        return [segment({"lines": [1, len(lines)]}, text)]

    sections = []
    first, heading = 1, ""
    for number, line in enumerate(lines, 1):
        if match := re.match(r"^#{1,6}\s+(.+)", line):
            if number > first:
                sections.append((first, number - 1, heading))
            first, heading = number, match[1]
    sections.append((first, len(lines), heading))
    return [
        segment({"section": title, "lines": [start, end]}, "\n".join(lines[start - 1:end]))
        for start, end, title in sections
        if any(line.strip() for line in lines[start - 1:end])
    ]


def extract_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        dialect = csv.Sniffer().sniff(file.readline(), delimiters=",;\t")
        file.seek(0)
        reader = csv.reader(file, dialect)
        headers = next(reader)
        last_line = reader.line_num
        segments = []
        for row in reader:
            start_line, last_line = last_line + 1, reader.line_num
            if not any(row):
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV row width differs from header at line {start_line}")
            value = "\n".join(f"{header}: {cell}" for header, cell in zip(headers, row))
            segments.append(segment({"lines": [start_line, last_line], "columns": headers}, value))
        return segments


def extract_excel(path):
    workbook = load_workbook(path, data_only=True)
    segments = []
    try:
        for sheet in workbook:
            headers = {}
            for row in sheet.iter_rows():
                cells = [cell for cell in row if cell.value is not None]
                if not cells:
                    continue
                location = {"sheet": sheet.title, "range": f"{cells[0].coordinate}:{cells[-1].coordinate}"}
                if not headers and len(cells) > 1:
                    headers = {cell.column: cell for cell in cells}
                elif headers and len(cells) > 1:
                    location["headers"] = {
                        cell.coordinate: {
                            "cell": headers[cell.column].coordinate,
                            "text": headers[cell.column].value,
                        }
                        for cell in cells
                        if cell.column in headers
                    }
                value = "\n".join(f"{cell.coordinate}: {cell.value}" for cell in cells)
                segments.append(segment(location, value))
    finally:
        workbook.close()
    return segments


def extract_pdf(path):
    return [
        segment({"page": number}, page.extract_text(extraction_mode="layout"))
        for number, page in enumerate(PdfReader(path).pages, 1)
    ]


def extract_word(path):
    document = Document(path)
    segments = []
    heading = ""
    paragraph_number = table_number = 0
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            table_number += 1
            for row_number, row in enumerate(item.rows, 1):
                value = " | ".join(cell.text for cell in row.cells)
                segments.append(segment({"section": heading, "table": table_number, "row": row_number}, value))
        else:
            paragraph_number += 1
            if item.style.name in {"Title", "Subtitle"} or item.style.name.startswith("Heading"):
                heading = item.text.strip()
            if item.text.strip():
                segments.append(segment({"section": heading, "paragraph": paragraph_number}, item.text))
    return segments


READERS = {
    ".txt": extract_text,
    ".md": extract_text,
    ".csv": extract_csv,
    ".xlsx": extract_excel,
    ".pdf": extract_pdf,
    ".docx": extract_word,
}


def main():
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    failures = []
    for path in sorted(DOCUMENTS.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        relative = unicodedata.normalize("NFC", path.relative_to(ROOT).as_posix())
        target = EXTRACTED / Path(relative).relative_to("documents")
        target = target.with_suffix(target.suffix + ".json")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if (previous["sha256"], previous["extractor_version"]) == (digest, EXTRACTOR_VERSION):
                continue
        try:
            fragments = READERS[path.suffix.lower()](path)
            if not fragments or any(not item["original"] for item in fragments):
                raise ValueError("No readable text in one or more segments")
            for number, fragment in enumerate(fragments, 1):
                fragment["id"] = number
            result = {
                "source": relative,
                "sha256": digest,
                "extractor_version": EXTRACTOR_VERSION,
                "fragments": fragments,
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            failures.append(f"{relative}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()

"""Translate prepared EN/FR documents locally, keeping fragment boundaries."""

import argparse
from collections import Counter
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "prepared" / "original"
TRANSLATED = ROOT / "prepared" / "ru"
MODELS = {"en": "en-ru", "fr": "fr-ru"}
VERSION = 2
TECHNICAL = re.compile(
    r"(?<![\w-])(?:[A-Z]{2,}(?:[-_][A-Za-z0-9]+)*|[A-Z][a-z]+[A-Z][A-Za-z]*|"
    r"Starter|Team|Enterprise|LumenDesk|MiB|GiB|KiB|\d+(?:[.,:]\d+)*)(?![\w-])"
)
STRUCTURE = re.compile(r"^(\s*(?:#{1,6}\s+|[-*]\s+|\d+[.)]\s+|[A-Z]+\d+:\s*))")
PARAGRAPHS = re.compile(r"(\n\s*\n)")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(\s+)(?=[A-ZÀ-Ü«“])")
NUMBER = re.compile(r"(\d+(?:[.,:]\d+)*)")
LITERAL = re.compile(
    r"(?:\d[\d .,:%+/-]*(?:\s+UTC)?|[A-Za-z]+/[A-Za-z_]+|[A-Za-z]+[-_]\d[\w-]*|"
    r"dead_letter|[1-5]xx|UTC|MiB|GiB|KiB|Starter|Team|Enterprise|LumenDesk|[A-Z]{2,})"
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def numbers(text):
    return Counter(NUMBER.findall(text))


def source_pieces(text, suffix):
    if suffix == ".pdf":
        text = "".join(part if PARAGRAPHS.fullmatch(part) else re.sub(r"(?<=\S) *\n *(?=\S)", " ", part)
                       for part in PARAGRAPHS.split(text))
        return re.split(r"(\n+)", text)
    return re.split(r"(\n)", text)


def unit_templates(text, suffix):
    """Preserve table structure, headings and separators while translating prose."""
    for line in source_pieces(text, suffix):
        if not line.strip():
            yield line, False
            continue
        if suffix == ".csv" and ":" in line:
            header, value = line.split(":", 1)
            yield header, True
            yield ":", False
            yield value, bool(value.strip())
            continue
        prefix = STRUCTURE.match(line)
        if prefix:
            yield prefix[0], False
            line = line[prefix.end():]
        for part in re.split(r"(\s*\|\s*)", line):
            yield part, bool(part.strip() and "|" not in part)


class Translator:
    def __init__(self, language):
        path = ROOT / "models" / MODELS[language]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
        self.model.eval()

    def batches(self, texts):
        result = {}
        unique = sorted(set(texts), key=len)
        for start in range(0, len(unique), 12):
            batch = unique[start:start + 12]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            if inputs.input_ids.shape[-1] > 512:
                raise ValueError("Translation input exceeds 512 tokens")
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_length=512, num_beams=2)
            values = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            result.update(zip(batch, values))
        return result

    def translate(self, document):
        suffix = Path(document["source"]).suffix
        templates = {}
        for fragment in document["fragments"]:
            templates[fragment["id"]] = list(unit_templates(fragment["original"], suffix))
        pieces = {key: [(part, translate)
                        for unit, translate in parts
                        for part in (SENTENCE_BOUNDARY.split(unit) if translate else [unit])]
                  for key, parts in templates.items()}
        translations = self.translate_units(unit for parts in pieces.values() for unit, use in parts if use)
        return {str(key): "".join(translations.get(unit, unit) if use else unit for unit, use in parts)
                for key, parts in pieces.items()}

    def translate_units(self, units):
        units = set(units)
        active = {unit: unit.strip() for unit in units
                  if re.search(r"[A-Za-zÀ-ÿ]{2}", unit)
                  and not LITERAL.fullmatch(unit.strip()) and not TECHNICAL.fullmatch(unit.strip())}
        machine = self.batches(active.values())
        fallback = {}
        fixed = {}
        for unit, source in active.items():
            draft = machine[source]
            if draft and numbers(source) == numbers(draft):
                fixed[unit] = draft
            else:
                fallback[unit] = NUMBER.split(source)
        parts = {text.strip() for split in fallback.values() for text in split[::2]
                 if re.search(r"[A-Za-zÀ-ÿ]{2}", text)}
        translated_parts = self.batches(parts)
        for unit, split in fallback.items():
            for i in range(0, len(split), 2):
                original = split[i]
                candidate = translated_parts.get(original.strip(), original.strip())
                if candidate and not numbers(candidate):
                    split[i] = with_spacing(original, candidate)
            fixed[unit] = "".join(split)
        return {unit: with_spacing(unit, translated) for unit, translated in fixed.items()}


def with_spacing(original, content):
    start = original[:len(original) - len(original.lstrip())]
    end = original[len(original.rstrip()):]
    return start + content + end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=Path, default=ROOT / "registry.json")
    parser.add_argument("--language", choices=MODELS)
    args = parser.parse_args()
    languages = {item["source"]: item["language"] for item in json.loads(args.languages.read_text(encoding="utf-8"))}
    torch.set_num_threads(4)
    translators = {}
    for path in sorted(ORIGINAL.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        language = languages[document["source"]]
        if language == "ru" or args.language and language != args.language:
            continue
        target = TRANSLATED / path.relative_to(ORIGINAL)
        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            if (old["sha256"], old["extractor_version"], old["translation_version"]) == (
                document["sha256"], document["extractor_version"], VERSION
            ):
                continue
        if language not in translators:
            translators[language] = Translator(language)
        translations = translators[language].translate(document)
        mismatches = [
            fragment["id"] for fragment in document["fragments"]
            if numbers(fragment["original"]) != numbers(translations[str(fragment["id"])])
        ]
        write_json(target, {
            "source": document["source"],
            "sha256": document["sha256"],
            "extractor_version": document["extractor_version"],
            "translation_version": VERSION,
            "model": f"Helsinki-NLP/opus-mt-{MODELS[language]}",
            "numeric_review": mismatches,
            "fragments": translations,
        })
        print(f"{document['source']}: {len(translations)}; numeric review {mismatches}", flush=True)


if __name__ == "__main__":
    main()

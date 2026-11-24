"""Build an index, run role-scoped retrieval, and append grounded chat answers."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
CACHED_INDEX = ROOT / "index" / "corpus.npz"
HISTORY = ROOT / "history"
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
ROLES = ("support_agent", "support_lead")
DECISIONS = ("answer", "clarify", "refuse", "escalate")
SETTINGS = {"model": MODEL, "revision": REVISION, "normalize": True, "prefix": "",
            "max_tokens": 2048, "per_source": 4, "word_neighbors": 4, "algorithm_version": 3}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(render_json(value), encoding="utf-8")
        path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def doc_map(folder):
    return {document["source"]: document for path in sorted(folder.rglob("*.json"))
            if (document := read_json(path))}


def spreadsheet_cells(fragments, translations):
    cells = {}
    for fragment in fragments:
        sheet = fragment["location"]["sheet"]
        text = translations.get(str(fragment["id"]), fragment["original"])
        for line in text.splitlines():
            address, separator, value = line.partition(": ")
            if separator and re.fullmatch(r"[A-Z]+[0-9]+", address):
                cells[sheet, address] = value
    return cells


def prepared_corpus():
    extracted = doc_map(PREPARED / "original")
    translated = doc_map(PREPARED / "ru")
    registry = read_json(ROOT / "registry.json")
    sources = {record["source"] for record in registry}
    if sources != extracted.keys() or len(registry) != len(sources):
        raise ValueError("Registry and extraction must contain the same unique sources")
    allowed = []
    for record in registry:
        source = record["source"]
        language = record["language"]
        document = extracted[source]
        fragments = document["fragments"]
        russian = {}
        if language != "ru":
            localized = translated[source]
            identifiers = {str(item["id"]) for item in fragments}
            if localized["sha256"] != document["sha256"] or (
                localized["extractor_version"] != document["extractor_version"]
            ) or identifiers != localized["fragments"].keys() or any(
                not text.strip() for text in localized["fragments"].values()
            ):
                raise ValueError(f"Missing or outdated translation: {source}")
            russian = localized["fragments"]
        if record["status"] != "current":
            continue
        roles = record["roles"]
        if not roles:
            continue
        if set(roles) - set(ROLES):
            raise ValueError(f"Unknown role in registry: {source}")
        excluded = set(record.get("exclude_fragments", []))
        if excluded - {item["id"] for item in fragments}:
            raise ValueError(f"Unknown fragment in registry: {source}")
        selected = [{**item, "text": russian.get(str(item["id"]), item["original"])}
                    for item in fragments if item["id"] not in excluded]
        suffix = Path(source).suffix
        cells = spreadsheet_cells(fragments, russian) if suffix == ".xlsx" else {}
        for group in grouped(selected, suffix):
            location = group[0]["location"]
            section = location.get("section", "") if language == "ru" else ""
            sheet = location.get("sheet")
            if sheet:
                section = cells.get((sheet, "A1"), sheet if language == "ru" else "")
            headers = [cells.get((sheet, item["cell"]), "") for item in location.get("headers", {}).values()]
            context = " ".join(filter(None, [record["purpose"], section, " / ".join(filter(None, headers))]))
            allowed.append({"source": source, "language": language, "roles": roles,
                            "context": context, "fragments": group})
    return allowed


def grouped(fragments, suffix):
    if suffix not in {".docx", ".csv"}:
        yield from ([fragment] for fragment in fragments)
        return

    def block(fragment):
        if suffix == ".docx":
            return fragment["location"].get("section", "")
        return fragment["original"].split("\n", 1)[0]

    group = []
    for fragment in fragments:
        if group and (fragment["id"] != group[-1]["id"] + 1
                      or block(fragment) != block(group[-1])
                      or sum(len(piece["text"]) for piece in group) + len(fragment["text"]) > 1600):
            yield group
            group = []
        group.append(fragment)
    if group:
        yield group


def data_fingerprint():
    digest = hashlib.sha256(render_json(SETTINGS).encode())
    paths = [ROOT / "registry.json", *sorted(PREPARED.rglob("*.json"))]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_encoder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu", trust_remote_code=False)
    model.max_seq_length = SETTINGS["max_tokens"]
    return model


def index_text(chunk):
    return chunk["context"] + "\n" + "\n".join(piece["text"] for piece in chunk["fragments"])


def build():
    fingerprint = data_fingerprint()
    corpus = prepared_corpus()
    if not corpus:
        raise ValueError("No allowed material in registry")
    model = load_encoder()
    embeddings = model.encode([index_text(chunk) for chunk in corpus], prompt="",
                              normalize_embeddings=True, batch_size=8, show_progress_bar=True)
    CACHED_INDEX.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHED_INDEX.with_name(f"{uuid4().hex}.npz")
    np.savez_compressed(temporary, data=render_json(corpus), embeddings=embeddings, fingerprint=fingerprint)
    temporary.replace(CACHED_INDEX)
    print(f"{len(corpus)} chunks indexed")


def load_index():
    with np.load(CACHED_INDEX, allow_pickle=False) as index:
        fingerprint = index["fingerprint"].item()
        if fingerprint != data_fingerprint():
            raise ValueError("Index is outdated. Run python rag.py build")
        return json.loads(index["data"].item()), index["embeddings"], fingerprint


def canonical_text(chunk):
    return " ".join(" ".join(item["text"] for item in chunk["fragments"]).split())


def result_hit(chunk, rank, score, method):
    hit = {key: value for key, value in chunk.items() if key != "roles"}
    return {**hit, "id": f"R{rank}", "score": round(float(score), 6), "retrieval": method, "copies": []}


def retrieve(role, question, limit, corpus, embeddings, model):
    permitted = [i for i, chunk in enumerate(corpus) if role in chunk["roles"]]
    vector = model.encode(question, prompt="", normalize_embeddings=True, show_progress_bar=False)
    scores = embeddings[permitted] @ vector
    by_fragment = {
        (corpus[index]["source"], corpus[index]["fragments"][0]["id"]): offset
        for offset, index in enumerate(permitted)
        if corpus[index]["source"].endswith(".docx")
    }
    source_hits = Counter()
    seen = {}
    hits = []
    for rank in np.argsort(-scores, kind="stable"):
        chunk = corpus[permitted[rank]]
        if source_hits[chunk["source"]] >= SETTINGS["per_source"]:
            continue
        fingerprint = canonical_text(chunk)
        if fingerprint in seen:
            seen[fingerprint]["copies"].append({"source": chunk["source"], "fragments": chunk["fragments"]})
            continue
        if len(hits) >= limit:
            continue
        hit = result_hit(chunk, len(hits) + 1, scores[rank], "dense")
        hits.append(hit)
        seen[fingerprint] = hit
        source_hits[hit["source"]] += 1
    expanded = 0
    for hit in hits[:6]:
        if expanded >= SETTINGS["word_neighbors"]:
            break
        next_piece = hit["fragments"][-1]["id"] + 1
        offset = by_fragment.get((hit["source"], next_piece))
        if offset is None:
            continue
        chunk = corpus[permitted[offset]]
        fingerprint = canonical_text(chunk)
        if fingerprint not in seen:
            extra = result_hit(chunk, len(hits) + 1, scores[offset], "next_word_section")
            hits.append(extra)
            seen[fingerprint] = extra
            expanded += 1
    return hits


def run_query(role, question, limit, corpus, embeddings, fingerprint, model):
    if role not in ROLES or not isinstance(question, str) or not question.strip() or not 1 <= limit <= 30:
        raise ValueError("A supported role, non-empty question and limit between 1 and 30 are required")
    matches = retrieve(role, question, limit, corpus, embeddings, model)
    instant = datetime.now(timezone.utc)
    run = f"{instant:%Y%m%dT%H%M%S}-{uuid4().hex[:12]}"
    entry = {"run": run, "time_utc": instant.isoformat(), "role": role, "question": question,
             "settings": {**SETTINGS, "limit": limit, "index_input_sha256": fingerprint},
             "matches": matches}
    write_json(HISTORY / f"{run}.search.json", entry)
    return entry


def save_answer(run, payload):
    if not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{12}", run):
        raise ValueError("Invalid run ID")
    history = read_json(HISTORY / f"{run}.search.json")
    if payload["decision"] not in DECISIONS or not payload["answer"].strip():
        raise ValueError("Non-empty answer and valid decision required")
    if payload["decision"] in {"answer", "escalate"} and not payload["citations"]:
        raise ValueError("This decision requires citations")
    hits = {item["id"]: item for item in history["matches"]}
    citations = []
    for spec in payload["citations"]:
        hit = hits[spec["hit"]]
        fragment = next((piece for piece in hit["fragments"] if piece["id"] == spec["fragment"]), None)
        if fragment is None or not spec["quote"].strip() or spec["quote"] not in fragment["text"]:
            raise ValueError(f"Citation must be a verbatim extract from {spec['hit']}")
        citation = {"hit": spec["hit"], "fragment": spec["fragment"], "quote": spec["quote"],
                    "source": hit["source"], "location": fragment["location"]}
        if hit["language"] != "ru":
            citation["original"] = fragment["original"]
        citations.append(citation)
    response = {"answer": payload["answer"], "citations": citations, "decision": payload["decision"]}
    write_json(HISTORY / f"{run}.json", {**history, **response})
    print(render_json(response), end="")


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    ask = commands.add_parser("search")
    ask.add_argument("--role", required=True, choices=ROLES)
    ask.add_argument("--limit", type=int, default=14)
    ask.add_argument("question")
    batch = commands.add_parser("batch-search")
    batch.add_argument("input", type=Path, help="JSON array with role and question only")
    batch.add_argument("--limit", type=int, default=14)
    answer = commands.add_parser("save-answer")
    answer.add_argument("run")
    answer.add_argument("payload", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "save-answer":
        save_answer(args.run, read_json(args.payload))
    else:
        corpus, embeddings, fingerprint = load_index()
        model = load_encoder()
        if args.command == "search":
            entry = run_query(args.role, args.question, args.limit, corpus, embeddings, fingerprint, model)
            print(render_json(entry), end="")
        else:
            for item in read_json(args.input):
                if item.keys() != {"role", "question"}:
                    raise ValueError("Batch input must contain only role and question per item")
                entry = run_query(item["role"], item["question"], args.limit, corpus, embeddings, fingerprint, model)
                print(json.dumps({"run": entry["run"], **item}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

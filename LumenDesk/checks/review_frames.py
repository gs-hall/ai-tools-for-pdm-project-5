import sys
from pathlib import Path
from collections import Counter
import json
import os
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag

frame = int(sys.argv[1])
registry = rag.read_json(Path("registry.json"))
originals = rag.doc_map(Path("prepared/original"))
translations = rag.doc_map(Path("prepared/ru"))
if frame == 1:
    source_files = [
        f.as_posix()
        for f in Path("documents").rglob("*")
        if f.is_file() and f.suffix in {".md", ".txt", ".csv", ".xlsx", ".pdf", ".docx"}
    ]
    assert len(source_files) == len(originals) == len(registry) == 50
    assert len(translations) == 26
    print("50 originals | 50 extracted copies | 26 translations")
    print(
        ", ".join(
            f"{k}={v}"
            for k, v in sorted(Counter(Path(s).suffix for s in source_files).items())
        )
    )
    print(
        ", ".join(
            f"{k}={v}"
            for k, v in sorted(Counter(r["language"] for r in registry).items())
        )
    )
    for record in registry:
        if Path(record["source"]).name in {"Ротация ключей.md", "Выгрузка карточек.md"}:
            print(
                Path(record["source"]).name,
                "→",
                ", ".join(Path(s).name for s in record.get("duplicates", [])),
            )
elif frame == 2:
    source = next(s for s in originals if s.endswith("SAML cutover - September.md"))
    o = originals[source]["fragments"][2]
    t = translations[source]["fragments"]["3"]
    assert 3 in rag.reviewed_translation_ids(translations[source])
    for phrase in (
        "Keep one registered emergency administrator outside SAML enforcement.",
        "If a privileged account is locked out, route the case to the support lead immediately.",
    ):
        assert phrase in o["original"]
        print("EN:", phrase)
    for phrase in (
        "Оставьте одного зарегистрированного аварийного администратора вне обязательного входа через SAML.",
        "Если заблокирована привилегированная учётная запись, немедленно передайте обращение руководителю поддержки.",
    ):
        assert phrase in t
        print("RU:", phrase)
    print(Path(source).name, "строки", *o["location"]["lines"], "— сверено по смыслу")
else:
    mapping = {3: 10, 4: 3, 5: 4, 6: 6, 7: 7, 8: 9, 9: 12}
    question = rag.read_json(Path("checks/questions.json"))[mapping[frame] - 1]
    keep = [entry.split("=", 1) for entry in sys.argv[2:]]
    os.environ.update(
        HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="5"
    )
    result = subprocess.check_output(
        [
            sys.executable,
            "rag.py",
            "search",
            "--role",
            question["role"],
            question["question"],
        ],
        text=True,
    )
    entry = json.loads(result)
    print(f"rag.py search --role {entry['role']} | BGE-M3 | {entry['run']}")
    if not keep:
        print("Фрагменты из разрешённых источников:", len(entry["matches"]))
        print(
            "Источники с «злоупотреблен» в названии:",
            sum("злоупотреблен" in h["source"].lower() for h in entry["matches"]),
        )
        print(
            "Первые источники:",
            ", ".join(Path(h["source"]).name for h in entry["matches"][:5]),
        )
    for hit in entry["matches"]:
        for name, fragment_ids in keep:
            if not hit["source"].endswith(name):
                continue
            for f in hit["fragments"]:
                if str(f["id"]) in fragment_ids.split(","):
                    location = {
                        k: v
                        for k, v in f["location"].items()
                        if k not in {"headers", "columns", "section"}
                    }
                    print(hit["id"], Path(hit["source"]).name, location)
                    t = f["text"]
                    if len(t) > 820:
                        excerpt = [
                            s
                            for s in t.splitlines()
                            if any(
                                term in s
                                for term in (
                                    "15 процентов",
                                    "20 процентов",
                                    "14 июля",
                                    "09:00",
                                    "45 минут",
                                    "совместно",
                                    "не позднее",
                                    "120 минут",
                                )
                            )
                        ]
                        t = "\n".join(excerpt)[:820] if excerpt else t[:820]
                    print(t)

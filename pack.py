"""Region packs: a small, self-contained index for staff deployed to one country.

A pack holds the country's documents, all clinical guidelines and central protocols
(they apply everywhere), the country's field experiences, and the matching embeddings.
Run a device on a pack:  FIELDMED_PACK=data/packs/DR_Congo uvicorn app.server:app
"""
import json
import re
import shutil
import sqlite3

import numpy as np

from .build_index import SCHEMA
from .config import DATA, DB_PATH, EMB_PATH
from .sync import PROTO_GENRE

PACKS = DATA / "packs"
GUIDELINE_GENRES = ("Clinical Guideline Chapter", PROTO_GENRE)


def slug(country: str) -> str:
    return re.sub(r"\W+", "_", country).strip("_")


def build(country: str, guidelines: bool = True, src_db=DB_PATH) -> dict:
    out = PACKS / slug(country)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    src = sqlite3.connect(src_db)
    src.row_factory = sqlite3.Row
    docs = [d for d in src.execute("SELECT * FROM docs WHERE duplicate_of IS NULL")
            if country in (d["countries"] or "").split(", ")
            or (guidelines and d["genre"] in GUIDELINE_GENRES)]
    ids = {d["id"] for d in docs}

    emb = np.load(EMB_PATH) if EMB_PATH.exists() else None
    has_vec = lambda cid: emb is not None and cid <= len(emb)
    chunks = [c for c in src.execute("SELECT * FROM chunks ORDER BY id") if c["doc_id"] in ids]
    # Chunks with embeddings first, so pack row i of embeddings.npy is pack chunk id i+1.
    chunks.sort(key=lambda c: not has_vec(c["id"]))

    db = sqlite3.connect(out / "index.sqlite")
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?)", [tuple(d) for d in docs])
    titles = {d["id"]: d["title"] for d in docs}
    for new_id, c in enumerate(chunks, 1):
        db.execute("INSERT INTO chunks VALUES (?,?,?,?)", (new_id, c["doc_id"], c["heading"], c["text"]))
        db.execute("INSERT INTO chunks_fts(rowid, title, heading, text) VALUES (?,?,?,?)",
                   (new_id, titles[c["doc_id"]], c["heading"], c["text"]))
    db.commit()
    db.close()
    n_vec = sum(has_vec(c["id"]) for c in chunks)
    if n_vec:
        np.save(out / "embeddings.npy", emb[[c["id"] - 1 for c in chunks[:n_vec]]])

    manifest = {"country": country, "guidelines": guidelines, "docs": len(docs), "chunks": len(chunks),
                "has_vectors": bool(n_vec),
                "experiences": sum(d["ext"] == "exp" for d in docs),
                "protocols": sum(d["ext"] == "proto" for d in docs)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = shutil.make_archive(str(out), "zip", out)
    full = DB_PATH.stat().st_size + (EMB_PATH.stat().st_size if EMB_PATH.exists() else 0)
    return manifest | {"zip": f"{slug(country)}.zip", "size_mb": round(PACKS.joinpath(archive).stat().st_size / 1e6, 1),
                       "full_index_mb": round(full / 1e6, 1)}

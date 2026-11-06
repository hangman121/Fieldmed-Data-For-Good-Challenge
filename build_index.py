"""Off-site step 1: build data/index.sqlite (docs, chunks, FTS5 BM25 index).

Run from fieldmed/:  python -m pipeline.build_index
"""
import csv
import hashlib
import sqlite3

from .config import DATA, DB_PATH, META_CSV, ORIGINAL, PARSED
from .prepare import COUNTRIES, DISEASES, chunks, clean, tags, title_of

SCHEMA = """
CREATE TABLE docs(id TEXT PRIMARY KEY, title TEXT, genre TEXT, language TEXT, ext TEXT,
                  words INTEGER, countries TEXT, diseases TEXT, duplicate_of TEXT);
CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id TEXT, heading TEXT, text TEXT);
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, heading, text, content='',
                                           tokenize='porter unicode61 remove_diacritics 2');
"""


def main():
    DATA.mkdir(exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)

    meta = {r["id"]: r for r in csv.DictReader(open(META_CSV, encoding="utf-8"))}
    ext_of = {p.stem: p.suffix[1:] for p in ORIGINAL.iterdir()}
    seen, n_docs, n_dupes = {}, 0, 0

    # Sorted so D-files win over the P-files that duplicate them.
    for path in sorted(PARSED.glob("*.md")):
        doc_id, ext = path.stem, ext_of[path.stem]
        text = clean(path.read_text(encoding="utf-8"), from_pdf=(ext == "pdf"))
        digest = hashlib.md5(text.encode()).hexdigest()
        m = meta.get(doc_id, {})
        row = [doc_id, title_of(doc_id, ext, text), m.get("genre", ""), m.get("language", ""), ext,
               len(text.split()), ", ".join(tags(text, COUNTRIES)), ", ".join(tags(text, DISEASES)),
               seen.get(digest)]
        db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?)", row)
        if digest in seen:
            n_dupes += 1
            continue
        seen[digest] = doc_id
        n_docs += 1
        for heading, chunk in chunks(text):
            cur = db.execute("INSERT INTO chunks(doc_id, heading, text) VALUES (?,?,?)",
                             (doc_id, heading, chunk))
            db.execute("INSERT INTO chunks_fts(rowid, title, heading, text) VALUES (?,?,?,?)",
                       (cur.lastrowid, row[1], heading, chunk))
    db.commit()
    n_chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"indexed {n_docs} docs ({n_dupes} duplicates skipped), {n_chunks} chunks -> {DB_PATH}")


if __name__ == "__main__":
    main()

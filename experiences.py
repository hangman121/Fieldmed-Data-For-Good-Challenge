"""Field experiences: stored in their own SQLite file (survives index rebuilds),
indexed into the search index so they are findable as soon as they are saved.

Sync model: entries are never edited in place, only added (and approved), so a site
exchanges "everything newer than my last sync" as a JSON change file without conflicts.
"""
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import DATA

EXP_DB = DATA / "experiences.sqlite"
FIELDS = ["author", "role", "country", "region", "topic", "facility", "period",
          "situation", "worked", "lesson"]
GENRE = "Field Experience (unverified)"

# Things that look like patient identifiers: phone numbers, e-mails, long ID numbers.
PII = re.compile(r"\+?\d[\d \-]{7,}\d|[\w.]+@[\w.]+\.\w+|\b\d{6,}\b")


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(EXP_DB, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute(f"""CREATE TABLE IF NOT EXISTS experiences(id TEXT PRIMARY KEY, created TEXT,
                   {', '.join(f + ' TEXT' for f in FIELDS)}, status TEXT DEFAULT 'unverified')""")
    return db


def pii_warnings(entry: dict) -> list[str]:
    return [f for f in ("situation", "worked", "lesson") if PII.search(entry.get(f) or "")]


def add(db: sqlite3.Connection, entry: dict) -> dict:
    row = {f: (entry.get(f) or "").strip() for f in FIELDS}
    row["id"] = entry.get("id") or "X" + uuid.uuid4().hex[:10]
    row["created"] = entry.get("created") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["status"] = entry.get("status") or "unverified"
    cols = ["id", "created", *FIELDS, "status"]
    db.execute(f"INSERT OR IGNORE INTO experiences({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
               [row[c] for c in cols])
    db.commit()
    return row


def since(db: sqlite3.Connection, ts: str = "") -> list[dict]:
    return [dict(r) for r in db.execute("SELECT * FROM experiences WHERE created > ? ORDER BY created", (ts,))]


def index_into(index_db: sqlite3.Connection, exp: dict):
    """Add one experience to the search index (docs + chunks + FTS) if it is not there yet."""
    if index_db.execute("SELECT 1 FROM docs WHERE id = ?", (exp["id"],)).fetchone():
        return
    title = f'Field experience: {exp["topic"]} in {exp["country"]} ({exp["period"]})'
    text = (f'{exp["role"]} at a {exp["facility"]} in {exp["region"]}, {exp["country"]}. '
            f'Situation: {exp["situation"]} What worked or failed: {exp["worked"]} '
            f'Lesson: {exp["lesson"]}')
    index_db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?)",
                     (exp["id"], title, GENRE, "en", "exp", len(text.split()),
                      exp["country"], exp["topic"], None))
    cur = index_db.execute("INSERT INTO chunks(doc_id, heading, text) VALUES (?,?,?)",
                           (exp["id"], "Field experience (unverified)", text))
    index_db.execute("INSERT INTO chunks_fts(rowid, title, heading, text) VALUES (?,?,?,?)",
                     (cur.lastrowid, title, "Field experience (unverified)", text))
    index_db.commit()

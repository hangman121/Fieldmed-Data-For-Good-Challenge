"""Knowledge gaps: questions the system could not answer, grouped by area and topic.

An area with ALERT_AT or more unanswered questions on one topic within DAYS days is
flagged as needing more data and protocols.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

from .config import DATA
from .prepare import COUNTRIES, DISEASES, tags

GAPS_DB = DATA / "gaps.sqlite"
ALERT_AT = 3
DAYS = 30


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(GAPS_DB, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS gaps(id INTEGER PRIMARY KEY, created TEXT, q TEXT,
                  area TEXT, region TEXT, topic TEXT, reason TEXT)""")
    return db


def log(db: sqlite3.Connection, q: str, site_country: str, site_region: str, reason: str):
    # A country named in the question wins over the asker's site.
    area = (tags(q, COUNTRIES, min_hits=1) or [site_country or "Unknown"])[0]
    topic = (tags(q, DISEASES, min_hits=1) or ["Other / unclassified"])[0]
    db.execute("INSERT INTO gaps(created, q, area, region, topic, reason) VALUES (?,?,?,?,?,?)",
               (datetime.now(timezone.utc).isoformat(timespec="seconds"), q.strip(), area,
                site_region.strip(), topic, reason))
    db.commit()


def report(db: sqlite3.Connection) -> dict:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat(timespec="seconds")
    groups = [dict(r) for r in db.execute(
        """SELECT area, topic, count(*) AS n, max(created) AS last,
                  group_concat(DISTINCT region) AS regions, group_concat(q, ' || ') AS questions
           FROM gaps WHERE created > ? GROUP BY area, topic ORDER BY n DESC, last DESC""", (cutoff,))]
    for g in groups:
        g["questions"] = list(dict.fromkeys(g["questions"].split(" || ")))[:5]
        g["regions"] = ", ".join(filter(None, (g["regions"] or "").split(",")))
        g["alert"] = g["n"] >= ALERT_AT
    recent = [dict(r) for r in db.execute("SELECT * FROM gaps ORDER BY created DESC LIMIT 30")]
    return {"alert_at": ALERT_AT, "days": DAYS, "groups": groups,
            "alerts": [g for g in groups if g["alert"]], "recent": recent}

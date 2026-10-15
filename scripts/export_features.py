"""Export one row of features per raised nudge from Vesper's event log.

Only derived features leave the database: no nudge text, no evidence lines, no
verdict notes and no full rule keys (older keys embed assignment names). The
output goes to data/private/, which is gitignored; only aggregates are public.

    uv run python scripts/export_features.py [path/to/vesper.db]
"""

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "vesper/data/vesper.db")
OUT = Path(__file__).resolve().parent.parent / "data/private/nudges.csv"

# Keys from `ruleKeyFor` look like "email:<hash>" or "due:<slug>". Earlier keys
# were invented by the model ("overdue-math20550-12-4-part-2"), so those are
# mapped onto the same categories and flagged as legacy.
CURRENT = {"email", "due", "overdue", "reminder", "unmatched", "habit", "study"}


def category(key: str) -> tuple[str, bool]:
    head = key.split(":", 1)[0]
    if ":" in key and head in CURRENT:
        return head, False
    k = key.lower()
    if "gym" in k or k.startswith("habit"):
        return "habit", True
    if k.startswith("overdue"):
        return "overdue", True
    if "unstarted" in k or "unsubmitted" in k:
        return "unstarted", True
    if "class-starting" in k:
        return "class", True
    if any(s in k for s in ("due-today", "due-tomorrow", "problem-set", "reading-assignment")):
        return "due", True
    if head in CURRENT:
        return head, True
    return "other", True


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    raised = con.execute(
        "select ts, json_extract(payload,'$.nudgeId'), json_extract(payload,'$.ruleKey'),"
        " json_extract(payload,'$.urgency'), json_extract(payload,'$.confidence'),"
        " length(json_extract(payload,'$.text')), json_array_length(json_extract(payload,'$.evidence'))"
        " from events where kind='nudge.raised' order by seq"
    ).fetchall()
    judged = {
        nid: (ts, verdict)
        for ts, nid, verdict in con.execute(
            "select ts, json_extract(payload,'$.nudgeId'), json_extract(payload,'$.verdict')"
            " from events where kind='nudge.judged' order by seq"
        )
    }  # a later verdict on the same nudge replaces an earlier one
    con.close()

    seen: Counter[str] = Counter()
    start = raised[0][0]
    rows = []
    for ts, nid, key, urgency, conf, text_len, ev_n in raised:
        cat, legacy = category(key or "")
        at = datetime.fromtimestamp(ts / 1000)  # local time on this Mac
        j = judged.get(nid)
        rows.append(
            {
                "day": (ts - start) // 86_400_000,
                "hour": at.hour,
                "weekday": at.strftime("%a"),
                "category": cat,
                "legacy_key": legacy,
                "urgency": urgency,
                "confidence": conf,
                "text_len": text_len,
                "evidence_n": ev_n or 0,
                "prior_fires": seen[key],
                "verdict": j[1] if j else None,
                "minutes_to_verdict": round((j[0] - ts) / 60_000, 1) if j else None,
            }
        )
        seen[key] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"{OUT}: {len(df)} nudges, {df.verdict.notna().sum()} judged, columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()

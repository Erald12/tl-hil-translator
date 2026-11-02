"""
database_builder.py  --  builds translations.db from phrase_data.py and provides
typo-tolerant (fuzzy) search in both directions.

Build (PC):
    python database_builder.py                 # creates/updates translations.db
    python database_builder.py --reset         # drop and rebuild from scratch

Search (PC or phone):
    python database_builder.py search "maayong gabe"           # auto-detect direction
    python database_builder.py search "kumusta ka" --dir tgl2hil
    python database_builder.py interactive

From code (e.g. via Chaquopy on Android):
    from database_builder import PhraseSearcher
    s = PhraseSearcher("translations.db")
    s.search("salamt")          # -> list of matches, best first
    s.best("maayong agaa", "hil2tgl")

Fuzzy engine: uses `rapidfuzz` when installed (PC). Chaquopy has no rapidfuzz
build for Android, so on the phone it automatically falls back to an equivalent
scorer built on Python's standard-library difflib. With ~1,300 phrases that
takes a few milliseconds per search either way.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sqlite3
import unicodedata
from pathlib import Path

try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process

    HAVE_RAPIDFUZZ = True
except ImportError:  # Android / Chaquopy
    HAVE_RAPIDFUZZ = False

DEFAULT_DB = Path(__file__).with_name("translations.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tagalog     TEXT NOT NULL,
    hiligaynon  TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    confidence  TEXT NOT NULL DEFAULT 'normal',   -- 'normal' | 'low' (AI was unsure)
    verified    INTEGER NOT NULL DEFAULT 0,       -- set to 1 after native-speaker review
    tgl_norm    TEXT NOT NULL,
    hil_norm    TEXT NOT NULL,
    UNIQUE (tagalog, hiligaynon, note)
);
CREATE INDEX IF NOT EXISTS idx_tgl_norm ON translations (tgl_norm);
CREATE INDEX IF NOT EXISTS idx_hil_norm ON translations (hil_norm);
"""

_PAREN = re.compile(r"\([^)]*\)")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation/hyphens, collapse spaces.
    'Gab-i?' -> 'gabi', 'Ñino' -> 'nino', 'Opo (add Nong)' -> 'opo'."""
    text = _PAREN.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("-", "").replace("'", "")
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def alternatives(text: str) -> list[str]:
    """'Okay lang / Maayo lang' -> ['okay lang', 'maayo lang'] (normalized)."""
    return [n for n in (normalize(p) for p in text.split("/")) if n]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_database(db_path: str | Path = DEFAULT_DB, reset: bool = False) -> int:
    from phrase_data import ENTRIES

    con = sqlite3.connect(str(db_path))
    try:
        if reset:
            con.execute("DROP TABLE IF EXISTS translations")
        con.executescript(SCHEMA)
        rows = [
            (e["tagalog"], e["hiligaynon"], e["category"], e["note"], e["confidence"],
             normalize(e["tagalog"]), normalize(e["hiligaynon"]))
            for e in ENTRIES
        ]
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO translations
               (tagalog, hiligaynon, category, note, confidence, tgl_norm, hil_norm)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        con.commit()
        inserted = con.total_changes - before
        total = con.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    finally:
        con.close()
    print(f"Inserted {inserted} new rows; {total} rows total in {db_path}")
    return total


# --------------------------------------------------------------------------- #
# Fuzzy search
# --------------------------------------------------------------------------- #
def _token_sort(s: str) -> str:
    return " ".join(sorted(s.split()))


def _difflib_score(a: str, b: str) -> float:
    """Pure-Python stand-in for rapidfuzz.fuzz.token_sort_ratio (0-100)."""
    return difflib.SequenceMatcher(None, _token_sort(a), _token_sort(b), autojunk=False).ratio() * 100


class PhraseSearcher:
    """Loads the DB into memory once, then answers fuzzy lookups quickly.

    direction: 'tgl2hil' (query is Tagalog), 'hil2tgl' (query is Hiligaynon), or 'auto'.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            self.rows = [dict(r) for r in con.execute("SELECT * FROM translations")]
        finally:
            con.close()
        # Search keys per direction: (normalized key, row index)
        self._keys: dict[str, list[tuple[str, int]]] = {"tgl2hil": [], "hil2tgl": []}
        for i, r in enumerate(self.rows):
            for k in alternatives(r["tagalog"]):
                self._keys["tgl2hil"].append((k, i))
            for k in alternatives(r["hiligaynon"]):
                self._keys["hil2tgl"].append((k, i))
        self._choices = {d: [k for k, _ in v] for d, v in self._keys.items()}

    def _scored(self, q: str, direction: str, limit: int, cutoff: float) -> list[tuple[float, int]]:
        keys, choices = self._keys[direction], self._choices[direction]
        if HAVE_RAPIDFUZZ:
            hits = _rf_process.extract(q, choices, scorer=_rf_fuzz.token_sort_ratio,
                                       limit=limit * 3, score_cutoff=cutoff)
            scored = [(float(score), keys[idx][1]) for _, score, idx in hits]
        else:
            scored = []
            for k, row_idx in keys:
                # cheap length filter before the slower difflib comparison
                if abs(len(k) - len(q)) > max(len(k), len(q)) * (1 - cutoff / 100) + 1:
                    continue
                s = _difflib_score(q, k)
                if s >= cutoff:
                    scored.append((s, row_idx))
        # exact matches always win
        for k, row_idx in keys:
            if k == q:
                scored.append((101.0, row_idx))
        return scored

    def search(self, query: str, direction: str = "auto", limit: int = 5,
               score_cutoff: float = 70.0) -> list[dict]:
        """Return up to `limit` matches, best first. Each match:
        {score, direction, source, translation, tagalog, hiligaynon, category, note, confidence, verified}
        """
        q = normalize(query)
        if not q:
            return []
        dirs = ["tgl2hil", "hil2tgl"] if direction == "auto" else [direction]
        best: dict[tuple[str, int], float] = {}
        for d in dirs:
            for score, row_idx in self._scored(q, d, limit, score_cutoff):
                key = (d, row_idx)
                best[key] = max(best.get(key, 0.0), score)

        ranked = sorted(best.items(), key=lambda kv: (-kv[1], self.rows[kv[0][1]]["confidence"] == "low"))
        out = []
        for (d, row_idx), score in ranked[:limit]:
            r = self.rows[row_idx]
            src, tgt = ("tagalog", "hiligaynon") if d == "tgl2hil" else ("hiligaynon", "tagalog")
            out.append({
                "score": round(min(score, 100.0), 1),
                "exact": score > 100,
                "direction": d,
                "source": r[src],
                "translation": r[tgt],
                **{k: r[k] for k in ("tagalog", "hiligaynon", "category", "note", "confidence", "verified")},
            })
        return out

    def best(self, query: str, direction: str = "auto", score_cutoff: float = 70.0) -> str | None:
        """Just the best translation string, or None if nothing is close enough."""
        hits = self.search(query, direction, limit=1, score_cutoff=score_cutoff)
        return hits[0]["translation"] if hits else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_hits(hits: list[dict]) -> None:
    if not hits:
        print("  (no match)")
    for h in hits:
        flag = " [LOW CONFIDENCE]" if h["confidence"] == "low" else ""
        note = f" ({h['note']})" if h["note"] else ""
        print(f"  {h['score']:5.1f}  {h['direction']}  {h['source']}{note}  ->  {h['translation']}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build / search translations.db")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--reset", action="store_true", help="drop and rebuild the table")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--dir", default="auto", choices=["auto", "tgl2hil", "hil2tgl"])
    sp.add_argument("--limit", type=int, default=5)
    sub.add_parser("interactive")
    args = ap.parse_args()

    if args.cmd is None:
        build_database(args.db, reset=args.reset)
        return
    if not Path(args.db).exists():
        build_database(args.db)

    s = PhraseSearcher(args.db)
    print(f"fuzzy engine: {'rapidfuzz' if HAVE_RAPIDFUZZ else 'difflib fallback'}")
    if args.cmd == "search":
        _print_hits(s.search(args.query, args.dir, args.limit))
    else:
        while True:
            try:
                q = input("\nsearch> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                _print_hits(s.search(q))


if __name__ == "__main__":
    main()

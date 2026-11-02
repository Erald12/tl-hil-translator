"""
fetch_corpus.py  --  download, clean, deduplicate and package open Tagalog <-> Hiligaynon
parallel data into one SQLite database (and optional JSON) for an offline Android app.

Standard library only (Python 3.9+). Run on your PC:

    python fetch_corpus.py                       # all sources -> corpus_out/corpus.db
    python fetch_corpus.py --json                # also write corpus_out/corpus.json
    python fetch_corpus.py --sources bible,tatoeba,opus       # skip the 153 MB Wiktionary download
    python fetch_corpus.py --include-phrases     # also merge phrase_data.py (AI-generated, unverified)
    python fetch_corpus.py --offline             # rebuild from already-downloaded files only
    python fetch_corpus.py --no-fts              # smaller DB without the full-text search index

What actually exists (checked 2026-11-02) -- direct Tagalog<->Hiligaynon data is scarce:
  source       pairs (approx)  kind       quality   license
  bible        ~30,300         sentence   high      CC BY-SA 4.0 (both Bibles)
  wiktionary   ~1,400          word       low       CC BY-SA 4.0 / GFDL (via kaikki.org)
  tatoeba      ~3              sentence   high      CC BY 2.0 FR
  opus         ~15             UI string  medium    see OPUS/Ubuntu (Ubuntu corpus)
OPUS itself lists only Ubuntu/Mozilla/Tatoeba/translatewiki for Hiligaynon, almost nothing
for hil-tl. The script queries the OPUS API live, so new corpora are picked up automatically.

NOTE ON LICENSES: CC BY-SA means you must credit the sources (the script writes
ATTRIBUTION.txt -- show it in your app's About screen) and share the *database* under
CC BY-SA too. Your app's own code can stay under any license.
NOTE ON DOMAIN: most sentence pairs are Bible verses (formal/religious register). Great for
vocabulary and grammar, but not conversational -- keep your phrase list for everyday speech.
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import io
import json
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

USER_AGENT = "tl-hil-offline-translator/1.0 (corpus builder; python-urllib)"


def _ssl_context():
    """System certificates + certifi's bundle (if installed).
    Windows Python only sees root certificates already cached in the Windows store, so some
    servers (e.g. object.pouta.csc.fi, used by OPUS) fail with CERTIFICATE_VERIFY_FAILED.
    `pip install certifi` fixes that; certificates are still fully verified."""
    import ssl
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


SSL_CONTEXT = _ssl_context()

# --------------------------------------------------------------------------- #
# Exact source URLs / API endpoints
# --------------------------------------------------------------------------- #
SOURCES = {
    "bible": {
        "description": "eBible.org verse-per-line (VPL) Bibles, aligned by book:chapter:verse",
        "urls": {
            "hil": "https://ebible.org/Scriptures/hil_vpl.zip",      # Biblica Libre Ang Pulong Sang Dios
            "tgl": "https://ebible.org/Scriptures/tglulb_vpl.zip",   # Tagalog Unlocked Literal Bible
        },
        "info_urls": ["https://ebible.org/hil/copyright.htm", "https://ebible.org/tglulb/copyright.htm"],
        "license": "CC BY-SA 4.0",
        "attribution": (
            "Hiligaynon: Biblica(R) Libre Ang Pulong Sang Dios, Copyright (c) 1996, 2006, 2011, 2022 "
            "Biblica, Inc., CC BY-SA 4.0. Tagalog: Unlocked Literal Bible (tglulb), Copyright (c) 2018 "
            "Door43 World Missions Community, CC BY-SA 4.0. Both obtained from https://ebible.org"
        ),
    },
    "tatoeba": {
        "description": "Tatoeba per-language exports + hil-tgl translation links",
        "urls": {
            "hil_sentences": "https://downloads.tatoeba.org/exports/per_language/hil/hil_sentences.tsv.bz2",
            "tgl_sentences": "https://downloads.tatoeba.org/exports/per_language/tgl/tgl_sentences.tsv.bz2",
            "links": "https://downloads.tatoeba.org/exports/per_language/hil/hil-tgl_links.tsv.bz2",
        },
        "license": "CC BY 2.0 FR",
        "attribution": "Sentences from Tatoeba (https://tatoeba.org), CC BY 2.0 FR.",
    },
    "opus": {
        "description": "OPUS API: every hil-tl corpus in Moses (plain text) format",
        # No trailing slash: '/opusapi/?' answers 308, which older urllib versions don't follow.
        "api": "https://opus.nlpl.eu/opusapi?source=hil&target=tl&preprocessing=moses&version=latest",
        "license": "per corpus, see https://opus.nlpl.eu",
        "attribution": "OPUS parallel corpora (https://opus.nlpl.eu), J. Tiedemann, LREC 2012.",
    },
    "wiktionary": {
        "description": "Wiktionary (English edition) via kaikki.org; Tagalog & Hiligaynon words "
                       "paired when they share the same part of speech + English gloss",
        "urls": {
            "hil": "https://kaikki.org/dictionary/Hiligaynon/kaikki.org-dictionary-Hiligaynon.jsonl",  # ~2 MB
            "tgl": "https://kaikki.org/dictionary/Tagalog/kaikki.org-dictionary-Tagalog.jsonl",        # ~153 MB
        },
        "license": "CC BY-SA 4.0 / GFDL",
        "attribution": "Wiktionary contributors (https://en.wiktionary.org), extracted by "
                       "Wiktextract/kaikki.org (https://kaikki.org), CC BY-SA 4.0.",
    },
    "phrases": {
        "description": "Local phrase_data.py (AI-generated, needs native-speaker review)",
        "license": "your own",
        "attribution": "Phrase list generated with AI assistance; unverified.",
    },
}

DEFAULT_SOURCES = ["bible", "tatoeba", "opus", "wiktionary"]
QUALITY_RANK = {"high": 0, "medium": 1, "low": 2, "unverified": 3}


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, offline: bool = False, retries: int = 4) -> Path:
    """Download url -> dest (resumable, retried). Skips if already complete."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        raise FileNotFoundError(f"--offline given but {dest} is missing ({url})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as r:
                if have and r.status != 206:      # server ignored Range -> start over
                    have = 0
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                with open(part, "ab" if have else "wb") as f:
                    done = have
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                print()
            if total and part.stat().st_size != total:
                raise IOError(f"incomplete: {part.stat().st_size} of {total} bytes")
            part.replace(dest)
            return dest
        except Exception as e:  # network errors: retry with backoff, resuming the .part file
            if "CERTIFICATE_VERIFY_FAILED" in str(e):   # retrying won't help
                raise RuntimeError(
                    f"SSL certificate check failed for {url}. Fix: run  pip install certifi  "
                    "and start the script again (finished downloads are kept)."
                ) from e
            print(f"\n  ! {url} attempt {attempt}/{retries}: {e}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return dest


def fetch_json(url: str, cache: Path, offline: bool) -> dict:
    return json.loads(download(url, cache, offline).read_bytes().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACE_LIKE = dict.fromkeys(map(ord, "           　\t"), " ")
_QUOTES = {ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("‘"): "'", ord("’"): "'",
           ord("–"): "-", ord("—"): " - "}
_BRACKETED = re.compile(r"\[([^\]]*)\]")          # Bible "supplied words": keep text, drop brackets
_MULTISPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")
_NORM_STRIP = re.compile(r"[^a-z0-9ñ ]+")


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_ZERO_WIDTH).translate(_SPACE_LIKE).translate(_QUOTES)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")   # control chars (e.g. \x81)
    s = s.replace("''", '"').replace("*", "")
    s = _BRACKETED.sub(r"\1", s)
    s = _MULTISPACE.sub(" ", s)
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    return s.strip(" -")


def norm_key(s: str) -> str:
    """Dedup key: lowercase, no accents except ñ, no punctuation/hyphens."""
    s = unicodedata.normalize("NFKD", s.lower().replace("ñ", "\0"))
    s = "".join(c for c in s if not unicodedata.combining(c)).replace("\0", "ñ")
    s = s.replace("-", "").replace("'", "")
    return _MULTISPACE.sub(" ", _NORM_STRIP.sub(" ", s)).strip()


def acceptable(tgl: str, hil: str, kind: str) -> bool:
    """Drop empty, oversized, misaligned or untranslated pairs."""
    if not tgl or not hil or not re.search(r"[A-Za-z]", tgl) or not re.search(r"[A-Za-z]", hil):
        return False
    if len(tgl) > 1500 or len(hil) > 1500:
        return False
    if kind == "sentence":
        short, long_ = sorted((len(tgl), len(hil)))
        if long_ > 30 and long_ / max(short, 1) > 2.5:         # likely misaligned
            return False
        if norm_key(tgl) == norm_key(hil) and len(tgl.split()) > 3:   # untranslated copy
            return False
    return True


# --------------------------------------------------------------------------- #
# Source parsers -> yield dict(tagalog, hiligaynon, kind, quality, ref)
# --------------------------------------------------------------------------- #
_VPL_LINE = re.compile(r"^(\S+) (\d+):(\d+) ?(.*)$")


def _read_vpl(zip_path: Path) -> dict[tuple[str, int], dict[int, str]]:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("_vpl.txt"))
        text = z.read(name).decode("utf-8-sig")
    chapters: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for line in text.splitlines():
        m = _VPL_LINE.match(line)
        if m and m.group(4).strip():
            book, ch, v, t = m.groups()
            chapters[(book, int(ch))][int(v)] = t
    return chapters


def parse_bible(raw: Path, offline: bool):
    urls = SOURCES["bible"]["urls"]
    hil = _read_vpl(download(urls["hil"], raw / "ebible" / "hil_vpl.zip", offline))
    tgl = _read_vpl(download(urls["tgl"], raw / "ebible" / "tglulb_vpl.zip", offline))
    for key in sorted(hil.keys() & tgl.keys()):
        h_ch, t_ch = hil[key], tgl[key]
        h_verses = sorted(h_ch)
        last_t = max(t_ch)
        for i, v in enumerate(h_verses):
            # The Hiligaynon text often bridges verses (e.g. 17-18 printed as one verse 17).
            # Merge the Tagalog verses up to the next Hiligaynon verse so meaning stays aligned.
            nxt = h_verses[i + 1] if i + 1 < len(h_verses) else last_t + 1
            span = [n for n in range(v, nxt) if n in t_ch]
            if not span:
                continue
            ref = f"{key[0]} {key[1]}:{v}" + (f"-{span[-1]}" if span[-1] != v else "")
            yield {
                "tagalog": " ".join(t_ch[n] for n in span),
                "hiligaynon": h_ch[v],
                "kind": "sentence",
                "quality": "high" if len(span) == 1 else "medium",
                "ref": ref,
            }


def _read_bz2_tsv(path: Path) -> list[list[str]]:
    with bz2.open(path, "rt", encoding="utf-8", newline="") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def parse_tatoeba(raw: Path, offline: bool):
    urls = SOURCES["tatoeba"]["urls"]
    d = raw / "tatoeba"
    hil = {r[0]: r[2] for r in _read_bz2_tsv(download(urls["hil_sentences"], d / "hil_sentences.tsv.bz2", offline)) if len(r) >= 3}
    tgl = {r[0]: r[2] for r in _read_bz2_tsv(download(urls["tgl_sentences"], d / "tgl_sentences.tsv.bz2", offline)) if len(r) >= 3}
    for row in _read_bz2_tsv(download(urls["links"], d / "hil-tgl_links.tsv.bz2", offline)):
        if len(row) >= 2 and row[0] in hil and row[1] in tgl:
            yield {"tagalog": tgl[row[1]], "hiligaynon": hil[row[0]], "kind": "sentence",
                   "quality": "high", "ref": f"tatoeba:{row[0]}-{row[1]}"}


def parse_opus(raw: Path, offline: bool):
    d = raw / "opus"
    listing = fetch_json(SOURCES["opus"]["api"], d / "api_hil-tl_moses.json", offline)
    for c in listing.get("corpora", []):
        url = c["url"]
        if not url.endswith(".txt.zip"):
            continue
        zpath = download(url, d / f"{c['corpus']}_{c['version']}_hil-tl.txt.zip", offline)
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            h_name = next((n for n in names if n.endswith(".hil")), None)
            t_name = next((n for n in names if n.endswith(".tl")), None)
            if not (h_name and t_name):
                continue
            h_lines = z.read(h_name).decode("utf-8", "replace").splitlines()
            t_lines = z.read(t_name).decode("utf-8", "replace").splitlines()
        if len(h_lines) != len(t_lines):
            print(f"  ! {c['corpus']}: line counts differ ({len(h_lines)} vs {len(t_lines)}), skipped")
            continue
        for i, (h, t) in enumerate(zip(h_lines, t_lines)):
            yield {"tagalog": t, "hiligaynon": h, "kind": "sentence", "quality": "medium",
                   "ref": f"opus:{c['corpus']}/{c['version']}#{i + 1}"}


_SKIP_POS = {"name", "character", "symbol", "suffix", "prefix", "infix", "affix",
             "circumfix", "phrase", "romanization", "punct"}
_PAREN = re.compile(r"\([^)]*\)")


def _wiktionary_index(path: Path) -> dict[tuple[str, str], set[str]]:
    """(pos, short English gloss) -> {words}. Only base entries, first gloss per sense."""
    idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    with open(path, "rb") as f:       # binary: some entries contain characters text mode treats as newlines
        for line in f:
            e = json.loads(line)
            if e.get("pos") in _SKIP_POS or " " in e.get("word", " ").strip():
                continue
            for s in e.get("senses", []):
                if "form_of" in s or "alt_of" in s or "form-of" in s.get("tags", []):
                    continue
                for g in s.get("glosses", [])[:1]:
                    g = _PAREN.sub("", g).strip().lower().rstrip(".")
                    g = g[3:] if g.startswith("to ") else g
                    if 0 < len(g.split()) <= 2:
                        idx[(e["pos"], g)].add(e["word"])
    return idx


def parse_wiktionary(raw: Path, offline: bool):
    urls = SOURCES["wiktionary"]["urls"]
    d = raw / "kaikki"
    hil = _wiktionary_index(download(urls["hil"], d / "hiligaynon.jsonl", offline))
    tgl = _wiktionary_index(download(urls["tgl"], d / "tagalog.jsonl", offline))
    for key in sorted(hil.keys() & tgl.keys()):
        h_words, t_words = sorted(hil[key]), sorted(tgl[key])
        if len(h_words) * len(t_words) > 6:       # very generic gloss -> too many wrong combinations
            continue
        for h in h_words:
            for t in t_words:
                yield {"tagalog": t, "hiligaynon": h, "kind": "word", "quality": "low",
                       "ref": f"gloss:{key[0]}:{key[1]}"}


def parse_phrases(raw: Path, offline: bool):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phrase_data import ENTRIES
    for e in ENTRIES:
        for h in e["hiligaynon"].split(" / "):
            yield {"tagalog": e["tagalog"], "hiligaynon": _PAREN.sub("", h).strip(),
                   "kind": "phrase" if " " in e["tagalog"] else "word",
                   "quality": "unverified", "ref": f"phrase_data:{e['category']}"}


PARSERS = {"bible": parse_bible, "tatoeba": parse_tatoeba, "opus": parse_opus,
           "wiktionary": parse_wiktionary, "phrases": parse_phrases}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
SCHEMA = """
DROP TABLE IF EXISTS pairs;
DROP TABLE IF EXISTS sources;
CREATE TABLE pairs (
    id          INTEGER PRIMARY KEY,
    tagalog     TEXT NOT NULL,
    hiligaynon  TEXT NOT NULL,
    kind        TEXT NOT NULL,      -- sentence | phrase | word
    source      TEXT NOT NULL,      -- bible | tatoeba | opus | wiktionary | phrases
    quality     TEXT NOT NULL,      -- high | medium | low | unverified
    ref         TEXT NOT NULL,      -- verse reference / corpus line / gloss
    tgl_norm    TEXT NOT NULL,      -- normalized lookup keys (lowercase, no punctuation);
    hil_norm    TEXT NOT NULL       -- uniqueness of (tgl_norm, hil_norm) is enforced in Python
);
-- Exact-lookup indexes only for short entries (words/phrases); sentences are searched via FTS5.
-- Keeps the file ~20 MB smaller than indexing every Bible verse.
CREATE INDEX idx_pairs_tgl ON pairs (tgl_norm) WHERE kind != 'sentence';
CREATE INDEX idx_pairs_hil ON pairs (hil_norm) WHERE kind != 'sentence';
CREATE INDEX idx_pairs_kind ON pairs (kind, quality);
CREATE TABLE sources (
    name TEXT PRIMARY KEY, description TEXT, urls TEXT, license TEXT,
    attribution TEXT, pairs INTEGER, fetched_at TEXT
);
"""


def build(selected: list[str], out_dir: Path, raw: Path, offline: bool, write_json: bool,
          no_fts: bool = False) -> None:
    pairs: dict[tuple[str, str], dict] = {}
    stats = {}
    for name in selected:
        print(f"[{name}] {SOURCES[name]['description']}")
        seen = kept = 0
        try:
            for p in PARSERS[name](raw, offline):
                seen += 1
                tgl, hil = clean_text(p["tagalog"]), clean_text(p["hiligaynon"])
                if not acceptable(tgl, hil, p["kind"]):
                    continue
                key = (norm_key(tgl), norm_key(hil))
                old = pairs.get(key)
                # duplicates: keep the higher-quality copy
                if old is None or QUALITY_RANK[p["quality"]] < QUALITY_RANK[old["quality"]]:
                    pairs[key] = {**p, "tagalog": tgl, "hiligaynon": hil, "source": name}
                    kept += 1 if old is None else 0
        except Exception as e:
            print(f"  ! {name} failed: {e!r} -- continuing with the other sources")
        stats[name] = kept
        print(f"  {seen} raw pairs -> {kept} new unique pairs after cleaning/dedup")

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "corpus.db"
    tmp_path = db_path.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)
    con = sqlite3.connect(tmp_path)
    con.executescript(SCHEMA)
    rows = sorted(pairs.items(), key=lambda kv: (QUALITY_RANK[kv[1]["quality"]], kv[1]["source"], kv[1]["ref"]))
    con.executemany(
        "INSERT INTO pairs (tagalog, hiligaynon, kind, source, quality, ref, tgl_norm, hil_norm) VALUES (?,?,?,?,?,?,?,?)",
        [(p["tagalog"], p["hiligaynon"], p["kind"], p["source"], p["quality"], p["ref"], k[0], k[1]) for k, p in rows],
    )
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    final_counts = dict(con.execute("SELECT source, COUNT(*) FROM pairs GROUP BY source").fetchall())
    for name in selected:
        s = SOURCES[name]
        urls = list(s.get("urls", {}).values()) + ([s["api"]] if "api" in s else [])
        con.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?)",
                    (name, s["description"], json.dumps(urls), s["license"], s["attribution"],
                     final_counts.get(name, 0), now))
    # Full-text search table, if this SQLite build has FTS5 (Android's usually does)
    fts = False
    if not no_fts:
        try:
            con.executescript("""
                CREATE VIRTUAL TABLE pairs_fts USING fts5(tagalog, hiligaynon, content='pairs', content_rowid='id');
                INSERT INTO pairs_fts(pairs_fts) VALUES ('rebuild');""")
            fts = True
        except sqlite3.OperationalError:
            print("  (this SQLite has no FTS5 -- skipped full-text index)")
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp_path.replace(db_path)

    attribution = ["Tagalog-Hiligaynon corpus -- data sources and licenses", ""]
    for name in selected:
        attribution += [f"* {name} ({SOURCES[name]['license']}): {SOURCES[name]['attribution']}"]
    attribution += ["", "The combined database is distributed under CC BY-SA 4.0."]
    (out_dir / "ATTRIBUTION.txt").write_text("\n".join(attribution) + "\n", encoding="utf-8")

    if write_json:
        doc = {
            "generated_at": now,
            "sources": {n: {k: SOURCES[n][k] for k in ("license", "attribution")} for n in selected},
            "pairs": [{"tgl": p["tagalog"], "hil": p["hiligaynon"], "kind": p["kind"],
                       "source": p["source"], "quality": p["quality"], "ref": p["ref"]} for _, p in rows],
        }
        (out_dir / "corpus.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")),
                                             encoding="utf-8")

    print(f"\nWrote {db_path} ({db_path.stat().st_size / 1e6:.1f} MB, {len(rows)} pairs, FTS5={'yes' if fts else 'no'})")
    for name in selected:
        print(f"  {name:11s} {final_counts.get(name, 0):6d}")
    if write_json:
        print(f"Wrote {out_dir / 'corpus.json'} ({(out_dir / 'corpus.json').stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {out_dir / 'ATTRIBUTION.txt'}  <- show this in your app")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help=f"comma list from {list(PARSERS)} (default: {','.join(DEFAULT_SOURCES)})")
    ap.add_argument("--include-phrases", action="store_true", help="also merge phrase_data.py")
    ap.add_argument("--out-dir", default="corpus_out", help="where corpus.db / corpus.json go")
    ap.add_argument("--cache-dir", default="corpus_cache", help="where raw downloads are kept")
    ap.add_argument("--json", action="store_true", help="also write corpus.json")
    ap.add_argument("--offline", action="store_true", help="only use files already in --cache-dir")
    ap.add_argument("--no-fts", action="store_true", help="skip the FTS5 full-text index (~6 MB smaller)")
    args = ap.parse_args()

    selected = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.include_phrases and "phrases" not in selected:
        selected.append("phrases")
    unknown = [s for s in selected if s not in PARSERS]
    if unknown:
        ap.error(f"unknown source(s): {unknown}")
    build(selected, Path(args.out_dir), Path(args.cache_dir) / "raw", args.offline, args.json, args.no_fts)


if __name__ == "__main__":
    main()

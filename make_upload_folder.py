"""
make_upload_folder.py  --  create  upload_to_github/  containing ONLY the files GitHub needs
(a few hundred KB), so you can drag-and-drop it on github.com without uploading the
500 MB models or other generated files.

    python make_upload_folder.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "upload_to_github"

FILES = [
    ".gitignore", "README.md", "how_to_run.txt",
    "model_setup.py", "translator.py", "phrase_data.py", "database_builder.py",
    "fetch_corpus.py", "prepare_android.py", "make_upload_folder.py",
    ".github/workflows/build-apk.yml",
]
ANDROID_DIR = "android-app"
SKIP_PARTS = {".gradle", "build", ".idea", "__pycache__", "assets", "python"}   # generated/copied at build time
SKIP_NAMES = {"local.properties"}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    count = 0
    for rel in FILES:
        src = ROOT / rel
        if src.exists():
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
        else:
            print(f"  (missing, skipped) {rel}")
    for src in (ROOT / ANDROID_DIR).rglob("*"):
        rel = src.relative_to(ROOT)
        if src.is_file() and not (set(rel.parts) & SKIP_PARTS) and src.name not in SKIP_NAMES:
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e3
    print(f"Created {OUT}  ({count} files, {size:.0f} KB)")
    print("Upload EVERYTHING inside this folder (including the .github folder) to your GitHub repository.")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print("   ", p.relative_to(OUT).as_posix())


if __name__ == "__main__":
    main()

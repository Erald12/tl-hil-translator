"""
prepare_android.py  --  copy the Python code and offline data into android-app/ before building.

Used automatically by the GitHub Actions build. Run it yourself only if you build
locally with Android Studio:   python prepare_android.py

  translator.py, database_builder.py  -> android-app/app/src/main/python/
  translations.db                     -> android-app/app/src/main/assets/       (both APKs)
  models/                             -> android-app/app/src/full/assets/models (full APK only)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "android-app" / "app" / "src"


def main() -> int:
    py_dir, main_assets, full_assets = APP / "main" / "python", APP / "main" / "assets", APP / "full" / "assets"
    for d in (py_dir, main_assets, full_assets):
        d.mkdir(parents=True, exist_ok=True)

    for name in ("translator.py", "database_builder.py"):
        shutil.copy2(ROOT / name, py_dir / name)
        print(f"copied {name} -> {py_dir.relative_to(ROOT)}")

    db = ROOT / "translations.db"
    if not db.exists():
        print("translations.db missing -> building it from phrase_data.py")
        sys.path.insert(0, str(ROOT))
        from database_builder import build_database
        build_database(db, reset=True)
    shutil.copy2(db, main_assets / "translations.db")
    print(f"copied translations.db -> {main_assets.relative_to(ROOT)}")

    models_src, models_dst = ROOT / "models", full_assets / "models"
    expected = ["tl-en", "en-hil", "hil-en", "en-tl"]
    missing = [m for m in expected if not (models_src / m / "meta.json").exists()]
    if missing:
        print(f"WARNING: models missing {missing} -> run model_setup.py first. "
              "The FULL APK would have no AI translation (LITE is unaffected).")
    if models_dst.exists():
        shutil.rmtree(models_dst)
    if models_src.exists():
        shutil.copytree(models_src, models_dst)
        size = sum(p.stat().st_size for p in models_dst.rglob("*") if p.is_file()) / 1e6
        print(f"copied models/ ({size:.0f} MB) -> {models_dst.relative_to(ROOT)}")
    return 1 if missing and "--strict" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())

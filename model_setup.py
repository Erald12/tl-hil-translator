"""
model_setup.py  --  run this ONCE on your PC (not on the phone).

Downloads small pre-trained OPUS-MT (MarianMT) translation models, exports
them to ONNX, quantizes them to INT8, and writes a phone-ready "models/"
folder that translator.py (and the Android app) can load fully offline.

Why not NLLB-200?
    NLLB-200 does NOT include Hiligaynon (there is no "hil_Latn" code in its
    tokenizer; the closest languages it has are ceb_Latn, war_Latn, ilo_Latn).
    It is also licensed CC-BY-NC (non-commercial only).

What we use instead (Helsinki-NLP OPUS-MT, CC-BY 4.0, ~75M params each):
    Tagalog    -> English   : Helsinki-NLP/opus-mt-tl-en
    English    -> Hiligaynon: Helsinki-NLP/opus-mt-en-hil
    Hiligaynon -> English   : Helsinki-NLP/opus-mt-hil-en
    English    -> Tagalog   : Helsinki-NLP/opus-mt-en-tl
There is no direct Tagalog<->Hiligaynon model, so translator.py goes through
English: tgl -> eng -> hil and hil -> eng -> tgl.

Result: 4 x ~135 MB INT8 ONNX models (~540 MB total, vs ~1.2 GB in FP32).
Only the 2 models for the current direction are loaded (~300-500 MB RAM),
which is comfortable on a vivo Y27 (6 GB).

Install (PC, Python 3.9-3.12):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers sentencepiece onnx onnxruntime "optimum[onnxruntime]"
    # newer Optimum releases moved the exporter to a separate package:
    pip install optimum-onnx        # only if `optimum.exporters.onnx` is missing

Usage:
    python model_setup.py                 # builds ./models
    python model_setup.py --out D:/models # custom output folder
    python model_setup.py --test          # build, then run a quick translation
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

# Folder name (inside models/) -> Hugging Face model id
MODELS = {
    "tl-en": "Helsinki-NLP/opus-mt-tl-en",
    "en-hil": "Helsinki-NLP/opus-mt-en-hil",
    "hil-en": "Helsinki-NLP/opus-mt-hil-en",
    "en-tl": "Helsinki-NLP/opus-mt-en-tl",
}

# Files the phone needs from the tokenizer/config (everything else is dropped)
TOKENIZER_FILES = ["source.spm", "target.spm", "vocab.json"]


def export_onnx(model_id: str, tmp_dir: Path) -> None:
    """Export encoder + decoder (no KV-cache variant, keeps it small & simple)."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Optimum ONNX exporter not found. Run:\n"
            '  pip install "optimum[onnxruntime]" optimum-onnx'
        ) from e

    # task "text2text-generation" (without "-with-past") produces exactly:
    #   encoder_model.onnx, decoder_model.onnx, config.json, tokenizer files
    main_export(
        model_name_or_path=model_id,
        output=str(tmp_dir),
        task="text2text-generation",
        opset=18,
        device="cpu",
    )


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic INT8 quantization: weights stored as int8, ~4x smaller file."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=False,
    )


def build_one(name: str, model_id: str, out_root: Path) -> None:
    out_dir = out_root / name
    if (out_dir / "meta.json").exists():  # written last = build finished
        print(f"[skip] {name}: already built")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"[1/3] {name}: downloading + exporting {model_id} to ONNX ...")
        export_onnx(model_id, tmp_dir)

        print(f"[2/3] {name}: quantizing to INT8 ...")
        quantize_int8(tmp_dir / "encoder_model.onnx", out_dir / "encoder.int8.onnx")
        quantize_int8(tmp_dir / "decoder_model.onnx", out_dir / "decoder.int8.onnx")

        print(f"[3/3] {name}: copying tokenizer + writing meta.json ...")
        for f in TOKENIZER_FILES:
            shutil.copy2(tmp_dir / f, out_dir / f)

        cfg = json.loads((tmp_dir / "config.json").read_text(encoding="utf-8"))
        vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
        meta = {
            "hf_model_id": model_id,
            "decoder_start_token_id": cfg.get("decoder_start_token_id", cfg["pad_token_id"]),
            "eos_token_id": cfg["eos_token_id"],
            "pad_token_id": cfg["pad_token_id"],
            "unk_token_id": vocab.get("<unk>", 1),
            "max_length": min(int(cfg.get("max_length") or 512), 512),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6
    print(f"[done] {name}: {size_mb:.1f} MB -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models", help="output folder (default: ./models)")
    ap.add_argument("--test", action="store_true", help="run a sample translation afterwards")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for name, model_id in MODELS.items():
        build_one(name, model_id, out_root)

    total = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file()) / 1e6
    print(f"\nAll models ready in {out_root}  (total {total:.0f} MB)")
    print("Copy this whole folder into your Android project: app/src/main/assets/models/")

    if args.test:
        from translator import Translator

        tr = Translator(out_root)
        for text, src, tgt in [
            ("Magandang umaga. Kumain ka na ba?", "tgl", "hil"),
            ("Maayong aga. Nagkaon ka na?", "hil", "tgl"),
        ]:
            print(f"\n[{src}->{tgt}] {text}\n  => {tr.translate(text, src, tgt)}")


if __name__ == "__main__":
    main()

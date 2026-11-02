"""
translator.py  --  offline Tagalog <-> Hiligaynon translation (runs on the phone).

Dependencies on the phone: only `sentencepiece` (no torch, no transformers).
Neural-network inference uses ONNX Runtime through one of two backends:

  * PyOrtBackend     -> Python `onnxruntime` package (PC testing, Termux)
  * ChaquopyBackend  -> ONNX Runtime for Android (Java/Kotlin library), called
                        from Python through Chaquopy's Java bridge. This is
                        needed because Chaquopy has no `onnxruntime` wheel.
                        Requires the small Kotlin class OrtHelper.kt.

The backend is picked automatically.

Usage:
    tr = Translator("/path/to/models")
    tr.tgl_to_hil("Salamat sa tulong mo.")
    tr.hil_to_tgl("Salamat sa imo bulig.")
    tr.translate(text, "tgl", "hil")      # generic form

Command line (PC):
    python translator.py models tgl hil "Magandang gabi sa inyong lahat."
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import sentencepiece as spm

# Path through English for each direction (OPUS-MT has no direct tgl<->hil model)
ROUTES = {
    ("tgl", "hil"): ["tl-en", "en-hil"],
    ("hil", "tgl"): ["hil-en", "en-tl"],
    ("tgl", "eng"): ["tl-en"],
    ("eng", "hil"): ["en-hil"],
    ("hil", "eng"): ["hil-en"],
    ("eng", "tgl"): ["en-tl"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Inference backends: encode(ids) once, then next_token(prefix) per step
# --------------------------------------------------------------------------- #
class PyOrtBackend:
    """Uses the Python onnxruntime package (desktop / Termux)."""

    def __init__(self, model_dir: Path, threads: int = 4):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(model_dir / "encoder.int8.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(model_dir / "decoder.int8.onnx"), so, providers=prov)
        self._hidden = None
        self._mask = None

    def encode(self, ids: list[int]) -> None:
        np = self.np
        input_ids = np.array([ids], dtype=np.int64)
        self._mask = np.ones_like(input_ids)
        self._hidden = self.enc.run(None, {"input_ids": input_ids, "attention_mask": self._mask})[0]

    def next_token(self, prefix: list[int], banned_id: int) -> int:
        np = self.np
        logits = self.dec.run(
            None,
            {
                "input_ids": np.array([prefix], dtype=np.int64),
                "encoder_attention_mask": self._mask,
                "encoder_hidden_states": self._hidden,
            },
        )[0][0, -1]  # logits of the last position: shape [vocab]
        logits[banned_id] = -np.inf
        return int(logits.argmax())

    def close(self) -> None:
        pass


class ChaquopyBackend:
    """Calls ONNX Runtime Android via the Kotlin class OrtHelper (see OrtHelper.kt).

    Only small int arrays cross the Python<->Java bridge; the encoder output and
    the 58k-wide logits stay on the Java side, so this is fast.
    """

    HELPER_CLASS = "com.example.tlhiltranslator.OrtHelper"  # change if your package differs

    def __init__(self, model_dir: Path, threads: int = 4):
        from java import jarray, jclass, jlong  # provided by Chaquopy

        self._jarray, self._jlong = jarray, jlong
        helper = jclass(self.HELPER_CLASS)
        self.h = helper(str(model_dir / "encoder.int8.onnx"), str(model_dir / "decoder.int8.onnx"), threads)

    def _arr(self, ids: list[int]):
        return self._jarray(self._jlong)(ids)

    def encode(self, ids: list[int]) -> None:
        self.h.encode(self._arr(ids))

    def next_token(self, prefix: list[int], banned_id: int) -> int:
        return int(self.h.nextToken(self._arr(prefix), banned_id))

    def close(self) -> None:
        self.h.close()


def _make_backend(model_dir: Path, threads: int):
    try:
        import java  # noqa: F401  -> we are inside a Chaquopy Android app
        return ChaquopyBackend(model_dir, threads)
    except ImportError:
        return PyOrtBackend(model_dir, threads)


# --------------------------------------------------------------------------- #
# One MarianMT model (one direction, e.g. tl -> en)
# --------------------------------------------------------------------------- #
class MarianOnnx:
    def __init__(self, model_dir: str | Path, threads: int = 4):
        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.id2tok = {i: t for t, i in self.vocab.items()}
        self.sp_src = spm.SentencePieceProcessor(model_file=str(d / "source.spm"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(d / "target.spm"))
        self.eos = self.meta["eos_token_id"]
        self.pad = self.meta["pad_token_id"]
        self.unk = self.meta["unk_token_id"]
        self.start = self.meta["decoder_start_token_id"]
        self.max_len = self.meta["max_length"]
        self.special = {self.eos, self.pad, self.unk}
        self.backend = _make_backend(d, threads)

    def _encode_text(self, text: str) -> list[int]:
        pieces = self.sp_src.encode(text, out_type=str)
        ids = [self.vocab.get(p, self.unk) for p in pieces]
        return ids[: self.max_len - 1] + [self.eos]

    def _decode_ids(self, ids: list[int]) -> str:
        pieces = [self.id2tok[i] for i in ids if i not in self.special and i in self.id2tok]
        return self.sp_tgt.decode_pieces(pieces).strip()

    def translate_sentence(self, text: str) -> str:
        src_ids = self._encode_text(text)
        self.backend.encode(src_ids)
        out = [self.start]
        limit = min(self.max_len, 2 * len(src_ids) + 16)
        for _ in range(limit):
            tok = self.backend.next_token(out, banned_id=self.pad)  # greedy search
            if tok == self.eos:
                break
            out.append(tok)
        return self._decode_ids(out[1:])

    def translate(self, text: str) -> str:
        out_lines = []
        for line in text.splitlines():  # keep the user's line breaks
            sents = [s for s in _SENT_SPLIT.split(line.strip()) if s]
            out_lines.append(" ".join(self.translate_sentence(s) for s in sents))
        return "\n".join(out_lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class Translator:
    """Loads models lazily, so only the ones you actually use take up RAM."""

    def __init__(self, models_root: str | Path, threads: int = 4):
        self.root = Path(models_root)
        self.threads = threads
        self._models: dict[str, MarianOnnx] = {}

    def _model(self, name: str) -> MarianOnnx:
        if name not in self._models:
            path = self.root / name
            if not (path / "meta.json").exists():
                raise FileNotFoundError(f"Model '{name}' not found in {self.root}. Run model_setup.py first.")
            self._models[name] = MarianOnnx(path, self.threads)
        return self._models[name]

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text or src == tgt:
            return text
        route = ROUTES.get((src, tgt))
        if route is None:
            raise ValueError(f"Unsupported direction {src}->{tgt}. Supported: {sorted(ROUTES)}")
        for name in route:
            text = self._model(name).translate(text)
        return text

    def tgl_to_hil(self, text: str) -> str:
        return self.translate(text, "tgl", "hil")

    def hil_to_tgl(self, text: str) -> str:
        return self.translate(text, "hil", "tgl")

    def close(self) -> None:
        for m in self._models.values():
            m.backend.close()
        self._models.clear()


# Convenience functions for calling from Kotlin through Chaquopy
_instance: Translator | None = None


def init(models_root: str, threads: int = 4) -> None:
    global _instance
    _instance = Translator(models_root, threads)


def translate(text: str, src: str, tgt: str) -> str:
    if _instance is None:
        raise RuntimeError("Call translator.init(models_root) first")
    return _instance.translate(text, src, tgt)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('usage: python translator.py <models_dir> <tgl|hil|eng> <tgl|hil|eng> "text"')
        sys.exit(1)
    print(Translator(sys.argv[1]).translate(" ".join(sys.argv[4:]), sys.argv[2], sys.argv[3]))

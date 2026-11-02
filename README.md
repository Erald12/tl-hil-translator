# Offline Tagalog ⇄ Hiligaynon translator (Android, vivo Y27)

## Why OPUS-MT instead of NLLB-200
NLLB-200 has no Hiligaynon (`hil_Latn` is not in its language list) and is non-commercial only (CC-BY-NC).
This project uses 4 Helsinki-NLP OPUS-MT models (CC-BY 4.0) and goes through English:
`tgl → eng → hil` and `hil → eng → tgl`. Going through English loses some meaning, and words the
models don't know (e.g. "kahapon") can come out untranslated.

**License:** CC-BY 4.0 requires attribution. Put "Translation models: Helsinki-NLP OPUS-MT (CC-BY 4.0)" in the app's About screen.

## 1. Build the models (PC)
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece onnx onnxruntime "optimum[onnxruntime]"
python model_setup.py --test
```
Output is `models/` (~544 MB): `tl-en/ en-hil/ hil-en/ en-tl/`, each with
`encoder.int8.onnx, decoder.int8.onnx, source.spm, target.spm, vocab.json, meta.json`.

Test on the PC: `python translator.py models tgl hil "Saan ang palengke?"`

## 2. Build the Android app
**Recommended: build in the cloud with GitHub Actions. You don't need Android Studio.** See `how_to_run.txt`.
`python make_upload_folder.py`, then upload `upload_to_github/` to a GitHub repository. Each push produces a
Release with `TLHil-full-buildN.apk` (phrase list + AI) and `TLHil-lite-buildN.apk` (phrase list only).

Project layout:
| Path | What |
|---|---|
| `android-app/` | Complete Gradle project (AGP 8.7.3, Kotlin 2.0.21, Chaquopy 17.2.0, Python 3.10) |
| `android-app/app/src/main/` | `MainActivity.kt`, layout, manifest (shared by both flavors) |
| `android-app/app/src/full/` | `OrtHelper.kt` + ONNX Runtime (full flavor only) |
| `prepare_android.py` | Copies `translator.py`, `database_builder.py`, `translations.db`, `models/` into the app before a build |
| `.github/workflows/build-apk.yml` | Cloud build: models → DB → copy → `gradle assembleFullDebug assembleLiteDebug` → Release |

Local alternative: `python prepare_android.py`, then open `android-app/` in Android Studio.

### How it fits together
- **Chaquopy** runs `translator.py` inside the app (tokenization with sentencepiece, plus the decoding loop).
- Chaquopy has **no `onnxruntime` wheel**, so the neural network runs in Microsoft's official
  **ONNX Runtime Android** library. `translator.py` detects Android and calls it through `OrtHelper.kt`.
- On first launch, `MainActivity` copies `assets/models` to internal storage (ONNX Runtime needs real
  file paths). After that it is 100 % offline. No INTERNET permission is needed.

### Size notes
- The APK is about 600 MB. That's fine for sideloading. The Google Play limit is 200 MB per base APK, so for Play,
  put the models in a **Play Asset Delivery** install-time asset pack.
- Peak RAM is about 450 MB per direction (only 2 of the 4 models are loaded).

## 3. Phrase dictionary + fuzzy search (works without the AI models)
- `phrase_data.py`: 1,302 Tagalog → Hiligaynon entries in 25 categories. It's plain text and easy to edit.
  It also exports `TAGALOG_TO_HILIGAYNON` (a Python dict) and `ENTRIES`.
  **The list was AI-generated.** Have a native speaker review it. Lines ending in `?` (16 entries) are the uncertain ones.
- `python database_builder.py` builds `translations.db`. Running it again only adds new rows; `--reset` rebuilds from scratch.
- Test a search: `python database_builder.py search "maayong gabe"`. Matching works in both directions, and typos are tolerated.

### On Android
1. `prepare_android.py` copies `database_builder.py` to `app/src/main/python/` and `translations.db` to `app/src/main/assets/`.
2. `MainActivity` copies the database to `filesDir` on first launch, the same way as the models.
3. Don't add rapidfuzz to Chaquopy's pip list. There's no Android build of it, so the code switches to `difflib` on its own and returns the same results.
4. Call it from Kotlin:
   ```kotlin
   val db = Python.getInstance().getModule("database_builder")
   val searcher = db.callAttr("PhraseSearcher", File(filesDir, "translations.db").absolutePath)
   val hit = searcher.callAttr("best", userText, "hil2tgl")   // null if no close match
   ```
5. Suggested app logic: try `searcher.best()` first, because it's instant and accurate for known phrases.
   If it returns nothing, fall back to the neural `translator.py`.
6. When a native speaker approves an entry, run `UPDATE translations SET verified = 1 WHERE id = ...`.

## 4. Open parallel corpus (`fetch_corpus.py`)
`python fetch_corpus.py --json --include-phrases` downloads about 165 MB once into `corpus_cache/`, then writes:
- `corpus_out/corpus.db`: about 30 MB. The `pairs` table plus an FTS5 full-text index called `pairs_fts`.
- `corpus_out/corpus.json`: about 12 MB, optional.
- `corpus_out/ATTRIBUTION.txt`: **required by the licenses**. Show it in the app's About screen.

| source | pairs | quality | license |
|---|---|---|---|
| eBible Hiligaynon + Tagalog Bibles, matched by verse | 30,004 | high (merged verses: medium) | CC BY-SA 4.0 |
| Wiktionary (kaikki.org), matched by English definition | 1,159 words | low (noisy) | CC BY-SA 4.0 |
| Tatoeba hil-tgl | 15 | high | CC BY 2.0 FR |
| OPUS (Ubuntu) | 18 UI strings | medium | per corpus |
| phrase_data.py (`--include-phrases`) | 1,266 | unverified | AI-generated |

On Android, copy `corpus.db` to `assets/` and then to `filesDir`, the same way as `translations.db`. Example queries:
```sql
SELECT tagalog, hiligaynon FROM pairs WHERE tgl_norm = 'tubig' AND kind != 'sentence';
SELECT p.tagalog, p.hiligaynon FROM pairs_fts f JOIN pairs p ON p.id = f.rowid
 WHERE pairs_fts MATCH 'tagalog:salamat' LIMIT 20;      -- example sentences containing a word
```
Most sentences are Bible verses, so the language is formal and religious. They're good as example sentences and as data for fine-tuning a model later, not as everyday conversation.

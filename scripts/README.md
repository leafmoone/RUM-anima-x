# Scripts

This directory contains maintained command-line entrypoints.

- `dev/anima_rum_xpred_train.py` - RUM reflow latent x-pred experiment CLI.
- `dev/import_chunked_cache.py` - import exported cache bucket tar files into `chunked_rum` `cache/chunk-XXXX` folders by sample index.
- `dev/import_anima_jlt_cache.py` - project-local import wrapper for prepared cache archives.

Current experiment wrapper:

```bash
python scripts/dev/import_anima_jlt_cache.py
```

Edit `SOURCE_DIRS` at the top of that script when new cache folders are ready.

Example:

```bash
python scripts/dev/import_chunked_cache.py \
  --src cache_data/095182-100212 \
  --chunk-root ./anima-jlt-xpred-turbo10-chunks \
  --start-index 69000 \
  --chunk-size 3000
```

Use `--dry-run` to scan target chunks without extracting. Existing sample files are skipped by default; add `--overwrite` only when you intentionally want to replace them.

Ordinary Anima LoRA/full-finetune training entrypoints are intentionally not included. This project only targets latent x-pred RUM conversion.

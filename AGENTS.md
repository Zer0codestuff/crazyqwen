# AGENTS.md

## Purpose

Local text-to-image app for Apple Silicon (target: the owner's MacBook, M4, 16 GB) running
`abenzerps/Qwen-Image-2.1-Uncensored-Quantized` (`qwen-image-2.1-UC-MLX-8bit.safetensors`) with a minimal web UI.

## Architecture

- `crazyqwen/models.py`: Hugging Face file locations; `resolve()` and `missing()` (offline check).
  Transformer from abenzerps; text encoder, VAE and tokenizer from `Qwen/Qwen-Image-2.1` (stock, unchanged);
  turbo LoRA `Viggle/Qwen-Image-2.1-viggle-turbo` v0.3 r256.
- `crazyqwen/engine.py`: inference on mflux 0.23 building blocks (`mflux.models.qwen21`).
  - `load_transformer`: loads the abenzerps MLX file as-is (affine 8-bit, group 64) into `Qwen21Transformer`.
    Key remaps: `modulation.0` -> `modulation.layers.1`, `time_text_embed.linear_N` -> `time_text_embed.timestep_embedder.linear_N`.
  - `StreamingTextEncoder`: bf16 Qwen3-VL text stack run layer by layer with lazy weights, each layer freed after use
    (~1.5 GB peak). Embeddings cached in `.cache/embeddings/`.
  - `TurboLoRA`: runtime (unbaked) LoRA wrappers in bf16; attached for Fast, detached for Quality.
  - `Engine.generate`: encode -> load -> denoise loop (copied from mflux `QwenImage21.generate_image`, plus progress,
    cancel, CFG) -> VAE decode (tiled above 1280x1280).
- `crazyqwen/server.py`: FastAPI on 127.0.0.1; one worker thread owns all MLX state; browser polls `/api/jobs/{id}`.
  Idle unload after `CRAZYQWEN_IDLE_UNLOAD` seconds (default 600). Outputs in `outputs/` (png + json).
- `crazyqwen/static/`: plain HTML/CSS/JS UI. Neutral grey canvas, amber accent from `assets/icon.svg`.
- `crazyqwen/cli.py`: `serve` (default), `download`, `generate`.

## Run

- `./run.sh` (uv sync, download if missing, serve on :7860).
- `uv run crazyqwen generate "prompt" --mode fast|quality`.
- No test suite; validate with one Fast generation at 1024x1024 and check timings/peak memory in the job metadata.

## Status

- Verified end to end on M4 16 GB (CLI and UI, Fast and Quality). 1024x1024: Fast ~43 s/step (~4 min 50 s total),
  Quality ~23 s/step (~15 min for 40 steps). Peaks: encode 1.3-2.3 GB, denoise 9.9 GB, decode 6.4 GB.
- Memory policy (`engine.py`): `LOW_MEMORY` (RAM < 32 GB) releases the transformer before the VAE decode;
  `TILED_DECODE` (RAM < 64 GB) decodes in 512 px tiles. Untiled decode with the transformer resident peaked at 23 GB.

## Recent changes

- 2026-10-20: initial implementation (engine, server, UI, CLI, run.sh, README guide).
- 2026-10-20: low-memory decode (release transformer, tiled VAE), per-phase peak memory in job metadata,
  UI stage no longer overflows the viewport, README speed and memory figures.

## Preferences and constraints

- Keep the text encoder in bf16 (mflux: quantizing it degrades prompt adherence); stream it instead.
- Do not re-quantize the transformer; use the 8-bit file the owner chose.
- Pillow pinned `<13` (13.0 fails to import on Python 3.13). Python pinned to 3.13.
- UI: minimal, image gets maximum area, no spinners, no extra copy.
- GitHub: work on branches, squash-merge PRs; Zer0codestuff is the only commit author, no agent trailers.

## Known issues / next steps

- The runtime turbo LoRA (r256, bf16, unbaked) roughly doubles step time (43 s vs 23 s at 1024). Worth profiling;
  baking into 8-bit weights needs a quality check first.
- Possible additions: image editing (model supports up to 10 reference images), 9-step turbo mode, live preview.

## Do not

- Do not load the full text encoder into memory at once on 16 GB machines.
- Do not use Gradio or replace the custom UI with a generic template.

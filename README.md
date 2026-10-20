<img src="assets/icon.svg" width="64" height="64" alt="">

# CrazyQwen

Local image generation on Apple Silicon with
[Qwen-Image 2.1 Uncensored](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized)
(MLX 8-bit), in a small web UI that opens in your browser. Everything runs on your Mac; prompts and images never
leave it.

> **Status:** early version. Model loading and the turbo LoRA are tested on an M4 with 16 GB. A full
> end-to-end generation has not been verified yet. If something fails, please open an issue with the terminal output.

## Requirements

- A Mac with Apple Silicon (M1 or newer). 16 GB of memory is the minimum; more is faster and leaves room for other apps.
- About 30 GB of free disk space for the model files.
- [uv](https://docs.astral.sh/uv/), the Python package manager: `brew install uv`
- An internet connection for the first run only.

## Install and start

```sh
git clone https://github.com/Zer0codestuff/crazyqwen.git
cd crazyqwen
./run.sh
```

The first run:

1. installs a private Python environment in `.venv` (nothing is installed system-wide),
2. downloads about 28 GB of model files into the Hugging Face cache (`~/.cache/huggingface`); this can take a while,
3. starts the app and opens http://127.0.0.1:7860 in your browser.

Later runs skip the download and open in a few seconds. Stop the app with `Ctrl+C` in the terminal.

## Using the app

1. Write a prompt on the right. Detailed descriptions work best: subject, setting, lighting, style, camera.
2. Pick a mode:
   - **Fast**: 6 steps with Viggle's turbo LoRA. About 5× faster than Quality, close in quality. Start here.
   - **Quality**: 40 steps with the uncensored model alone. Under *Quality settings* you can change the steps,
     and raise *Guidance* above 1 to use a negative prompt (this doubles the time per step).
3. Choose an aspect ratio and a size. Small is about 0.5 megapixels, Medium 1, Large 2. Larger images take
   proportionally longer.
4. Leave the seed empty for a random image, or set it to repeat a result. The same prompt, seed and settings give the
   same image.
5. Press **Generate** (or `⌘ Enter`). The bar at the top shows progress and the time left; **Cancel** stops after
   the current step.

Every image appears in the strip under the canvas. Click one (or use the arrow keys) to see its settings, then:

- **Reuse settings** copies its prompt, mode and seed into the form, so you can tweak and regenerate.
- **Download** saves the PNG.
- **Delete** removes it from disk.

The first image of a session takes longer: the model is loaded into memory (about 15 seconds) and a new prompt is
read by the text encoder (a few seconds). Afterwards only new prompts pay that cost.

## Where things are saved

- Images: `outputs/`, one `.png` per image with a `.json` of its settings next to it (also embedded in the PNG).
- Prompt cache: `.cache/embeddings/` (safe to delete).
- Model files: `~/.cache/huggingface/hub/`. To free the space, delete the folders `models--abenzerps--Qwen-Image-2.1-Uncensored-Quantized`,
  `models--Qwen--Qwen-Image-2.1` and `models--Viggle--Qwen-Image-2.1-viggle-turbo`.

## Terminal use

```sh
uv run crazyqwen generate "a lighthouse in a storm, oil painting" --mode fast --width 1024 --height 768
uv run crazyqwen generate "studio portrait, rim light" --mode quality --steps 40 --seed 42
uv run crazyqwen download             # fetch the model files without starting the UI
uv run crazyqwen serve --port 8000    # start the UI on another port
```

## How it fits in 16 GB

| Part | Memory | How |
| --- | --- | --- |
| Transformer (MLX 8-bit) | 7.6 GB | Loaded once from the abenzerps file, without re-quantizing. Unloaded after 10 idle minutes (`CRAZYQWEN_IDLE_UNLOAD=<seconds>` to change). |
| Text encoder (Qwen3-VL 8B, bf16) | ~1.5 GB peak | The full-precision weights (~16 GB) are streamed from disk one layer at a time instead of being loaded whole. Repeated prompts come from the cache. |
| Turbo LoRA | 1.4 GB | Attached only in Fast mode, removed in Quality mode. |
| VAE | ~1.4 GB | Large images are decoded in tiles. |

The text encoder stays at full precision on purpose: mflux notes that quantizing it noticeably hurts how well the
image follows the prompt.

## Troubleshooting

- **"Not ready: model files are missing or still downloading"**: run `./run.sh` again and let the download finish.
  An interrupted download resumes where it stopped.
- **Generation is very slow or the Mac swaps**: close memory-heavy apps (browsers with many tabs, editors), use Small
  or Medium size, and prefer Fast mode.
- **Port already in use**: `./run.sh --port 8000`.

## Credits and license

Built on [mflux](https://github.com/filipstrand/mflux) and [MLX](https://github.com/ml-explore/mlx). Model weights:
[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
[abenzerps/Qwen-Image-2.1-Uncensored-Quantized](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized),
[Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).

The weights are under the Qwen Research License (research and non-commercial use). The uncensored model has no
safety filter; you are responsible for what you generate.

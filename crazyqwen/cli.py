"""Command line entry points: serve (default), download, generate."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(prog="crazyqwen", description="Local Qwen-Image 2.1 on Apple Silicon")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the web UI (default)")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--no-browser", action="store_true")

    sub.add_parser("download", help="download every model file (~28 GB, once)")

    gen = sub.add_parser("generate", help="generate one image from the terminal")
    gen.add_argument("prompt")
    gen.add_argument("--mode", choices=["fast", "quality"], default="fast")
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--steps", type=int)
    gen.add_argument("--guidance", type=float, default=1.0)
    gen.add_argument("--negative-prompt", default="")

    args = parser.parse_args()
    command = args.command or "serve"

    if command == "download":
        from crazyqwen import models

        paths = models.resolve(download=True)
        print(f"Transformer: {paths.transformer}\nBase files:  {paths.base_root}\nTurbo LoRA:  {paths.turbo_lora}")
        return

    if command == "generate":
        from crazyqwen.engine import Engine, GenerationRequest
        from crazyqwen.server import save_image

        request = GenerationRequest(
            prompt=args.prompt,
            mode=args.mode,
            width=args.width,
            height=args.height,
            seed=args.seed,
            steps=args.steps,
            guidance=args.guidance,
            negative_prompt=args.negative_prompt,
        )

        def show(p):
            print(f"{p.phase} {p.step}/{p.total} {p.extra or ''}".strip(), flush=True)

        image, meta = Engine().generate(request, show)
        saved = save_image(image, meta)
        print(json.dumps(saved, indent=2))
        return

    import uvicorn

    from crazyqwen.server import create_app

    port = getattr(args, "port", 7860)
    url = f"http://127.0.0.1:{port}"
    if not getattr(args, "no_browser", False):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"CrazyQwen running at {url}")
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

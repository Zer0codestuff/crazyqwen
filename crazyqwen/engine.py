"""Qwen-Image 2.1 (uncensored, MLX 8-bit) inference tuned for 16 GB Apple Silicon.

Memory plan:
- The bf16 text encoder (~16 GB) is never fully resident. Its weights are mapped lazily and
  each decoder layer is evaluated, then dropped, so encoding peaks at roughly 1.5 GB.
  Prompt embeddings are cached on disk, so a repeated prompt skips the encoder entirely.
- The 8-bit transformer (~7.6 GB) is loaded from the abenzerps file as-is (no re-quantization).
  Below 32 GB of RAM it is released before the VAE decode (see LOW_MEMORY); otherwise it stays
  resident between images until the idle timeout unloads it.
- The turbo LoRA is attached at runtime only for Fast mode and detached for Quality mode.
"""

from __future__ import annotations

import gc
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

import mflux.models.qwen21.model.qwen21_scheduler  # noqa: F401  registers viggle_turbo
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.utils.image_util import ImageUtil
from PIL import Image

from crazyqwen import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_CACHE = PROJECT_ROOT / ".cache" / "embeddings"

FAST_STEPS = 6  # the turbo LoRA is distilled for exactly these sigma nodes
QUALITY_STEPS = 40  # upstream default

RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
# Measured on an M4 16 GB: transformer + LoRA hold 9.9 GB and an untiled 1024px VAE decode
# peaks at 16.4 GB on top of that. Below 32 GB, free the transformer before decoding and
# decode in 512px tiles (5.3 GB, differences invisible: mean 0.5/255). Reloading it costs
# a few seconds per image.
LOW_MEMORY = RAM_GB < 32
TILED_DECODE = RAM_GB < 64


class Cancelled(Exception):
    pass


@dataclass
class GenerationRequest:
    prompt: str
    mode: str = "fast"  # "fast" (turbo LoRA, 6 steps) or "quality" (40 steps)
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    steps: int | None = None  # quality mode only
    guidance: float = 1.0  # quality mode only; >1 with a negative prompt enables CFG
    negative_prompt: str = ""

    def normalized(self) -> "GenerationRequest":
        if self.mode not in ("fast", "quality"):
            raise ValueError("mode must be 'fast' or 'quality'")
        if not self.prompt.strip():
            raise ValueError("prompt is empty")
        width = max(256, min(2048, 16 * round(self.width / 16)))
        height = max(256, min(2048, 16 * round(self.height / 16)))
        seed = self.seed if self.seed is not None and self.seed >= 0 else int(time.time_ns() % 2**31)
        if self.mode == "fast":
            steps, guidance, negative = FAST_STEPS, 1.0, ""
        else:
            steps = max(4, min(80, self.steps or QUALITY_STEPS))
            guidance = max(1.0, min(10.0, float(self.guidance)))
            negative = self.negative_prompt.strip() if guidance > 1.0 else ""
        return GenerationRequest(self.prompt.strip(), self.mode, width, height, seed, steps, guidance, negative)


@dataclass
class Progress:
    phase: str  # loading | encoding | denoising | decoding
    step: int = 0
    total: int = 0
    extra: dict = field(default_factory=dict)


ProgressFn = Callable[[Progress], None]


def _peak_gb() -> float:
    return round(mx.get_peak_memory() / 1e9, 2)


def _free() -> None:
    gc.collect()
    mx.clear_cache()


class StreamingTextEncoder:
    """Runs the bf16 Qwen3-VL text stack one layer at a time (exact bf16 math, ~1.5 GB peak)."""

    def __init__(self, base_root: Path):
        self.base_root = base_root
        self.tokenizer = TokenizerLoader.load_all(
            definitions=Qwen21WeightDefinition.get_tokenizers(), model_path=str(base_root)
        )["qwen21"]
        self._drop = Qwen21PromptEncoder._system_prefix_length(self.tokenizer)

    def _component(self):
        return next(c for c in Qwen21WeightDefinition.get_components() if c.name == "text_encoder")

    def _build(self) -> Qwen21TextEncoder:
        # Weights stay lazy: nothing is read from disk until a layer is evaluated.
        module = Qwen21TextEncoder()
        component = self._component()
        weights = WeightLoader.load_single_local(component, self.base_root)
        supplied = dict(tree_flatten(weights.components[component.name]))
        supplied = Qwen21Initializer._normalize_text_encoder_weights(supplied, module)
        weights.components[component.name] = tree_unflatten(list(supplied.items()))
        WeightApplier.apply_and_quantize(
            weights=weights,
            models={component.name: module},
            quantize_arg=None,
            weight_definition=Qwen21WeightDefinition,
        )
        return module

    @staticmethod
    def _cache_path(prompt: str) -> Path:
        digest = hashlib.sha256(("qwen-image-2.1-te-v1\0" + prompt).encode()).hexdigest()[:32]
        return EMBED_CACHE / f"{digest}.safetensors"

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        prompt = prompt if prompt.strip() else " "
        path = self._cache_path(prompt)
        if path.exists():
            cached = mx.load(str(path))
            return cached["embeds"], cached["mask"]

        tokens = self.tokenizer.tokenize(prompt)
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask
        te = self._build()

        batch, seq_len = input_ids.shape
        hidden = te.embed_tokens(input_ids)
        mx.eval(hidden)
        te.embed_tokens = None
        _free()

        # Same masks and positions as Qwen21TextEncoder.__call__ (single, unpadded prompt).
        padding = mx.where(attention_mask == 1, 0.0, -float("inf")).astype(mx.float32)[:, None, None, :]
        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(idx[None, :] > idx[:, None], -float("inf"), 0.0).astype(mx.float32)
        mask4d = mx.broadcast_to(causal[None, None], (batch, 1, seq_len, seq_len)) + padding
        positions = mx.broadcast_to(idx[None, :], (batch, seq_len))
        rope = te.rotary_emb(hidden, positions)

        for i in range(len(te.layers)):
            hidden, _ = te.layers[i](hidden, attention_mask=mask4d, position_embeddings=rope)
            mx.eval(hidden)
            te.layers[i] = nn.Module()  # drop this layer's weights before reading the next
            _free()

        hidden = te.norm(hidden)
        embeds = hidden[:, self._drop :, :]
        mask = attention_mask[:, self._drop :]
        mx.eval(embeds, mask)
        del te
        _free()

        EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), {"embeds": embeds, "mask": mask})
        return embeds, mask


def _remap_transformer_key(key: str) -> str:
    # The abenzerps file uses stable-diffusion.cpp names; map the two that differ from mflux.
    # Shared modulation: "modulation.0" there, Sequential(SiLU, Linear) -> "modulation.layers.1" here.
    if key.startswith("modulation.0."):
        return "modulation.layers.1." + key.removeprefix("modulation.0.")
    # Timestep MLP: "time_text_embed.linear_N" -> "time_text_embed.timestep_embedder.linear_N".
    if key.startswith("time_text_embed.linear_"):
        return "time_text_embed.timestep_embedder." + key.removeprefix("time_text_embed.")
    return key


def load_transformer(path: Path) -> Qwen21Transformer:
    weights = mx.load(str(path))  # lazy, memory-mapped reads on first use
    weights = {_remap_transformer_key(k): v for k, v in weights.items()}
    transformer = Qwen21Transformer()
    quantized = {k.removesuffix(".scales") for k in weights if k.endswith(".scales")}
    # The file is affine 8-bit, group size 64 (scales shape = in_features / 64).
    nn.quantize(transformer, group_size=64, bits=8, class_predicate=lambda p, m: p in quantized)
    expected = dict(tree_flatten(transformer.parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    shapes = [k for k in expected.keys() & weights.keys() if expected[k].shape != weights[k].shape]
    if missing or unexpected or shapes:
        raise ValueError(
            f"transformer checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}, shapes={shapes[:8]}"
        )
    transformer.load_weights(list(weights.items()), strict=True)
    mx.eval(transformer.parameters())
    return transformer


def load_vae(base_root: Path) -> Qwen21VAE:
    vae = Qwen21VAE()
    component = next(c for c in Qwen21WeightDefinition.get_components() if c.name == "vae")
    weights = WeightLoader.load_single_local(component, base_root)
    supplied = dict(tree_flatten(weights.components["vae"]))
    supplied = Qwen21Initializer._normalize_vae_weights(supplied, vae)
    weights.components["vae"] = tree_unflatten(list(supplied.items()))
    WeightApplier.apply_and_quantize(
        weights=weights, models={"vae": vae}, quantize_arg=None, weight_definition=Qwen21WeightDefinition
    )
    return vae


def _set_child(root: nn.Module, path: str, value) -> None:
    *parents, last = path.split(".")
    node = root
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


class TurboLoRA:
    """Attaches/detaches the Viggle turbo LoRA as runtime wrappers around the 8-bit layers."""

    def __init__(self, path: Path):
        self.path = path
        self.wrappers: list[tuple[str, LoRALinear]] = []

    @property
    def attached(self) -> bool:
        return bool(self.wrappers)

    def attach(self, transformer: Qwen21Transformer) -> None:
        if self.attached:
            return
        LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[str(self.path)],
            lora_scales=[1.0],
            bake_lora=False,  # baking into 8-bit weights would lose most of the low-rank update
        )
        for name, module in transformer.named_modules():
            if isinstance(module, LoRALinear):
                # Keep the factors in bf16: float32 masters would double the LoRA's memory.
                module.lora_A = module.lora_A.astype(mx.bfloat16)
                module.lora_B = module.lora_B.astype(mx.bfloat16)
                self.wrappers.append((name, module))
        if not self.wrappers:
            raise RuntimeError("turbo LoRA matched no transformer layers")
        mx.eval([(w.lora_A, w.lora_B) for _, w in self.wrappers])

    def detach(self, transformer: Qwen21Transformer) -> None:
        for name, wrapper in self.wrappers:
            _set_child(transformer, name, wrapper.linear)
        self.wrappers = []
        _free()


class Engine:
    """Single-threaded owner of all MLX state. Call every method from the same thread."""

    def __init__(self):
        self.paths: models.ModelPaths | None = None
        self.text_encoder: StreamingTextEncoder | None = None
        self.transformer: Qwen21Transformer | None = None
        self.vae: Qwen21VAE | None = None
        self.turbo: TurboLoRA | None = None
        self.last_used = 0.0
        self.lock = threading.Lock()
        # Keep MLX's buffer cache small so freed memory goes back to macOS quickly.
        mx.set_cache_limit(2 * 1024**3)

    @property
    def loaded(self) -> bool:
        return self.transformer is not None or self.vae is not None

    def _ensure_paths(self) -> models.ModelPaths:
        if self.paths is None:
            self.paths = models.resolve(download=False)
        return self.paths

    def load(self, progress: ProgressFn | None = None) -> None:
        paths = self._ensure_paths()
        if self.text_encoder is None:
            self.text_encoder = StreamingTextEncoder(paths.base_root)
        if self.transformer is None:
            if progress:
                progress(Progress("loading"))
            self.transformer = load_transformer(paths.transformer)
            self.turbo = TurboLoRA(paths.turbo_lora)
        if self.vae is None:
            self.vae = load_vae(paths.base_root)
        self.last_used = time.time()

    def unload_transformer(self) -> None:
        self.transformer = None
        self.turbo = None
        _free()

    def unload(self) -> None:
        self.vae = None
        self.unload_transformer()

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> tuple[Image.Image, dict]:
        req = request.normalized()
        report = progress or (lambda p: None)
        started = time.perf_counter()
        peaks: dict[str, float] = {}
        mx.reset_peak_memory()

        # 1. Encode the prompt first, while the least memory is in use.
        paths = self._ensure_paths()
        if self.text_encoder is None:
            self.text_encoder = StreamingTextEncoder(paths.base_root)
        report(Progress("encoding"))
        embeds, mask = self.text_encoder.encode(req.prompt)
        neg = self.text_encoder.encode(req.negative_prompt) if req.negative_prompt else None
        encoded_at = time.perf_counter()
        peaks["encode"] = _peak_gb()
        mx.reset_peak_memory()
        if cancelled():
            raise Cancelled()

        # 2. Transformer (+ turbo LoRA for Fast mode).
        self.load(report)
        transformer, turbo = self.transformer, self.turbo
        if req.mode == "fast":
            turbo.attach(transformer)
        else:
            turbo.detach(transformer)
        loaded_at = time.perf_counter()

        config = Config(
            model_config=ModelConfig.qwen_image_21(),
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
            guidance=req.guidance,
            scheduler="viggle_turbo" if req.mode == "fast" else "linear",
        )
        scheduler = config.scheduler
        latents = Qwen21LatentCreator.create_noise(req.seed, config.height, config.width)

        report(Progress("denoising", 0, req.steps))
        step_times = []
        with transformer.inference_projections():
            for t in range(req.steps):
                if cancelled():
                    raise Cancelled()
                t0 = time.perf_counter()
                noise = transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=embeds,
                    encoder_hidden_states_mask=mask,
                )
                if neg is not None:
                    noise_neg = transformer(
                        t=t,
                        config=config,
                        hidden_states=latents,
                        encoder_hidden_states=neg[0],
                        encoder_hidden_states_mask=neg[1],
                    )
                    noise = noise_neg + req.guidance * (noise - noise_neg)
                latents = scheduler.step(noise=noise, timestep=t, latents=latents)
                mx.eval(latents)
                step_times.append(time.perf_counter() - t0)
                report(Progress("denoising", t + 1, req.steps, {"step_seconds": round(step_times[-1], 2)}))
        denoised_at = time.perf_counter()
        peaks["denoise"] = _peak_gb()

        # 3. Decode. On small-memory Macs the transformer is released first.
        report(Progress("decoding"))
        if LOW_MEMORY:
            del transformer, turbo
            self.unload_transformer()
        mx.reset_peak_memory()
        latents = Qwen21LatentCreator.unpack_latents(latents, config.height, config.width)
        tiling = TilingConfig() if TILED_DECODE else None
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=tiling)
        image = ImageUtil.to_pil(decoded)
        del decoded, latents
        peaks["decode"] = _peak_gb()
        _free()
        self.last_used = time.time()

        finished = time.perf_counter()
        meta = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "mode": req.mode,
            "width": config.width,
            "height": config.height,
            "seed": req.seed,
            "steps": req.steps,
            "guidance": req.guidance,
            "model": f"{models.TRANSFORMER_REPO}/{models.TRANSFORMER_FILE}",
            "lora": models.TURBO_FILE if req.mode == "fast" else None,
            "timings": {
                "encode": round(encoded_at - started, 2),
                "load": round(loaded_at - encoded_at, 2),
                "denoise": round(denoised_at - loaded_at, 2),
                "decode": round(finished - denoised_at, 2),
                "total": round(finished - started, 2),
                "per_step": round(sum(step_times) / max(1, len(step_times)), 2),
            },
            "peak_memory_gb": peaks,
        }
        return image, meta

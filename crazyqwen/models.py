"""Locations of the model files and the one-time download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

TRANSFORMER_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"
TRANSFORMER_FILE = "qwen-image-2.1-UC-MLX-8bit.safetensors"

# Text encoder (stock Qwen3-VL-8B), VAE and tokenizer come from the official repo:
# the uncensored release reuses them unchanged.
BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_PATTERNS = ["processor/*", "vae/*", "text_encoder/*", "scheduler/*", "model_index.json"]

TURBO_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_FILE = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors"


@dataclass(frozen=True)
class ModelPaths:
    transformer: Path
    base_root: Path  # snapshot folder with processor/, vae/, text_encoder/
    turbo_lora: Path


def resolve(download: bool = False) -> ModelPaths:
    """Return local paths. With download=False this never touches the network."""
    offline = not download
    transformer = hf_hub_download(TRANSFORMER_REPO, TRANSFORMER_FILE, local_files_only=offline)
    base_root = snapshot_download(BASE_REPO, allow_patterns=BASE_PATTERNS, local_files_only=offline)
    turbo = hf_hub_download(TURBO_REPO, TURBO_FILE, local_files_only=offline)
    return ModelPaths(Path(transformer), Path(base_root), Path(turbo))


def missing() -> str | None:
    """None when every file is on disk, otherwise a short reason."""
    try:
        paths = resolve(download=False)
    except Exception as exc:  # noqa: BLE001 - any lookup failure means "not downloaded"
        return "model files are missing or still downloading"
    shards = list((paths.base_root / "text_encoder").glob("*.safetensors"))
    if len(shards) < 4:
        return "text encoder download is incomplete"
    return None

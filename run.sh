#!/bin/zsh
# Install dependencies, download the models once (~28 GB), then open the web UI.
set -e
cd "${0:A:h}"

if ! command -v uv >/dev/null; then
  echo "uv is required: https://docs.astral.sh/uv/ (brew install uv)"
  exit 1
fi

uv sync --quiet
if ! uv run python -c "from crazyqwen import models; import sys; sys.exit(models.missing() is not None)"; then
  echo "Downloading model files (about 28 GB, only the first time)..."
  HF_XET_HIGH_PERFORMANCE=1 uv run crazyqwen download
fi
exec uv run crazyqwen serve "$@"

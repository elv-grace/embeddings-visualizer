#!/usr/bin/env bash
# Build the isolated interpreter that face (`face_vectors`) queries run in.
#
# Why isolated
# ------------
# model-celeb-vector needs numpy<1.20, torch 1.9 and mxnet 1.9. The service
# itself runs SigLIP 2 and Qwen on torch 2.13 and numpy 2.x, and those cannot
# coexist -- installing one downgrades the other's numpy and breaks it, along
# with UMAP and scikit-learn. So the tagger's model gets its own interpreter and
# `src/celeb_embedder.py` talks to it over a pipe (`tools/celeb_worker.py`).
#
# Why this is cheap despite the old pins
# -------------------------------------
# The tagger sets `gpu: -1` and runs InsightFace on CPU on purpose, to match
# celeb/model.py. So the CPU wheels are enough: no CUDA 10.1, no cudatoolkit,
# and none of the ~10 GB the tagger's conda container would cost. The result is
# about 1.5 GB.
#
# Usage:  tools/setup_celeb_env.sh          (idempotent; re-run to repair)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CELEB_ENV_DIR:-$REPO/.celebenv}"
PY="$ENV_DIR/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (it provisions Python 3.8; this box ships only 3.10," >&2
  echo "and numpy<1.20 has no 3.10 wheels). See https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "==> provisioning Python 3.8"
uv python install 3.8

echo "==> creating $ENV_DIR"
uv venv --python 3.8 "$ENV_DIR"

# torch 1.9 CPU comes from PyTorch's own index; everything else from PyPI.
# Versions match model-celeb-vector/setup.py, except mxnet: the CPU build
# replaces mxnet-cu101, since the tagger keeps mxnet on CPU anyway.
echo "==> installing the tagger's stack (CPU)"
uv pip install --python "$PY" \
  "numpy<1.20" \
  "mxnet==1.9.1" \
  "torch==1.9.0" \
  "torchvision==0.10.0" \
  "facenet_pytorch==2.5.2" \
  "easydict==1.9" \
  "scikit_learn==1.0.2" \
  "scikit-image==0.17.2" \
  "pandas==1.3.5" \
  "networkx==2.6.3" \
  "opencv-python" \
  "dacite" \
  "loguru" \
  --extra-index-url https://download.pytorch.org/whl/cpu

# The tagger's source, if it is not already beside this repo.
#
# Fetched rather than vendored. Three pieces are needed and only two of them are
# packaged: common-ml and celeb_vector have setup.py, but `celeb` (which holds
# FaceModel, the InsightFace wrapper) has none -- model-celeb has no setup.py at
# its root, which is why its own Containerfile COPYs the directory. So a clone is
# what actually supplies all three.
#
# Copying those files into this repo would work and is deliberately not done: a
# query has to produce the same vector the tagger produced, and a copy that
# drifts does not fail, it returns wrong neighbours. A clone stays versioned and
# updatable.
#
# Both repos are private (git@github.com:eluv-io/...), so this needs an SSH key
# with access -- the same requirement the tagger's own Containerfile has.
SRC_DIR="$ENV_DIR/src"
clone_if_missing() {
  local name="$1" url="$2" probe="$3"
  if [ -e "$probe" ]; then
    echo "    $name: already present at $probe"
    return
  fi
  if [ -d "$SRC_DIR/$name" ]; then
    echo "    $name: already cloned into $SRC_DIR/$name"
    return
  fi
  echo "    $name: not found locally, cloning"
  mkdir -p "$SRC_DIR"
  if ! git clone --depth 1 "$url" "$SRC_DIR/$name" 2>&1 | sed 's/^/      /'; then
    echo "      could not clone $url -- it is private, so this needs an SSH key" >&2
    echo "      with access. Or place the checkout beside this repo and re-run." >&2
    return 1
  fi
}

echo "==> locating the tagger's source"
REPO_PARENT="$(dirname "$REPO")"
clone_if_missing "model-celeb" "git@github.com:eluv-io/model-celeb.git" \
  "$REPO_PARENT/model-celeb/model-celeb-vector/celeb_vector/model.py" || true
clone_if_missing "common-ml" "git@github.com:eluv-io/common-ml.git" \
  "$REPO_PARENT/common-ml/common_ml" || true

echo "==> verifying"
"$PY" - <<'PY'
import sys
print("  python", sys.version.split()[0])
import numpy, torch, mxnet
print("  numpy", numpy.__version__, "| torch", torch.__version__, "| mxnet", mxnet.__version__)
PY

cat <<EOF

Done. The service finds this automatically at $ENV_DIR;
CELEB_PYTHON overrides the interpreter it uses.

Two things are NOT installed here and must already exist:
  * model-celeb-vector and common-ml checkouts (CELEB_EMBEDDING_PATH /
    COMMON_ML_PATH override where they are looked for)
  * the InsightFace r100 weights. Note config.yml names
    /ml/models/celeb_detection, but the checkpoint actually lives at
    /ml/models/celeb (models/model-r100-ii/model-{symbol.json,0000.params}).
    CELEB_MODEL_PATH overrides it.
EOF

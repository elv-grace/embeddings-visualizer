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

# The tagger's own source, installed rather than copied.
#
# One install brings both packages it needs: since
# eluv-io/model-celeb@110e446 the vector subdirectory's setup.py declares
# `packages=['celeb_vector', 'celeb']` with a package_dir mapping `celeb` to
# ../celeb, so `celeb.face_model` (the InsightFace wrapper) arrives with it.
# Both repos are readable over https, so no SSH key is needed.
#
# Cloned to a temp directory first, rather than `pip install git+https://...`,
# because model-celeb carries a `buildscripts` submodule pointing at the private
# qluvio/buildscripts. pip and uv both recurse submodules, so the direct form
# blocks on a credential prompt for a repo that is not needed to build this
# package. --no-recurse-submodules avoids it; nothing is left behind afterwards.
#
# --no-deps is deliberate. install_requires pins `mxnet-cu101==1.9.1`, a CUDA
# 10.1 build that is a very large download and buys nothing here: the tagger
# sets `gpu: -1` and runs InsightFace on CPU on purpose, so the plain `mxnet`
# installed above is what it actually uses.
#
# Copying these files into this repo would work and is deliberately not done: a
# query has to produce the same vector the tagger produced, and a copy that
# drifts does not fail, it returns wrong neighbours.
export GIT_TERMINAL_PROMPT=0    # fail fast instead of blocking on a prompt

SRC="$(mktemp -d)"
trap 'rm -rf "$SRC"' EXIT

install_from_git() {
  local name="$1" url="$2" ref="$3" subdir="${4:-}"
  echo "==> installing $name"
  git clone --depth 1 --branch "$ref" --no-recurse-submodules -q "$url" "$SRC/$name"
  uv pip install --python "$PY" --no-deps "$SRC/$name${subdir:+/$subdir}"
}

install_from_git model-celeb https://github.com/eluv-io/model-celeb.git \
  vector-faces model-celeb-vector
install_from_git common-ml https://github.com/eluv-io/common-ml.git vector-tags

echo "==> verifying"
"$PY" - <<'PYCHECK'
import sys
print("  python", sys.version.split()[0])
import numpy, torch, mxnet
print("  numpy", numpy.__version__, "| torch", torch.__version__, "| mxnet", mxnet.__version__)
from celeb_vector.model import CelebVectorizer   # the import a query actually makes
print("  celeb_vector + celeb + common_ml import OK")
PYCHECK

cat <<EOF

Done. The service finds this automatically at $ENV_DIR;
CELEB_PYTHON overrides the interpreter it uses.

One thing is NOT installed here and must already exist: the InsightFace r100
weights. Note config.yml names /ml/models/celeb_detection, but the checkpoint
actually lives at /ml/models/celeb
(models/model-r100-ii/model-{symbol.json,0000.params}). CELEB_MODEL_PATH
overrides where they are looked for.
EOF

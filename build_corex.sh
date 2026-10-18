#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COREX_HOME="${COREX_HOME:-/usr/local/corex}"
SPCONV_SOURCE="${SPCONV_SOURCE:-}"
CUMM_SOURCE="${CUMM_SOURCE:-}"
DIST_DIR="${1:-${REPO_DIR}/dist_corex}"

# Use installed dependencies by default. Source overrides must be explicit.
if [[ -n "${SPCONV_SOURCE}" && ! -f "${SPCONV_SOURCE}/spconv/__init__.py" ]]; then
    echo "spconv source not found: ${SPCONV_SOURCE}" >&2
    exit 1
fi
if [[ -n "${CUMM_SOURCE}" && ! -f "${CUMM_SOURCE}/cumm/__init__.py" ]]; then
    echo "cumm source not found: ${CUMM_SOURCE}" >&2
    exit 1
fi
[[ -x "${COREX_HOME}/bin/clang++" ]] || {
    echo "CoreX compiler not found: ${COREX_HOME}/bin/clang++" >&2
    exit 1
}

export COREX_HOME
export PATH="${COREX_HOME}/bin:${PATH}"
for package_source in "${CUMM_SOURCE}" "${SPCONV_SOURCE}"; do
    if [[ -n "${package_source}" ]]; then
        export PYTHONPATH="${package_source}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
done
export MAX_JOBS="${MAX_JOBS:-8}"
if [[ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    CMAKE_CUDA_ARCHITECTURES="$("${PYTHON_BIN}" -c 'import torch; print(torch.corex_arch)')"
fi
export CMAKE_CUDA_ARCHITECTURES
export SPCONV_DISABLE_JIT="${SPCONV_DISABLE_JIT:-1}"

"${PYTHON_BIN}" - <<'PY'
import os
import torch
import spconv.pytorch
from cumm.constants import COREX_ARCH
from spconv.__version__ import __version__

if not getattr(torch, "corex", False):
    raise SystemExit("A CoreX PyTorch build is required")
torch_arch = getattr(torch, "corex_arch", None)
requested = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
if requested not in {"ivcore11", "ivcore20", "ivcore30"}:
    raise SystemExit(f"Unsupported Corex target: {requested}")
if requested != torch_arch:
    raise SystemExit(f"Corex target {requested} does not match PyTorch target {torch_arch}")
if torch_arch != COREX_ARCH:
    raise SystemExit(
        f"cumm target {COREX_ARCH} does not match PyTorch target {torch_arch}"
    )
print(f"CoreX PyTorch {torch.__version__}, target {torch_arch}")
print(f"spconv {__version__}, target {COREX_ARCH}")
PY

mkdir -p "${DIST_DIR}"
DIST_DIR="$(cd "${DIST_DIR}" && pwd)"
cd "${REPO_DIR}"
"${PYTHON_BIN}" setup.py bdist_wheel --dist-dir "${DIST_DIR}"

echo "Built CoreX wheel:"
ls -lh "${DIST_DIR}"/pcdet-*.whl
echo "Install the matching CoreX spconv separately before using pcdet."

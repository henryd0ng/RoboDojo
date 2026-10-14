#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[tactile] installation failed at line ${LINENO}; see output above" >&2' ERR
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TACEX_ROOT="${TACEX_ROOT:-${ROOT_DIR}/third_party/RoboDojoTacEx}"
MODE="install"
case "${1:-}" in
  --prepare-only) MODE="prepare" ;;
  --check) MODE="check" ;;
  -h|--help)
    cat <<'EOF'
Usage: bash scripts/install_tactile.sh [--prepare-only | --check]
Use an existing Linux x86_64 / Python 3.11 environment with Isaac Sim 5.1,
Isaac Lab 2.3 and PyTorch 2.7/CUDA 12.8. No environment is activated for you.
--prepare-only fetches pinned sources and patches, without pip or compilation.
--check verifies source identity, packages and ELF resolution without a GPU run.
An existing unmanaged TACEX_ROOT is never overwritten.

Options via environment:
  PYTHON_BIN              target Python (default: python)
  TACEX_ROOT              new source checkout (default: third_party/RoboDojoTacEx)
  ROBODOJO_CUDA_HOME      CUDA 12.8 toolkit (default: CUDA_HOME or Python prefix)
  ROBODOJO_CUDA_ARCH      required target architecture, e.g. 89 or 120
  ROBODOJO_BUILD_JOBS     CMake parallelism (default: 4)
  ROBODOJO_VCPKG_ROOT     pinned vcpkg checkout (default: .cache/tactile/vcpkg)
  CC, CXX                optional compatible host compiler paths

See scripts/internal/tactile_source.json for source revisions and license notes.
Downloaded source/build directories are local dependencies, not PR contents.
EOF
    exit 0 ;;
  "") ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "Only one mode argument is supported" >&2; exit 2; fi

bootstrap_args=(--root "${TACEX_ROOT}")
if [[ "${MODE}" == "check" ]]; then bootstrap_args+=(--check); fi
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/internal/tactile_bootstrap.py" "${bootstrap_args[@]}"
if [[ "${MODE}" == "prepare" ]]; then exit 0; fi
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/internal/tactile_install_check.py" --base
PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
CUDA_ROOT="${ROBODOJO_CUDA_HOME:-${CUDA_HOME:-${PYTHON_PREFIX}}}"
if [[ "${MODE}" == "check" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/internal/tactile_install_check.py" --root "${TACEX_ROOT}" --cuda-root "${CUDA_ROOT}"
  exit 0
fi
for executable in git cmake readelf ldd; do
  command -v "${executable}" >/dev/null || { echo "[tactile] missing tool: ${executable}" >&2; exit 2; }
done
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "[tactile] set ROBODOJO_CUDA_HOME to a CUDA 12.8 toolkit with bin/nvcc; a PyTorch runtime alone cannot compile libuipc" >&2; exit 2
fi
NVCC_VERSION="$("${CUDA_ROOT}/bin/nvcc" --version)"
if [[ "${NVCC_VERSION}" != *"release 12.8"* ]]; then echo "[tactile] CUDA toolkit must be 12.8: ${CUDA_ROOT}" >&2; exit 2; fi
CUDA_ARCH="${ROBODOJO_CUDA_ARCH:-}"
CUDA_ARCH="${CUDA_ARCH//./}"
if [[ ! "${CUDA_ARCH}" =~ ^[0-9]{2,3}$ ]]; then
  echo "[tactile] set ROBODOJO_CUDA_ARCH for your GPU (e.g. 89 or 120)" >&2; exit 2
fi
BUILD_JOBS="${ROBODOJO_BUILD_JOBS:-4}"
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then echo "[tactile] ROBODOJO_BUILD_JOBS must be positive" >&2; exit 2; fi

VCPKG_ROOT="${ROBODOJO_VCPKG_ROOT:-${ROOT_DIR}/.cache/tactile/vcpkg}"
VCPKG_REVISION="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["vcpkg_revision"])' "${ROOT_DIR}/scripts/internal/tactile_source.json")"
if [[ ! -e "${VCPKG_ROOT}" ]]; then
  mkdir -p "${VCPKG_ROOT}"
  git -C "${VCPKG_ROOT}" init -q
  git -C "${VCPKG_ROOT}" remote add origin https://github.com/microsoft/vcpkg.git
  git -C "${VCPKG_ROOT}" fetch --depth=1 origin "${VCPKG_REVISION}"
  git -C "${VCPKG_ROOT}" checkout --detach "${VCPKG_REVISION}"
fi
if [[ "$(git -C "${VCPKG_ROOT}" rev-parse HEAD)" != "${VCPKG_REVISION}" ]]; then
  echo "[tactile] vcpkg revision mismatch; choose a separate ROBODOJO_VCPKG_ROOT" >&2; exit 2
fi
if ! git -C "${VCPKG_ROOT}" diff --quiet HEAD --; then
  echo "[tactile] tracked vcpkg files are modified; choose a clean ROBODOJO_VCPKG_ROOT" >&2; exit 2
fi
if [[ ! -x "${VCPKG_ROOT}/vcpkg" ]]; then "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics; fi
# Constrain the base stack so optional dependencies cannot upgrade it.
constraints="$(mktemp)"
trap 'rm -f "${constraints}"' EXIT
"${PYTHON_BIN}" - > "${constraints}" <<'PY'
from importlib.metadata import version
for name in ("isaacsim", "isaaclab", "torch", "torchvision", "numpy"):
    print(f"{name}=={version(name)}")
PY
"${PYTHON_BIN}" -m pip install -c "${constraints}" -r "${ROOT_DIR}/scripts/requirements-tactile.txt"

TACEX_SOURCE="${TACEX_ROOT}/third_party/TacEx"
UIPC_ROOT="${TACEX_SOURCE}/source/tacex_uipc"
BUILD_ROOT="${UIPC_ROOT}/build"
export CUDA_HOME="${CUDA_ROOT}" CUDA_PATH="${CUDA_ROOT}" CUDACXX="${CUDA_ROOT}/bin/nvcc"
export CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
export VCPKG_OVERLAY_PORTS="${ROOT_DIR}/scripts/internal/tactile_vcpkg_overlays"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" VCPKG_MAX_CONCURRENCY="${BUILD_JOBS}"
# The patched post-build hook produces artifacts without an implicit pip install.
cmake -S "${UIPC_ROOT}/libuipc" -B "${BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
  -DUIPC_PYTHON_EXECUTABLE_PATH="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')" \
  -DUIPC_BUILD_PYBIND=ON -DUIPC_BUILD_GUI=OFF -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_EXAMPLES=OFF -DUIPC_BUILD_BENCHMARKS=OFF \
  -DMUDA_BUILD_EXAMPLE=OFF -DMUDA_BUILD_TEST=OFF \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  "-DCMAKE_BUILD_RPATH=\$ORIGIN;${PYTHON_PREFIX}/lib;${CUDA_ROOT}/lib64;${CUDA_ROOT}/lib;${CUDA_ROOT}/targets/x86_64-linux/lib"
cmake --build "${BUILD_ROOT}" --parallel "${BUILD_JOBS}"
"${PYTHON_BIN}" -m pip install --no-deps --no-build-isolation -e "${BUILD_ROOT}/python"
ROBODOJO_TACEX_SKIP_BUILD=1 "${PYTHON_BIN}" -m pip install --no-deps --no-build-isolation \
  -e "${TACEX_SOURCE}/source/tacex" -e "${TACEX_SOURCE}/source/tacex_assets" -e "${UIPC_ROOT}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/internal/tactile_install_check.py" --root "${TACEX_ROOT}" --cuda-root "${CUDA_ROOT}"
echo "[tactile] source build installed; native contact/render validation is a separate GPU check."

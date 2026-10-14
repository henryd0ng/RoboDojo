# Optional tactile installation

Use the existing RoboDojo environment: Linux x86_64, Python 3.11, Isaac Sim 5.1,
Isaac Lab 2.3, PyTorch 2.7 and CUDA 12.8. The installer uses the selected Python;
it does not activate an environment or replace the Isaac stack.

## Existing local installation

```bash
conda activate robodojo-tactile
export PYTHON_BIN=python
export TACEX_ROOT="$PWD/third_party/RoboDojoTacEx-x5-pad-precision64-20261013"
bash scripts/install_tactile.sh --check
```

This directory is the current local installation. Its name does not select
floating-point precision; the accepted CAD preset uses float32 attachment targets.
Do not delete a source directory used by editable packages.

## Fresh installation

Choose a **new** source directory and the CUDA architecture for the target GPU:

```bash
export PYTHON_BIN=python
export TACEX_ROOT="$PWD/third_party/RoboDojoTacEx-local"
export ROBODOJO_CUDA_HOME=/path/to/cuda-12.8
export ROBODOJO_CUDA_ARCH=120
export ROBODOJO_BUILD_JOBS=2
bash scripts/install_tactile.sh --prepare-only
bash scripts/install_tactile.sh
bash scripts/install_tactile.sh --check
```

`--prepare-only` downloads pinned sources and GelSight Mini assets and applies
patches without compiling. Building also requires CUDA 12.8 `nvcc`, CMake >=3.26,
Git and a compatible C++ compiler. An unmanaged existing directory is rejected.
`CC`/`CXX` select host compilers; `ROBODOJO_VCPKG_ROOT` can select the pinned vcpkg checkout.

`--check` validates source identity, installed packages and ELF dependencies; it
does not run simulation. See [the demo](../tests/tactile/integration/README.md) for contact rendering.

[scripts/internal/tactile_source.json](../scripts/internal/tactile_source.json)
records revisions, patches and upstream license declarations. Keep original
license notices. The pinned libuipc root declares Apache-2.0 while its Python
package declares GPLv3; TacEx declares MIT in metadata/README but has no root
LICENSE at this revision. Source/build directories remain local dependencies.

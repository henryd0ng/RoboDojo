#!/usr/bin/env python3
"""Read-only package/ELF checks; does not launch Isaac or a GPU simulation."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse


def check_base():
    if sys.version_info[:2] != (3, 11) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("this build targets Linux x86_64 / Python 3.11")
    from packaging.version import Version

    for name, release in {"isaacsim": (5, 1), "isaaclab": (2, 3), "torch": (2, 7)}.items():
        found = version(name)
        if Version(found).release[:2] != release:
            raise RuntimeError(f"{name}=={found}; requires {release[0]}.{release[1]}.x")
    import torch

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"PyTorch CUDA runtime is {torch.version.cuda}; requires 12.8")
    # No CUDA availability/device query: base checks need no GPU context.
    print("[tactile] existing Isaac/PyTorch base is compatible")


def check_installed(root, cuda_root):
    root = root.resolve()
    for name, expected in {"tacex": "0.1.0", "tacex-assets": "0.1.0", "tacex-uipc": "0.1.0", "pyuipc": "0.9.0"}.items():
        dist = distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        url = urlparse(direct.get("url", ""))
        if (
            dist.version != expected
            or url.scheme != "file"
            or not Path(unquote(url.path)).resolve().is_relative_to(root)
        ):
            raise RuntimeError(f"{name} must be {expected} from {root}; found {dist.version}, {direct.get('url')}")
        print(f"[tactile] {name}=={dist.version}: {direct['url']}")
    from packaging.requirements import Requirement

    requirements = Path(__file__).resolve().parents[1] / "requirements-tactile.txt"
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        found = version(requirement.name)
        if requirement.specifier and found not in requirement.specifier:
            raise RuntimeError(f"{requirement.name}=={found}; requires {requirement.specifier}")
    if version("torch-scatter") != "2.1.2+pt27cu128":
        raise RuntimeError("torch-scatter requires the pt27cu128 build")
    spec = find_spec("uipc")
    if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
        raise RuntimeError("the imported uipc package is not from the selected source checkout")
    binary_root = Path(spec.origin).parent / "modules/Release/bin"
    binaries = list(binary_root.glob("*.so*"))
    if not list(binary_root.glob("pyuipc*.so")) or not (binary_root / "libuipc_backend_cuda.so").is_file():
        raise RuntimeError(f"missing compiled pyuipc/CUDA backend under {binary_root}")
    allowed = [
        root,
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        cuda_root.resolve(),
        Path("/usr/lib").resolve(),
        Path("/lib").resolve(),
        Path("/lib64").resolve(),
    ]
    for binary in binaries:
        dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
        for value in re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic):
            for entry in value.split(":"):
                if (
                    entry
                    and not entry.startswith("$ORIGIN")
                    and not any(Path(entry).resolve().is_relative_to(base) for base in allowed)
                ):
                    raise RuntimeError(f"foreign library search path in {binary.name}: {entry}")
        linked = subprocess.check_output(["ldd", str(binary)], text=True)
        if "not found" in linked:
            raise RuntimeError(f"unresolved native dependency in {binary.name}:\n{linked}")
        for entry in re.findall(r"(?:=>\s+|^\s*)(/\S+)", linked, re.MULTILINE):
            if not any(Path(entry).resolve().is_relative_to(base) for base in allowed):
                raise RuntimeError(f"foreign runtime dependency in {binary.name}: {entry}")
    print(
        f"[tactile] checked {len(binaries)} ELF files; dependencies resolve within "
        "the selected source/environment/toolkit or system libraries"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    args = parser.parse_args()
    if args.base:
        check_base()
    elif args.root and args.cuda_root:
        check_installed(args.root, args.cuda_root)
    else:
        parser.error("use --base or provide --root and --cuda-root")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, PackageNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[tactile] installation check failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

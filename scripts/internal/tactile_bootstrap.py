#!/usr/bin/env python3
"""Fetch the pinned public GelSight Mini sources without touching an existing checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "tactile_source.json"
MARKER = ".robodojo-tactile-source.json"


def run(*args, cwd=None, capture=False, input=None):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        input=input,
        text=True,
        check=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def identity():
    spec = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(MANIFEST.read_bytes())
    for patch in spec["patches"]:
        digest.update((HERE / patch).read_bytes())
    return spec, digest.hexdigest()


def source_diff(root, spec):
    return run("git", "diff", "--binary", "--full-index", "HEAD", "--", spec["subdirectory"], cwd=root, capture=True)


def verify(root, spec, fingerprint):
    marker = root / MARKER
    if not marker.is_file():
        raise RuntimeError(
            f"refusing to overwrite unmanaged directory {root}; choose an empty TACEX_ROOT. "
            "Existing local/prebuilt installations are left unchanged."
        )
    saved = json.loads(marker.read_text())
    if saved.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"source/patch version changed at {root}; prepare a new TACEX_ROOT instead of modifying it in place"
        )
    actual_root = Path(run("git", "rev-parse", "--show-toplevel", cwd=root, capture=True).strip()).resolve()
    revision = run("git", "rev-parse", "HEAD", cwd=root, capture=True).strip()
    if actual_root != root.resolve() or revision != spec["revision"]:
        raise RuntimeError(f"unexpected source repository or revision at {root}")
    if hashlib.sha256(source_diff(root, spec).encode()).hexdigest() != saved["diff_sha256"]:
        raise RuntimeError(
            f"tracked source files changed after preparation at {root}; refusing to build an unrecorded patch set"
        )
    source = root / spec["subdirectory"]
    required = [
        root / "LICENSE",
        source / "UNIVTAC_BASELINE.md",
        source / "source/tacex_uipc/libuipc/LICENSE",
        source / "source/tacex_uipc/libuipc/python/LICENSE",
        source / "source/tacex_uipc/libuipc/python/README.md",
        source / "source/tacex_uipc/libuipc/external/muda/CMakeLists.txt",
        source / "source/tacex_uipc/libuipc/scripts/SymEigen/README.md",
        source / "source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Case.usd",
        source / "source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/Gelpad_high_res.usd",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"incomplete source checkout: {path}")
        with path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/"):
                raise RuntimeError(f"unresolved Git LFS pointer: {path}")
    return source


def prepare(root, spec, fingerprint):
    if root.exists():
        return verify(root, spec, fingerprint)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=root.name + ".prepare-", dir=root.parent))
    try:
        run("git", "init", "-q", str(temporary))
        run("git", "remote", "add", "origin", spec["repository"], cwd=temporary)
        run("git", "config", "remote.origin.promisor", "true", cwd=temporary)
        run("git", "config", "remote.origin.partialclonefilter", "blob:none", cwd=temporary)
        run("git", "fetch", "--depth=1", "--filter=blob:none", "origin", spec["revision"], cwd=temporary)
        run("git", "sparse-checkout", "init", "--no-cone", cwd=temporary)
        data = f"/{spec['subdirectory']}/source/tacex_assets/tacex_assets/data"
        patterns = ["/LICENSE", "/README.md", f"/{spec['subdirectory']}/"]
        patterns += [
            f"!{data}/{part}/"
            for part in (
                "Policies",
                "Props",
                "Robots",
                "Sensors/GF225",
                "Sensors/XenseWS",
                "Sensors/GelSight_Mini/calibs/pix2pix",
            )
        ]
        run("git", "sparse-checkout", "set", "--no-cone", "--stdin", cwd=temporary, input="\n".join(patterns) + "\n")
        run("git", "checkout", "--detach", spec["revision"], cwd=temporary)
        for patch in spec["patches"]:
            command = ("git", "apply", "--directory=" + spec["subdirectory"])
            run(*command, "--check", str(HERE / patch), cwd=temporary)
            run(*command, str(HERE / patch), cwd=temporary)
        saved = dict(
            spec, fingerprint=fingerprint, diff_sha256=hashlib.sha256(source_diff(temporary, spec).encode()).hexdigest()
        )
        (temporary / MARKER).write_text(json.dumps(saved, indent=2) + "\n")
        verify(temporary, spec, fingerprint)
        temporary.rename(root)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return verify(root, spec, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="verify only; never fetch or modify sources")
    args = parser.parse_args()
    spec, fingerprint = identity()
    source = (verify if args.check else prepare)(args.root.resolve(), spec, fingerprint)
    print(f"[tactile] verified source {spec['repository']}@{spec['revision']}: {source}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[tactile] source preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

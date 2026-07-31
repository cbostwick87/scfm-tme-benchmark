"""Provenance capture: run manifest (P1), file hashes (P3), seeds (P4),
wall-clock timing (P6), and the DECISIONS log (P7).

Nothing here records host identity. The manifest deliberately captures the
GPU model, driver, CUDA and package versions -- which is what a third party
needs to reproduce -- and no instance name, address, or account identifier.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import time
from contextlib import contextmanager

_CHUNK = 1 << 20


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: str | pathlib.Path) -> str:
    """SHA256 of a file, chunked -- inputs here are multi-GiB (P3)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    """Versions for every package whose behaviour could change a number (P1)."""
    names = [
        "scanpy", "anndata", "numpy", "scipy", "pandas", "sklearn", "torch",
        "transformers", "scvi", "harmonypy", "celltypist", "statsmodels",
        "cellxgene_census", "matplotlib",
    ]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[n] = None
    return out


def gpu_witness() -> dict:
    """Confirm the GPU is actually usable, and assert the precision contract.

    The T4 is Turing: fp16 tensor cores work, bf16 does NOT. A silent bf16
    path would either fail or fall back to a slow emulation, so bf16
    availability is recorded explicitly rather than assumed.
    """
    info: dict = {"torch_available": False}
    try:
        import torch
    except Exception as exc:
        info["error"] = f"torch import failed: {exc}"
        return info
    info["torch_available"] = True
    info["torch_version"] = torch.__version__
    info["cuda_runtime"] = torch.version.cuda
    info["cuda_is_available"] = bool(torch.cuda.is_available())
    if not info["cuda_is_available"]:
        return info
    info["device_name"] = torch.cuda.get_device_name(0)
    info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    info["vram_MiB"] = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    # fp16 witness: a real matmul, not a flag check.
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    info["fp16_matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
    info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    del a
    torch.cuda.empty_cache()
    return info


def manifest(cfg: dict | None = None, extra: dict | None = None) -> dict:
    """Assemble the run manifest (P1, P2, P4)."""
    man = {
        "timestamp_utc": utcnow(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu_witness(),
        "packages": package_versions(),
    }
    if cfg is not None:
        man["seeds"] = cfg.get("run", {}).get("seeds")
        man["census_version"] = cfg.get("census", {}).get("version")
        man["config_name"] = cfg.get("run", {}).get("name")
    if extra:
        man.update(extra)
    return man


def write_manifest(man: dict, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    return path


@contextmanager
def timed(label: str, sink: list | None = None):
    """Wall-clock timing so compute-cost claims in the report are measured (P6)."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    rec = {"step": label, "seconds": round(dt, 3), "timestamp_utc": utcnow()}
    if sink is not None:
        sink.append(rec)
    print(f"[timing] {label}: {dt:.1f}s", flush=True)


def log_decision(
    decision: str, reason: str, consequence: str, phase: str,
    path: str | pathlib.Path = "DECISIONS.md",
) -> None:
    """Append to the DECISIONS log (P7). Called at the moment of the decision."""
    path = pathlib.Path(path)
    lines = path.read_text().rstrip("\n").split("\n") if path.exists() else []
    n = sum(1 for ln in lines if ln.startswith("| ") and ln.split("|")[1].strip().isdigit())
    row = f"| {n + 1} | {utcnow()} | {phase} | {decision} | {reason} | {consequence} |"
    with path.open("a") as fh:
        fh.write(row + "\n")

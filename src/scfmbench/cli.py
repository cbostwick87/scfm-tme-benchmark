"""Thin CLI. `manifest` is the only subcommand that runs outside a stage --
it captures the provenance record (P1) including the GPU witness."""
from __future__ import annotations

import argparse
import json
import pathlib

from scfmbench import config, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scfmbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="write the run manifest")
    m.add_argument("--config", default="configs/default.yaml")
    m.add_argument("--out", default="results/run_manifest.json")
    args = ap.parse_args(argv)

    if args.cmd == "manifest":
        cfg = config.load(args.config)
        man = provenance.manifest(cfg)
        p = provenance.write_manifest(man, args.out)
        gpu = man["gpu"]
        print(json.dumps({k: gpu.get(k) for k in
                          ("cuda_is_available", "device_name", "capability",
                           "vram_MiB", "fp16_matmul_ok", "bf16_supported")}, indent=2))
        print(f"wrote {p}")
        if gpu.get("cuda_is_available") and gpu.get("bf16_supported"):
            print("NOTE: bf16 reported as supported -- the config pins fp16; leave it pinned "
                  "unless the precision choice is re-validated.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

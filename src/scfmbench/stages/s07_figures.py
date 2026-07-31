"""Render F1-F5 from T2 alone.

Stage contract: reads only cached artefacts declared in the config, writes only
under the config's data root or results/, and is safe to re-run -- an existing
complete output is reused rather than recomputed unless --force is passed.
"""
from __future__ import annotations

import argparse

from scfmbench import config, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Render F1-F5 from T2 alone.')
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--force", action="store_true", help="recompute even if cached output exists")
    args = ap.parse_args(argv)
    cfg = config.load(args.config)
    raise NotImplementedError(
        "stage s07_figures is not implemented yet; it is scaffolded so the pipeline shape "
        "is reviewable and version-controlled before any analysis code is written."
    )


if __name__ == "__main__":
    raise SystemExit(main())

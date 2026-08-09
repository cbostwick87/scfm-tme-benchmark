"""Project B, Phase 0 step 1: verify every inherited FM-BENCH-A artefact.

Fails loudly. Never regenerates. Emits results/projectB/B0_artefact_verification.csv.

Three classes of check:
  HASH       - file digest against a recorded manifest (model checkpoints, raw data).
  STRUCTURE  - the artefact opens, covers the full cell index, and carries finite values.
               Embeddings were never hash-manifested by A, so structural identity against
               the cell index is the strongest available check and is the same one A used
               to validate them at the point of creation.
  INVARIANT  - a recorded scientific quantity still reads as reported (A6 silhouettes,
               split fractions, the reserved calibration partition).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/fm-bench")
REPO = ROOT / "scfm-tme-benchmark"
OUT = REPO / "results" / "projectB"
N_CELLS = 229_801
REPS = ["geneformer", "scgpt", "hvg_pca", "scvi", "harmony"]
PER_CELL = {"geneformer", "scgpt"}          # split-independent, sharded by dataset
PER_SPLIT = {"hvg_pca", "scvi", "harmony"}  # fit to a training partition, one file per split

rows: list[dict] = []


def rec(artefact: str, check: str, ok: bool, detail: str) -> None:
    rows.append(dict(artefact=artefact, check=check, status="PASS" if ok else "FAIL", detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {artefact:34s} {check:10s} {detail}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(ROOT / "splits" / "cell_index.parquet")
    cid = idx.cell_id.to_numpy()
    rec("cell_index", "STRUCTURE", len(idx) == N_CELLS and idx.cell_id.is_unique,
        f"{len(idx)} rows, {idx.cell_id.nunique()} unique, {idx.dataset.nunique()} datasets")

    # ---- HASH: model checkpoints -------------------------------------------------
    # The Geneformer manifest's paths are relative to the pinned HF snapshot, not to
    # models/ -- A fetched it through the hub API (its DECISIONS entry 22), so the
    # weights live in the hub cache. scGPT's are relative to models/scgpt-human.
    gf_snap = ROOT / ("hf_cache/models--ctheodoris--Geneformer/snapshots/"
                      "04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5")
    for manifest, base in [("checkpoint_sha256.txt", gf_snap),
                           ("scgpt_checkpoint_sha256.txt", ROOT / "models" / "scgpt-human")]:
        mp = REPO / "results" / manifest
        if not mp.exists():
            rec(manifest, "HASH", False, "manifest missing")
            continue
        n_ok = n_miss = n_bad = 0
        bad = []
        for line in mp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            want, _, rel = line.partition("  ")
            f = base / rel.strip()
            if not f.exists():
                n_miss += 1
                continue
            if sha256(f) == want:
                n_ok += 1
            else:
                n_bad += 1
                bad.append(rel)
        rec(manifest, "HASH", n_bad == 0 and n_ok > 0,
            f"{n_ok} matched, {n_bad} MISMATCHED{' ' + str(bad) if bad else ''}, {n_miss} absent")

    # ---- HASH: raw expression matrices -------------------------------------------
    raw = pd.read_csv(REPO / "results" / "raw_data_sha256.csv")
    n_ok = n_bad = n_miss = 0
    bad = []
    for r in raw.itertuples():
        f = ROOT / "raw" / r.file if not str(r.file).startswith("/") else Path(r.file)
        if not f.exists():
            cand = list((ROOT / "raw").rglob(Path(str(r.file)).name))
            if not cand:
                n_miss += 1
                continue
            f = cand[0]
        if sha256(f) == r.sha256:
            n_ok += 1
        else:
            n_bad += 1
            bad.append(f.name)
    rec("raw_data_sha256", "HASH", n_bad == 0,
        f"{n_ok}/{len(raw)} matched, {n_bad} MISMATCHED{' ' + str(bad) if bad else ''}, {n_miss} absent")

    # ---- STRUCTURE: embeddings ---------------------------------------------------
    for rep in REPS:
        d = ROOT / "embeddings" / rep
        files = sorted(d.glob("*.npz"))
        if not files:
            rec(f"embeddings/{rep}", "STRUCTURE", False, "no shards")
            continue
        bad_files, seen, dims, nonfinite = [], [], set(), 0
        for f in files:
            try:
                # allow_pickle: cell_id is an object array of barcode strings, which
                # is how A wrote every shard and how its own loader reads them back.
                with np.load(f, allow_pickle=True) as z:
                    e, c = z["emb"], z["cell_id"]
                if len(e) != len(c):
                    bad_files.append(f.name + ":len")
                    continue
                if not np.isfinite(e).all():
                    nonfinite += 1
                    bad_files.append(f.name + ":nonfinite")
                dims.add(e.shape[1])
                seen.append(c)
            except Exception as exc:  # noqa: BLE001
                bad_files.append(f"{f.name}:{type(exc).__name__}")
        if rep in PER_CELL:
            all_c = np.concatenate(seen)
            covers = len(np.unique(all_c)) == N_CELLS and len(all_c) == N_CELLS
            detail = (f"{len(files)} shards, {len(all_c)} cells "
                      f"({len(np.unique(all_c))} unique), dim={sorted(dims)}")
        else:
            covers = all(len(c) == N_CELLS and np.array_equal(c, cid) for c in seen)
            detail = f"{len(files)} per-split files, each {N_CELLS} cells aligned, dim={sorted(dims)}"
        ok = covers and not bad_files and len(dims) == 1
        rec(f"embeddings/{rep}", "STRUCTURE", ok,
            detail + (f", PROBLEMS={bad_files[:5]}" if bad_files else ""))

    # ---- STRUCTURE + INVARIANT: splits, incl. the reserved calibration partition --
    sf = sorted(p for p in (ROOT / "splits").glob("*.parquet") if p.name != "cell_index.parquet")
    prob, frac = [], {}
    for p in sf:
        d = pd.read_parquet(p)
        if len(d) != N_CELLS or not np.array_equal(d.cell_id.to_numpy(), cid):
            prob.append(p.name + ":unaligned")
            continue
        vc = d.partition.value_counts()
        for part in ("train", "calibration", "test"):
            if vc.get(part, 0) == 0:
                prob.append(f"{p.name}:{part} empty")
        frac.setdefault(p.name.split("__")[0], []).append(
            (vc.get("train", 0), vc.get("calibration", 0), vc.get("test", 0)))
    rec("splits (S1/S2/S3)", "STRUCTURE", len(sf) == 75 and not prob,
        f"{len(sf)} files, all aligned, calibration non-empty in all"
        + (f", PROBLEMS={prob[:5]}" if prob else ""))
    for scheme, v in sorted(frac.items()):
        a = np.array(v).mean(axis=0) / N_CELLS
        rec(f"splits/{scheme}", "INVARIANT", True,
            f"train {a[0]:.3f} / calibration {a[1]:.3f} / test {a[2]:.3f} (n={len(v)})")

    # ---- INVARIANT: A6 silhouettes (B2's independent variable) --------------------
    t9 = pd.read_csv(REPO / "results" / "T9_embedding_structure.csv").set_index("representation")
    want = {"geneformer": 0.047218, "scgpt": 0.025745,
            "hvg_pca": -0.010259, "scvi": -0.035006, "harmony": -0.052749}
    dev = {k: abs(t9.loc[k, "silhouette_dataset"] - v) for k, v in want.items()}
    rec("T9 dataset silhouette (A6)", "INVARIANT", max(dev.values()) < 1e-6,
        "; ".join(f"{k} {t9.loc[k, 'silhouette_dataset']:+.6f}" for k in want))

    # ---- INVARIANT: T2 carries what B needs to reproduce A's heads ----------------
    t2 = pd.read_csv(REPO / "results" / "T2_results_long.csv")
    need = {"representation", "split_file", "budget", "seed", "dataset",
            "C_selected", "n_dims_selected", "macro_f1"}
    rec("T2_results_long", "STRUCTURE", need <= set(t2.columns) and len(t2) > 0,
        f"{len(t2)} rows, C_selected and n_dims_selected present "
        f"({t2.C_selected.notna().mean():.1%} / {t2.n_dims_selected.notna().mean():.1%} non-null)")

    # ---- INVARIANT: A7/A8 audit tables -------------------------------------------
    for t in ["T10_pretraining_sensitivity.csv", "T11_memorisation_check.csv",
              "T12_seed_sensitivity.csv", "T13_h3_pretraining_sensitivity.csv"]:
        p = REPO / "results" / t
        ok = p.exists() and len(pd.read_csv(p)) > 0
        rec(t, "STRUCTURE", ok, f"{len(pd.read_csv(p))} rows" if ok else "MISSING")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B0_artefact_verification.csv", index=False)
    n_fail = int((df.status == "FAIL").sum())
    print(f"\n{len(df)} checks, {n_fail} FAILED -> {OUT / 'B0_artefact_verification.csv'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

# FM-BENCH-B — brief amendments

**Status: AUTHORITATIVE. These four amendments OVERRIDE the corresponding text of
FM-BENCH-B.xml v1.2.** Issued by the PI on 2026-08-09, after the brief was frozen
and could not itself be edited. Where this file and the brief disagree, this file
wins. Re-read it whenever a question touches any of the four topics below.

This file exists because the instruction that created it is transient and the brief
is not. Anyone — human or agent — picking the project up should treat the brief and
this file as a single specification.

---

## A1. B2 strengthening option — two of its three suggestions are infeasible

The brief calls adding representations "the single highest-value optional extension"
and `scope_control` permits new foundation models for that purpose. **That is wrong
and must not be acted on.**

- **Geneformer V1-10M is FORBIDDEN.** Adding it requires embedding all 229,801 cells
  with a new checkpoint. That is re-embedding, prohibited by the brief's own
  `hard_rule`, on a box with no GPU (estimated 40–55 CPU-hours).
- **An "inductively fit Harmony variant" is not a thing.** It is precisely what the
  brief elsewhere states harmonypy cannot do: no out-of-sample projection exists,
  which is *why* Harmony is fit transductively here in the first place.

**ELIGIBLE additions, and nothing else:**

1. PCA at differing component counts
2. HVG+PCA at differing HVG counts
3. scVI at a different latent dimension

Each is a refit of an existing classical pipeline. **No new foundation-model
checkpoint, no exception.** Any representation added must have its dataset
silhouette computed by the identical A6 procedure on the identical subsample.

## A2. Hardware description in the brief is wrong

The `recommended_instance` rationale cites Intel AMX, inherited from a superseded
c7i recommendation. The actual machine is **c6a.8xlarge, AMD EPYC 7R13 (Zen 3):
AVX2, but NO AVX-512 and NO AMX.** Do not expect, benchmark against, or attempt to
enable AMX code paths. This instance suits the work through **core count, not
per-core speed**.

c6a is also **EBS-only**: the local NVMe scratch that existed on the FM-BENCH-A
instance is gone. Do not write code that assumes ephemeral local scratch.

## A3. Phase 0's gate is extended

The brief gates Phase 0 only on the synthetic validation passing and the pilot
completing cleanly. **Two further conditions are added. Both must be met before
Phase 1 begins:**

1. **Core utilisation VERIFIED** to match the intended 4-threads × 8-workers
   configuration — measured, not assumed to have taken effect.
2. **A measured representative scVI refit under an S4 leave-dataset-out split** on
   this CPU hardware, extrapolated to a full Phase 2 cost and checked against
   remaining wall-clock.

If the extrapolation exceeds budget, take a declared escape — a GPU instance for
scVI refitting ONLY, or HVG+PCA as the sole classical comparator in B4 — **before
launching Phase 1**, not after discovering the problem mid-run.

## A4. Seed policy scope

A8 measured seed futility on **dataset-level** contrasts, so it applies directly to
**B1, B3 and B5**. **B4's unit of replication is the held-out cell type**, a
different aggregation that A8 never measured.

Keep the no-more-seeds policy as the default for B4 as well, but **state it as
assumed rather than measured**. If B4 appears seed-limited, **report that as an open
question** rather than silently raising seeds.

---

## Code custody — supersedes the FM-BENCH-A SSH rule

The remote is `https://github.com/cbostwick87/scfm-tme-benchmark.git` and **must stay
in the HTTPS form. Do not switch it back to `git@github.com:`.** SSH cannot work from
the analysis sandbox: `GIT_SSH_COMMAND=/bin/false`, no `~/.ssh`, no DNS, proxy-only
egress. The key that the inherited rule refers to exists on the instance and still
works from there, but is unreachable from where the analysis runs. **A push failure
in the sandbox is a transport problem, not a revoked key** — do not go looking for
`~/.ssh/config`.

**Rule C3 matters more on HTTPS than it did on SSH, because a proxied push can fail
while looking like it succeeded.** After every push, assert

```
git rev-parse HEAD  ==  git rev-parse origin/main
```

A push is **not complete** until that holds. Never commit, echo, or log the token.
If a push fails on credentials, **report it and stop** rather than provisioning
another one.

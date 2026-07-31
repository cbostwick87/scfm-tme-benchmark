# Split manifests

Split *definitions* (scheme, seed, holdout unit, partition sizes, and the
donor/group -> partition mapping) are committed here as small CSV/JSON so a third
party can verify the leakage controls without re-running the pipeline.

The per-cell index arrays are large and live on the data volume under
`${SCFM_DATA_ROOT}/splits/`; they are regenerated deterministically by
`make stage.splits` from the committed manifests and the recorded seeds.

The **calibration** partition is reserved for downstream Project B (conformal
prediction) and is not read by this project.

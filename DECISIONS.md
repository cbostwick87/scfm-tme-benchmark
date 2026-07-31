# DECISIONS log

Every deviation from the pre-specified analysis plan, every fallback taken, and every
judgement call with a scientific consequence, recorded with reason and timestamp
(UTC, ISO-8601). Required by provenance requirement P7. This log is surfaced in the
report, not buried.

Append-only. Newest entries at the bottom. Entries are written when the decision is
made, not reconstructed afterwards.

| # | timestamp (UTC) | phase | decision | reason | consequence |
|---|---|---|---|---|---|
| 1 | 2026-07-31T17:30Z | 0 setup | Author field set to a personal address rather than the GitHub `noreply` form. | Operator instruction, asked and answered explicitly. | The address is permanently public in commit history and is routinely scraped. Flagged to the operator before the first commit. |

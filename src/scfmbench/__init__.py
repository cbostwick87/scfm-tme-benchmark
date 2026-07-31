"""scfmbench -- leakage-controlled benchmarking harness for single-cell
foundation-model embeddings vs. classical representations.

Stage modules under `scfmbench.stages` are each independently runnable and
resumable from cached artefacts. Nothing in this package writes to a path
that is not derived from the config's data root.
"""
__version__ = "0.1.0"

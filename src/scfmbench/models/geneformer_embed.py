"""Zero-shot Geneformer cell embeddings.

The `geneformer` package is deliberately NOT imported: its __init__ pulls in a
fine-tuning stack (peft, ray, tensorboard, datasets) that this study never uses,
and installing that chain destabilised the environment every baseline depends on
(see DECISIONS). Instead the checkpoint is loaded through transformers and the
rank-value encoding is implemented here from the PUBLISHED dictionaries.

That places an obligation on this module: the encoding must be VERIFIED, not
assumed. `verify_encoding()` checks the properties the published method
specifies, and the embedding stage refuses to run if any check fails.

Rank-value encoding (Theodoris et al. 2023):
  1. Each cell's expression is normalised to counts-per-10k.
  2. Each gene is divided by its non-zero median expression across the
     pretraining corpus (the `gene_median_dictionary`), which deprioritises
     ubiquitously high genes and elevates cell-state-distinguishing ones.
  3. Genes are ranked in DESCENDING normalised value; the rank ORDER is the
     input sequence. Magnitudes are discarded -- the encoding is the ordering.
  4. The sequence is truncated to the model's input length (2048 for V1,
     4096 for V2) and mapped through the token dictionary.

Cell embedding = mean of final hidden states over non-padding tokens. The BERT
pooler is NOT used: this checkpoint's pooler weights are newly initialised, so
pooling through it would push every cell through an untrained layer.
"""
from __future__ import annotations

import pickle
import pathlib

import numpy as np
import scipy.sparse as sp


class GeneformerEncoder:
    def __init__(self, dict_dir: str | pathlib.Path, model_input_size: int = 4096):
        d = pathlib.Path(dict_dir)
        self.token_dict = pickle.load(open(d / "token_dictionary_gc104M.pkl", "rb"))
        self.median_dict = pickle.load(open(d / "gene_median_dictionary_gc104M.pkl", "rb"))
        self.name2id = pickle.load(open(d / "gene_name_id_dict_gc104M.pkl", "rb"))
        self.model_input_size = model_input_size
        self.pad_token = self.token_dict.get("<pad>", 0)
        self.cls_token = self.token_dict.get("<cls>")
        self.eos_token = self.token_dict.get("<eos>")

    def map_genes(self, gene_symbols) -> tuple[np.ndarray, np.ndarray, dict]:
        """Map dataset gene symbols to (column index, token id, median).

        Genes absent from the token dictionary are DROPPED, and the fraction
        dropped is returned so it can be reported per dataset -- a dataset whose
        genes are largely unrepresented in the model vocabulary is a caveat on
        that dataset's result, not a silent partial encoding.
        """
        cols, toks, meds = [], [], []
        for j, sym in enumerate(gene_symbols):
            ens = self.name2id.get(sym)
            if ens is None:
                continue
            tok = self.token_dict.get(ens)
            med = self.median_dict.get(ens)
            if tok is None or med is None or med <= 0:
                continue
            cols.append(j); toks.append(tok); meds.append(med)
        stats = {"n_input_genes": len(gene_symbols), "n_mapped": len(cols),
                 "frac_mapped": len(cols) / max(len(gene_symbols), 1)}
        return np.asarray(cols, dtype=np.int64), np.asarray(toks, dtype=np.int64), \
               np.asarray(meds, dtype=np.float64), stats

    def encode_rows(self, X: sp.csr_matrix, cols: np.ndarray, toks: np.ndarray,
                    meds: np.ndarray) -> list[np.ndarray]:
        """Rank-value encode each row of a CSR block. Returns token id sequences."""
        Xs = X[:, cols].tocsr()
        out = []
        for i in range(Xs.shape[0]):
            lo, hi = Xs.indptr[i], Xs.indptr[i + 1]
            if hi == lo:
                out.append(np.array([], dtype=np.int64)); continue
            vals = Xs.data[lo:hi].astype(np.float64)
            idx = Xs.indices[lo:hi]
            total = vals.sum()
            if total <= 0:
                out.append(np.array([], dtype=np.int64)); continue
            norm = (vals / total) * 1e4 / meds[idx]      # CP10k then median-scale
            order = np.argsort(-norm, kind="stable")      # DESCENDING rank order
            seq = toks[idx[order]][: self.model_input_size]
            out.append(seq)
        return out


def verify_encoding(enc: GeneformerEncoder) -> dict:
    """Assert the published properties of the rank-value encoding.

    These are cheap checks against a hand-built example. They exist because the
    encoding is reimplemented here rather than taken from the reference package:
    an encoding that is subtly wrong would produce embeddings that look fine,
    cluster plausibly, and quietly misrepresent the model being benchmarked.
    """
    checks = {}
    genes = [g for g in list(enc.name2id) if enc.token_dict.get(enc.name2id[g])
             and enc.median_dict.get(enc.name2id[g], 0) > 0][:4]
    if len(genes) < 4:
        raise RuntimeError("dictionary too small to verify encoding")
    cols, toks, meds, _ = enc.map_genes(genes)

    # 1. A gene with higher median-normalised value must rank earlier.
    counts = np.zeros((1, len(genes)))
    counts[0] = meds * np.array([1.0, 4.0, 2.0, 3.0])   # normalised -> 1,4,2,3
    seq = enc.encode_rows(sp.csr_matrix(counts), cols, toks, meds)[0]
    expected = toks[np.argsort(-np.array([1.0, 4.0, 2.0, 3.0]), kind="stable")]
    checks["rank_order_follows_median_normalised_value"] = bool(np.array_equal(seq, expected))

    # 2. The encoding is scale-invariant: multiplying a cell's counts by a
    #    constant must not change the ordering (CP10k normalisation).
    seq2 = enc.encode_rows(sp.csr_matrix(counts * 17.0), cols, toks, meds)[0]
    checks["scale_invariant"] = bool(np.array_equal(seq, seq2))

    # 3. The median dictionary must actually change the ordering relative to raw
    #    counts -- if it did not, the encoding would be plain count ranking.
    raw = np.array([[1.0, 1.0, 1.0, 1.0]]) * meds.max()
    seq3 = enc.encode_rows(sp.csr_matrix(raw), cols, toks, meds)[0]
    raw_order = toks[np.argsort(-(raw[0] / meds), kind="stable")]
    checks["median_scaling_applied"] = bool(np.array_equal(seq3, raw_order))

    # 4. Zero-count genes must be absent from the sequence.
    z = np.zeros((1, len(genes))); z[0, 1] = meds[1] * 5
    seq4 = enc.encode_rows(sp.csr_matrix(z), cols, toks, meds)[0]
    checks["zeros_excluded"] = bool(len(seq4) == 1 and seq4[0] == toks[1])

    # 5. Truncation respects the model input size.
    checks["truncates_to_input_size"] = True
    if len(enc.token_dict) > enc.model_input_size:
        many = [g for g in list(enc.name2id)
                if enc.token_dict.get(enc.name2id[g])
                and enc.median_dict.get(enc.name2id[g], 0) > 0][: enc.model_input_size + 50]
        c2, t2, m2, _ = enc.map_genes(many)
        if len(c2) > enc.model_input_size:
            big = sp.csr_matrix(np.random.default_rng(0).random((1, len(many))) * 100 + 1)
            s5 = enc.encode_rows(big, c2, t2, m2)[0]
            checks["truncates_to_input_size"] = bool(len(s5) == enc.model_input_size)

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(
            f"Geneformer rank-value encoding failed verification: {failed}. "
            f"Refusing to embed -- a subtly wrong encoding produces plausible-looking "
            f"embeddings that misrepresent the model."
        )
    return checks

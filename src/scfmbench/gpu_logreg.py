"""Multinomial logistic regression with L2, fit on GPU, matching scikit-learn.

WHY THIS EXISTS. The unrestricted-budget cells fit ~138,000 x 768 with 14
classes; on CPU that is ~1100 s each and 270 h across the grid, while the T4
sits at 0% utilisation. This is dense matrix algebra, which is what the GPU is
for.

WHAT IT MUST NOT CHANGE. Guardrail 3 requires the SAME classifier head for
every representation -- the experiment isolates the representation, not the
optimiser. So this is not "a GPU classifier", it is the same estimator solved
on different hardware, and that claim is only worth anything if it is checked.
`assert_matches_sklearn` compares coefficients, predicted labels and macro-F1
against scikit-learn on real data and RAISES on divergence; the sweep calls it
once per session before any GPU fit is trusted.

THE OBJECTIVE, stated explicitly so it can be audited against scikit-learn's:

    min_W  0.5 * ||W||_F^2  +  C * sum_i sw_i * ( -log softmax(x_i W + b)_{y_i} )

matching sklearn's LogisticRegression(penalty="l2", C=C, solver="lbfgs",
class_weight="balanced"): the penalty is NOT scaled by C, the data term IS,
the intercept is unpenalised, and sample weights under `balanced` are
n_samples / (n_classes * count(class_i)).

fp32 is used, not fp16: L-BFGS needs accurate gradients, the matrices here are
small enough that fp32 is not the bottleneck, and a precision-induced
difference from scikit-learn would be indistinguishable from a bug.
"""
from __future__ import annotations

import numpy as np


class GPULogisticRegression:
    """sklearn-compatible surface for the pieces the sweep actually uses."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, tol: float = 1e-6,
                 class_weight: str | None = "balanced", random_state: int = 0,
                 device: str | None = None):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.random_state = random_state
        self._device = device

    def fit(self, X, y):
        import torch
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)

        Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y_idx, dtype=torch.long, device=dev)

        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=k).astype(np.float64)
            cw = n / (k * np.maximum(counts, 1))
            sw = torch.as_tensor(cw[y_idx], dtype=torch.float32, device=dev)
        else:
            sw = torch.ones(n, dtype=torch.float32, device=dev)

        W = torch.zeros((d, k), dtype=torch.float32, device=dev, requires_grad=True)
        b = torch.zeros(k, dtype=torch.float32, device=dev, requires_grad=True)

        opt = torch.optim.LBFGS([W, b], max_iter=self.max_iter, tolerance_grad=self.tol,
                                tolerance_change=self.tol * 1e-2, history_size=10,
                                line_search_fn="strong_wolfe")
        lsm = torch.nn.functional.log_softmax

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = Xt @ W + b
            nll = -lsm(logits, dim=1).gather(1, yt[:, None]).squeeze(1)
            # penalty NOT scaled by C; intercept unpenalised -- as in sklearn
            loss = 0.5 * (W * W).sum() + self.C * (sw * nll).sum()
            loss.backward()
            return loss

        opt.step(closure)

        self.coef_ = W.detach().cpu().numpy().T.copy()       # (k, d), sklearn order
        self.intercept_ = b.detach().cpu().numpy().copy()
        self._W = W.detach()
        self._b = b.detach()
        self._dev = dev
        with torch.no_grad():
            self.n_iter_ = np.array([int(opt.state_dict()["state"][0]["n_iter"])])
        return self

    def _logits(self, X):
        import torch
        Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=self._dev)
        with torch.no_grad():
            return Xt @ self._W + self._b

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self._logits(X), dim=1).cpu().numpy()

    def get_params(self, deep=True):
        return {"C": self.C, "max_iter": self.max_iter, "tol": self.tol,
                "class_weight": self.class_weight, "random_state": self.random_state,
                "device": self._device}

    def set_params(self, **kw):
        for a, v in kw.items():
            setattr(self, "_device" if a == "device" else a, v)
        return self


def assert_matches_sklearn(X, y, C: float = 1.0, tol_pred: float = 0.99,
                           tol_f1: float = 0.005, verbose: bool = True) -> dict:
    """Verify the GPU estimator reproduces scikit-learn on REAL data, or raise.

    WHAT IS AND IS NOT A VALID CRITERION -- established by measurement, not
    assumption. The objective is strictly convex, so it has ONE minimum, and in
    fp64 the two solvers reach it exactly: relative objective gap 3e-11 to 4e-09
    and coefficient correlation 1.000000 at C = 0.01, 1.0 and 100.0.

    Coefficient correlation is therefore NOT a useful gate in the configuration the
    sweep actually runs. At scikit-learn's default tolerance it stopped early
    (objective 1873.79 against the GPU's 1852.42 on the same data -- the GPU found
    the BETTER optimum), and the coefficients differed with correlation 0.988 even
    though labels agreed on 99.7% of cells and macro-F1 differed by 0.0004. A gate
    on coefficient correlation would have rejected a correct implementation for
    out-converging the reference.

    So the gate is on what the study reports: predicted labels and macro-F1. A
    genuinely wrong objective cannot agree on 99%+ of labels AND land within 0.005
    macro-F1 across a range of regularisation strengths. Coefficient correlation is
    still measured and reported, because a large drop is diagnostic, but it is not
    fatal on its own.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    sk = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                            class_weight="balanced", random_state=0).fit(X, y)
    gp = GPULogisticRegression(C=C, random_state=0).fit(X, y)

    p_sk, p_gp = sk.predict(X), gp.predict(X)
    agree = float((p_sk == p_gp).mean())
    f1_sk = f1_score(y, p_sk, average="macro", zero_division=0)
    f1_gp = f1_score(y, p_gp, average="macro", zero_division=0)
    cc = float(np.corrcoef(sk.coef_.ravel(), gp.coef_.ravel())[0, 1])
    out = {"label_agreement": agree, "macro_f1_sklearn": float(f1_sk),
           "macro_f1_gpu": float(f1_gp), "macro_f1_abs_diff": abs(float(f1_sk - f1_gp)),
           "coef_correlation": cc, "n": int(len(y)), "d": int(X.shape[1]),
           "k": int(len(np.unique(y)))}
    if verbose:
        print("GPU-vs-sklearn equivalence:", out, flush=True)

    fails = []
    if agree < tol_pred:
        fails.append(f"label agreement {agree:.4f} < {tol_pred}")
    if out["macro_f1_abs_diff"] > tol_f1:
        fails.append(f"macro-F1 differs by {out['macro_f1_abs_diff']:.4f} > {tol_f1}")
    if fails:
        raise RuntimeError(
            "GPU logistic regression does not reproduce scikit-learn: "
            + "; ".join(fails)
            + f" (coefficient correlation {cc:.4f}, reported for diagnosis). "
              "Refusing to use it -- guardrail 3 requires the SAME classifier head "
              "for every representation, so an optimiser that finds a different "
              "solution would confound the comparison the study exists to make.")
    return out

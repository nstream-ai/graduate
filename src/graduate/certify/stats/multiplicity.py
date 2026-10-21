"""Multiplicity control across a family of slice-level p-values.

All functions return a boolean array, True meaning "reject H0", i.e. the slice
may be certified. Undefined slices must be passed in as p = 1.0 so they count
toward the family size S.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _pvals(pvalues: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("pvalues must lie in [0, 1] and not be NaN")
    return p


def bonferroni(pvalues: ArrayLike, delta: float = 0.05) -> NDArray[np.bool_]:
    """Family-wise error control: reject where p <= delta / S."""
    p = _pvals(pvalues)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    return p <= delta / p.size


def holm(pvalues: ArrayLike, delta: float = 0.05) -> NDArray[np.bool_]:
    """Family-wise error control, step-down (Holm 1979): rejects everything Bonferroni does,
    and possibly more, under arbitrary dependence."""
    p = _pvals(pvalues)
    m = p.size
    passed = np.zeros(m, dtype=bool)
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        if p[i] > delta / (m - rank):
            break
        passed[i] = True
    return passed


def _step_up(p: NDArray[np.float64], q: float, scale: float) -> NDArray[np.bool_]:
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    ok = p[order] <= q * np.arange(1, m + 1) / (m * scale)
    k = int(np.max(np.nonzero(ok)[0])) + 1 if ok.any() else 0
    passed = np.zeros(m, dtype=bool)
    passed[order[:k]] = True
    return passed


def benjamini_hochberg(pvalues: ArrayLike, q: float = 0.05) -> NDArray[np.bool_]:
    """FDR control, valid under independence or positive dependence (PRDS)."""
    return _step_up(_pvals(pvalues), q, 1.0)


def benjamini_yekutieli(pvalues: ArrayLike, q: float = 0.05) -> NDArray[np.bool_]:
    """FDR control under arbitrary dependence. More conservative than BH."""
    p = _pvals(pvalues)
    scale = float(np.sum(1.0 / np.arange(1, p.size + 1))) if p.size else 1.0
    return _step_up(p, q, scale)

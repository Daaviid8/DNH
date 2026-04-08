"""
_criterion_py.py
================
Pure-Python fallback for _criterion.pyx.
Automatically used when the Cython extension is not compiled.
API is identical to the Cython module.
"""

import numpy as np
from math import exp as _exp

EXP           = 0
RATIONAL      = 1
EXP_NORM      = 2
RATIONAL_NORM = 3

_CRITERION_MAP = {
    'exp':           EXP,
    'rational':      RATIONAL,
    'exp_norm':      EXP_NORM,
    'rational_norm': RATIONAL_NORM,
}


def criterion_code(name: str) -> int:
    try:
        return _CRITERION_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown criterion {name!r}. "
            f"Valid values: {list(_CRITERION_MAP)}"
        )


def dnh_impurity(counts_arr, gamma: float, criterion: int) -> float:
    counts = np.asarray(counts_arr, dtype=np.int64)
    N  = int(counts.sum())
    if N == 0:
        return 0.0
    K  = len(counts)
    ss = int(np.dot(counts, counts))
    nd = float(N)
    S  = K * ss / (nd * nd) - 1.0
    if S < 0.0:
        S = 0.0
    if criterion == EXP_NORM or criterion == RATIONAL_NORM:
        if K > 1:
            S /= (K - 1)
    if criterion == RATIONAL or criterion == RATIONAL_NORM:
        return 1.0 / (1.0 + gamma * S)
    return _exp(-gamma * S)


def best_threshold_1d(z_arr, y_arr, gamma: float, n_classes: int,
                      parent_imp: float, criterion: int,
                      min_samples_leaf: int = 1):
    z   = np.asarray(z_arr, dtype=np.float64)
    y   = np.asarray(y_arr, dtype=np.int32)
    N   = len(z)
    K   = n_classes

    if N < 2 * min_samples_leaf + 2:
        return None, -1e300

    order = np.argsort(z, kind='quicksort')
    z     = z[order]
    y     = y[order]

    total = np.bincount(y, minlength=K).astype(np.int64)
    lC    = np.zeros(K, dtype=np.int64)
    ss_l  = 0
    ss_r  = int(np.dot(total, total))

    norm = float(K - 1) if K > 1 else 1.0
    best_gain = -1e300
    best_i    = -1

    for i in range(N - 1):
        k    = int(y[i])
        lC_k = int(lC[k])
        rC_k = int(total[k]) - lC_k

        ss_l += 2 * lC_k + 1
        ss_r -= 2 * rC_k - 1
        lC[k] = lC_k + 1

        lN = i + 1
        rN = N - lN

        if z[i] == z[i + 1]:
            continue
        if lN < min_samples_leaf or rN < min_samples_leaf:
            continue

        lNd = float(lN)
        rNd = float(rN)

        S_l = K * ss_l / (lNd * lNd) - 1.0
        S_r = K * ss_r / (rNd * rNd) - 1.0
        if S_l < 0.0: S_l = 0.0
        if S_r < 0.0: S_r = 0.0

        if criterion == EXP_NORM or criterion == RATIONAL_NORM:
            S_l /= norm
            S_r /= norm

        if criterion == RATIONAL or criterion == RATIONAL_NORM:
            imp_l = 1.0 / (1.0 + gamma * S_l)
            imp_r = 1.0 / (1.0 + gamma * S_r)
        else:
            imp_l = _exp(-gamma * S_l)
            imp_r = _exp(-gamma * S_r)

        wImp = (lNd / N) * imp_l + (rNd / N) * imp_r
        gain = parent_imp - wImp

        if gain > best_gain:
            best_gain = gain
            best_i    = i

    if best_i < 0:
        return None, -1e300

    best_t = 0.5 * (z[best_i] + z[best_i + 1])
    return float(best_t), float(best_gain)


def lda_direction(X_arr, y_arr, n_classes: int):
    d         = X_arr.shape[1]
    e0        = np.zeros(d); e0[0] = 1.0   # deterministic fallback
    centroids = []
    for k in range(n_classes):
        mask = (y_arr == k)
        if mask.any():
            centroids.append(X_arr[mask].mean(axis=0))
    if len(centroids) < 2:
        return e0
    cents = np.array(centroids)
    gc    = cents.mean(axis=0)
    dists = np.linalg.norm(cents - gc, axis=1)
    v     = cents[dists.argmax()] - cents[dists.argmin()]
    n     = np.linalg.norm(v)
    if n < 1e-12:
        return e0
    return v / n


def random_unit(d: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(d)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v    = np.zeros(d)
        v[0] = 1.0
        return v
    return v / n


def riemannian_step(X_arr, y_arr, w_arr, gamma: float, n_classes: int,
                    parent_imp: float, lr: float, eps: float, criterion: int):
    d     = X_arr.shape[1]
    w     = np.asarray(w_arr, dtype=np.float64).copy()
    y_int = np.asarray(y_arr, dtype=np.int32)

    z_base = X_arr @ w
    _, base_gain = best_threshold_1d(z_base, y_int, gamma, n_classes, parent_imp, criterion)
    base_gain = max(base_gain, 0.0)

    W_pert = np.tile(w, (d, 1))
    W_pert[np.arange(d), np.arange(d)] += eps
    norms  = np.linalg.norm(W_pert, axis=1, keepdims=True)
    W_pert /= norms

    Z_pert = X_arr @ W_pert.T   # (N, d)

    grad = np.empty(d, dtype=np.float64)
    for j in range(d):
        _, gj = best_threshold_1d(Z_pert[:, j], y_int, gamma, n_classes, parent_imp, criterion)
        grad[j] = (max(gj, 0.0) - base_gain) / eps

    tang  = grad - np.dot(w, grad) * w
    w_new = w + lr * tang
    n     = np.linalg.norm(w_new)
    if n < 1e-12:
        return w.copy()
    return w_new / n

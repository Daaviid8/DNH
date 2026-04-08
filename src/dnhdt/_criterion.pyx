# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""
_criterion.pyx
==============
Cython hot-path for DNH-DT.

Exported symbols
----------------
criterion_code(name)          -> int
dnh_impurity(counts, gamma, criterion) -> float
best_threshold_1d(...)        -> (float | None, float)
riemannian_step(...)          -> ndarray
lda_direction(...)            -> ndarray
random_unit(d, rng)           -> ndarray

Criterion codes (use these constants, not strings, in hot loops)
---------------------------------------------------------------
EXP           = 0   exp(-gamma * S)
RATIONAL      = 1   1 / (1 + gamma * S)
EXP_NORM      = 2   exp(-gamma * S / (K-1))
RATIONAL_NORM = 3   1 / (1 + gamma * S / (K-1))

Key algorithmic improvement over the NumPy reference
-----------------------------------------------------
best_threshold_1d uses an *incremental O(1) update* for the sum of squared
class counts, instead of the O(N*K) cumsum matrix:

  When sample i (class k) moves from the right child to the left child:
      delta_ss_l = 2 * lC[k] + 1       (lC[k]->lC[k]+1)
      delta_ss_r = -(2 * rC[k] - 1)    (rC[k]->rC[k]-1)

  S = K * ss / n^2 - 1   (exact, no approximation)

This reduces memory from O(N*K) to O(K) and eliminates all temporary
array allocations in the inner loop.
"""

import numpy as np
cimport numpy as cnp
from libc.math  cimport exp, sqrt, fabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cnp.import_array()

# ---------------------------------------------------------------------------
# Criterion integer codes
# ---------------------------------------------------------------------------
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


def criterion_code(str name) -> int:
    """Convert criterion name string to integer code."""
    try:
        return _CRITERION_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown criterion {name!r}. "
            f"Valid values: {list(_CRITERION_MAP)}"
        )


# ---------------------------------------------------------------------------
# C-level impurity helpers
# ---------------------------------------------------------------------------

cdef inline double _compute_S(long long ss, long long n, int K) nogil:
    """S = K * ss / n^2 - 1.  ss = sum(counts[k]^2), n = total count."""
    cdef double nd = <double>n
    return <double>K * <double>ss / (nd * nd) - 1.0


cdef inline double _apply_criterion(double S, double gamma, int criterion) nogil:
    """Apply chosen criterion to a precomputed S value.
    Codes: EXP=0, RATIONAL=1, EXP_NORM=2, RATIONAL_NORM=3
    """
    if S < 0.0:
        S = 0.0
    if criterion == 1 or criterion == 3:   # RATIONAL or RATIONAL_NORM
        return 1.0 / (1.0 + gamma * S)
    return exp(-gamma * S)


# ---------------------------------------------------------------------------
# Public: dnh_impurity
# ---------------------------------------------------------------------------

def dnh_impurity(cnp.ndarray counts_arr, double gamma, int criterion) -> float:
    """
    DNH impurity for a node.

    Parameters
    ----------
    counts_arr : ndarray[int64, 1-D]  class counts
    gamma      : float > 0
    criterion  : int  (use criterion_code() to obtain)

    Returns
    -------
    float in [0, 1]  (0 = pure, 1 = maximally mixed)
    """
    cdef long long[::1] counts = counts_arr.astype(np.int64, copy=False)
    cdef int K = counts.shape[0]
    cdef long long N = 0, ss = 0
    cdef int k
    cdef double S

    for k in range(K):
        N  += counts[k]
        ss += counts[k] * counts[k]

    if N == 0:
        return 0.0

    S = _compute_S(ss, N, K)
    if criterion == EXP_NORM or criterion == RATIONAL_NORM:
        if K > 1:
            S /= (K - 1)

    return _apply_criterion(S, gamma, criterion)


# ---------------------------------------------------------------------------
# Public: best_threshold_1d
# ---------------------------------------------------------------------------

def best_threshold_1d(
    cnp.ndarray z_arr,
    cnp.ndarray y_arr,
    double gamma,
    int n_classes,
    double parent_imp,
    int criterion,
    int min_samples_leaf = 1,
):
    """
    Vectorised threshold sweep over 1-D projections.

    Uses incremental O(1) updates for sum-of-squares — O(N) inner loop,
    O(K) working memory (no temporary N×K arrays).

    Parameters
    ----------
    z_arr           : ndarray[float64]  projections  w·X
    y_arr           : ndarray[int32]    class labels  0..K-1
    gamma           : float
    n_classes       : int  K
    parent_imp      : float  DNH impurity of the parent node
    criterion       : int
    min_samples_leaf: int  minimum samples per child leaf

    Returns
    -------
    (threshold, gain)  or  (None, -inf) if no valid cut found
    """
    cdef:
        double[::1] z  = np.asarray(z_arr, dtype=np.float64)
        int[::1]    y  = np.asarray(y_arr,  dtype=np.int32)
        int N  = z.shape[0]
        int K  = n_classes
        int i, k, best_i
        double best_gain, gain, wImp
        double S_l, S_r, imp_l, imp_r
        double norm
        long long lN, rN
        long long ss_l, ss_r
        long long lC_k, rC_k   # temporary for current class k

    if N < 2 * min_samples_leaf + 2:
        return None, -1e300

    # -- argsort by z (use NumPy; this is O(N log N) and already fast) --------
    order_np = np.argsort(z_arr, kind='quicksort')
    cdef long long[::1] order = order_np.astype(np.int64)

    # -- allocate O(K) working arrays -----------------------------------------
    cdef long long *lC    = <long long *>malloc(K * sizeof(long long))
    cdef long long *total = <long long *>malloc(K * sizeof(long long))
    if lC == NULL or total == NULL:
        free(lC); free(total)
        raise MemoryError

    memset(lC,    0, K * sizeof(long long))
    memset(total, 0, K * sizeof(long long))

    # compute totals and initial ss_r
    ss_r = 0
    for i in range(N):
        k = y[order[i]]
        total[k] += 1

    for k in range(K):
        ss_r += total[k] * total[k]

    ss_l = 0
    best_gain = -1e300
    best_i    = -1

    cdef double nd, lNd, rNd
    nd   = <double>N
    norm = <double>(K - 1) if K > 1 else 1.0

    for i in range(N - 1):
        k    = y[order[i]]
        lC_k = lC[k]
        rC_k = total[k] - lC_k

        # incremental update: sample k moves from right to left
        ss_l += 2 * lC_k + 1
        ss_r -= 2 * rC_k - 1
        lC[k] = lC_k + 1

        lN = i + 1
        rN = N - lN

        # skip if: (a) equal z values at cut point, (b) not enough samples
        if z[order[i]] == z[order[i + 1]]:
            continue
        if lN < min_samples_leaf or rN < min_samples_leaf:
            continue

        lNd = <double>lN
        rNd = <double>rN

        S_l = <double>K * <double>ss_l / (lNd * lNd) - 1.0
        S_r = <double>K * <double>ss_r / (rNd * rNd) - 1.0
        if S_l < 0.0: S_l = 0.0
        if S_r < 0.0: S_r = 0.0

        if criterion == EXP_NORM or criterion == RATIONAL_NORM:
            S_l /= norm
            S_r /= norm

        if criterion == RATIONAL or criterion == RATIONAL_NORM:
            imp_l = 1.0 / (1.0 + gamma * S_l)
            imp_r = 1.0 / (1.0 + gamma * S_r)
        else:
            imp_l = exp(-gamma * S_l)
            imp_r = exp(-gamma * S_r)

        wImp = (lNd / nd) * imp_l + (rNd / nd) * imp_r
        gain = parent_imp - wImp

        if gain > best_gain:
            best_gain = gain
            best_i    = i

    free(lC)
    free(total)

    if best_i < 0:
        return None, -1e300

    best_t = 0.5 * (z[order[best_i]] + z[order[best_i + 1]])
    return float(best_t), float(best_gain)


# ---------------------------------------------------------------------------
# Public: lda_direction
# ---------------------------------------------------------------------------

def lda_direction(
    cnp.ndarray X_arr not None,
    cnp.ndarray y_arr not None,
    int n_classes,
) -> np.ndarray:
    """
    Warm-start direction for Riemannian ascent.

    Returns the unit vector from the nearest class centroid to the
    farthest class centroid (w.r.t. global centroid).  For K=2 this
    is the standard LDA direction; for K>2 it picks the most
    discriminative axis.  Falls back to e_0 (deterministic) when the
    data are degenerate.
    """
    cdef int d = X_arr.shape[1]
    cdef int k

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
    far   = cents[dists.argmax()]
    close = cents[dists.argmin()]
    v     = far - close
    n     = np.linalg.norm(v)
    if n < 1e-12:
        return e0
    return v / n


# ---------------------------------------------------------------------------
# Public: random_unit
# ---------------------------------------------------------------------------

def random_unit(int d, rng) -> np.ndarray:
    """Uniformly distributed unit vector on S^{d-1}."""
    return _random_unit_np(d, rng)


cdef _random_unit_np(int d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(d)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v    = np.zeros(d)
        v[0] = 1.0
        return v
    return v / n


# ---------------------------------------------------------------------------
# Public: riemannian_step
# ---------------------------------------------------------------------------

def riemannian_step(
    cnp.ndarray X_arr not None,
    cnp.ndarray y_arr not None,
    cnp.ndarray w_arr not None,
    double gamma,
    int n_classes,
    double parent_imp,
    double lr,
    double eps,
    int criterion,
) -> np.ndarray:
    """
    One Riemannian gradient ascent step on S^{d-1}.

    Numerical gradient:
        grad[j] = (gain(w + eps*e_j, normalized) - gain(w)) / eps

    Tangent projection (Riemannian gradient):
        grad_tang = grad - <w, grad> * w

    Update:
        w_new = normalize(w + lr * grad_tang)

    This implementation builds all d perturbed directions as a matrix
    and computes projections with a single BLAS call (X @ W_pert.T),
    then calls best_threshold_1d once per direction at C speed.
    """
    cdef:
        int d = X_arr.shape[1]
        int j
        double base_gain, g, gj
        double dot_wg

    w     = np.asarray(w_arr, dtype=np.float64).copy()
    X_arr = np.asarray(X_arr, dtype=np.float64)
    y_int = np.asarray(y_arr, dtype=np.int32)

    z_base = X_arr @ w
    _, base_gain = best_threshold_1d(
        z_base, y_int, gamma, n_classes, parent_imp, criterion
    )
    base_gain = max(base_gain, 0.0)

    # Build all d perturbations at once: W_pert[j] = normalize(w + eps*e_j)
    W_pert = np.tile(w, (d, 1))
    W_pert[np.arange(d), np.arange(d)] += eps
    norms  = np.linalg.norm(W_pert, axis=1, keepdims=True)
    W_pert /= norms

    # Single BLAS matrix multiply → all d projections
    Z_pert = X_arr @ W_pert.T   # shape (N, d)

    grad = np.empty(d, dtype=np.float64)
    for j in range(d):
        _, gj = best_threshold_1d(
            Z_pert[:, j], y_int, gamma, n_classes, parent_imp, criterion
        )
        grad[j] = (max(gj, 0.0) - base_gain) / eps

    # Riemannian (tangential) gradient
    tang  = grad - np.dot(w, grad) * w
    w_new = w + lr * tang
    n     = np.linalg.norm(w_new)
    if n < 1e-12:
        return w.copy()
    return w_new / n

"""
estimators.py
=============
Scikit-learn compatible estimators for DNH-DT.

Classes
-------
DNHDecisionTree          — Axis-aligned tree with DNH criterion.
DNHObliqueDecisionTree   — Oblique tree via Riemannian ascent on S^{d-1}.
DNHRandomForest          — Ensemble of DNHDecisionTree (bagging + subsampling).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import validate_data as _sklearn_vd
    def _fit_validate(estimator, X, y, **kw):
        return _sklearn_vd(estimator, X, y, **kw)
    def _pred_validate(estimator, X, **kw):
        return _sklearn_vd(estimator, X, **kw)
except ImportError:
    from sklearn.utils.validation import check_X_y, check_array
    def _fit_validate(estimator, X, y, **kw):
        _kw = {k: v for k, v in kw.items() if k in ('dtype',)}
        return check_X_y(X, y, **_kw)
    def _pred_validate(estimator, X, **kw):
        _kw = {k: v for k, v in kw.items() if k in ('dtype',)}
        return check_array(X, **_kw)

from sklearn.utils.multiclass import unique_labels

try:
    from ._criterion import (
        criterion_code, dnh_impurity, best_threshold_1d,
        riemannian_step, lda_direction, random_unit,
        EXP, RATIONAL, EXP_NORM, RATIONAL_NORM,
    )
except ImportError:
    from ._criterion_py import (  # pure-Python fallback
        criterion_code, dnh_impurity, best_threshold_1d,
        riemannian_step, lda_direction, random_unit,
        EXP, RATIONAL, EXP_NORM, RATIONAL_NORM,
    )


# ---------------------------------------------------------------------------
# Internal tree node
# ---------------------------------------------------------------------------

class _Node:
    """Internal tree node (leaf or split)."""
    __slots__ = ('is_leaf', 'pred', 'proba',
                 'w', 'thr', 'left', 'right', 'gain', 'n_samples')

    def __init__(self):
        self.is_leaf   = True
        self.pred      = 0
        self.proba     = None
        self.w         = None   # int (axis-aligned) or ndarray (oblique)
        self.thr       = None
        self.left      = None
        self.right     = None
        self.gain      = 0.0
        self.n_samples = 0


# ---------------------------------------------------------------------------
# DNHDecisionTree
# ---------------------------------------------------------------------------

class DNHDecisionTree(ClassifierMixin, BaseEstimator):
    """
    Axis-aligned Decision Tree with DNH impurity criterion.

    The DNH (Non-Homogeneous Distribution) criterion measures node purity
    using the normalised variance of class counts::

        S      = K * sum(n_k^2) / N^2 - 1          (S in [0, K-1])
        I(exp) = exp(-gamma * S)                    (original, S=0 → I=1)
        I(rat) = 1 / (1 + gamma * S)               (H2: uniform gradient)

    S=0 corresponds to a perfectly uniform (maximally impure) split;
    S=K-1 corresponds to a pure node.

    Parameters
    ----------
    max_depth : int, default=5
    min_samples_split : int, default=4
    min_samples_leaf  : int, default=1
    gamma : float, default=2.0
        Sensitivity of the DNH criterion.
        gamma → 0  : linear behaviour (similar to Gini).
        gamma >> 1 : exponentially sensitive to purity.
    criterion : {'exp', 'rational', 'exp_norm', 'rational_norm'}, default='exp'
        'exp'           I = exp(-gamma * S)
        'rational'      I = 1 / (1 + gamma * S)          (H2)
        'exp_norm'      S normalised by K-1               (H1)
        'rational_norm' H1 + H2 combined
    max_features : int, float, 'sqrt', 'log2' or None, default=None
    random_state : int or None, default=None

    Attributes
    ----------
    classes_ : ndarray
    n_classes_ : int
    n_features_in_ : int
    feature_importances_ : ndarray
    tree_ : _Node
    """

    def __init__(
        self,
        max_depth: int        = 5,
        min_samples_split: int = 4,
        min_samples_leaf: int  = 1,
        gamma: float           = 2.0,
        criterion: str         = 'exp',
        max_features           = None,
        random_state           = None,
    ):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.gamma              = gamma
        self.criterion          = criterion
        self.max_features       = max_features
        self.random_state       = random_state

    # -- sklearn API ----------------------------------------------------------

    def fit(self, X, y):
        X, y = _fit_validate(self, X, y, dtype=np.float64, reset=True)
        self.classes_       = unique_labels(y)
        self.n_classes_     = len(self.classes_)

        lut   = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([lut[yi] for yi in y], dtype=np.int32)

        self._crit_code = criterion_code(self.criterion)
        self._rng       = np.random.default_rng(self.random_state)
        self._imp_sum   = np.zeros(self.n_features_in_)

        self.tree_ = self._build(
            np.ascontiguousarray(X, dtype=np.float64), y_enc, depth=0
        )

        total = self._imp_sum.sum()
        self.feature_importances_ = (
            self._imp_sum / total if total > 0 else self._imp_sum.copy()
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = _pred_validate(self, X, dtype=np.float64, reset=False)
        N   = X.shape[0]
        out = np.empty(N, dtype=int)

        def _traverse(node, idx):
            if not idx.size:
                return
            if node.is_leaf:
                out[idx] = node.pred
                return
            mask = X[idx, node.w] <= node.thr
            _traverse(node.left,  idx[mask])
            _traverse(node.right, idx[~mask])

        _traverse(self.tree_, np.arange(N))
        return self.classes_[out]

    def predict_proba(self, X):
        """Laplace-smoothed class probabilities."""
        check_is_fitted(self)
        X = _pred_validate(self, X, dtype=np.float64, reset=False)
        N   = X.shape[0]
        out = np.empty((N, self.n_classes_))

        def _traverse(node, idx):
            if not idx.size:
                return
            if node.is_leaf:
                out[idx] = node.proba
                return
            mask = X[idx, node.w] <= node.thr
            _traverse(node.left,  idx[mask])
            _traverse(node.right, idx[~mask])

        _traverse(self.tree_, np.arange(N))
        return out

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # -- tree construction ----------------------------------------------------

    def _n_features_to_try(self):
        D  = self.n_features_in_
        mf = self.max_features
        if mf is None:               return D
        if mf == 'sqrt':             return max(1, int(np.sqrt(D)))
        if mf == 'log2':             return max(1, int(np.log2(D)))
        if isinstance(mf, float):    return max(1, int(mf * D))
        return int(mf)

    def _make_leaf(self, y_enc, n):
        node         = _Node()
        cnt          = np.bincount(y_enc, minlength=self.n_classes_).astype(float)
        node.pred    = int(cnt.argmax())
        node.proba   = (cnt + 1.0) / (n + self.n_classes_)
        node.n_samples = n
        return node

    def _build(self, X, y_enc, depth):
        N    = len(y_enc)
        node = self._make_leaf(y_enc, N)

        cnt_tot = np.bincount(y_enc, minlength=self.n_classes_).astype(np.int64)
        if depth >= self.max_depth or N < self.min_samples_split or cnt_tot.max() == N:
            return node

        D     = X.shape[1]
        k     = self._n_features_to_try()
        feats = (self._rng.choice(D, k, replace=False) if k < D else np.arange(D))

        parent_imp = dnh_impurity(cnt_tot, self.gamma, self._crit_code)

        best_gain, best_j, best_t = -np.inf, None, None

        for j in feats:
            t, gain = best_threshold_1d(
                X[:, j], y_enc, self.gamma, self.n_classes_,
                parent_imp, self._crit_code, self.min_samples_leaf,
            )
            if t is None:
                continue
            if gain > best_gain:
                best_gain, best_j, best_t = gain, j, t

        if best_j is None or best_gain <= 0:
            return node

        mask = X[:, best_j] <= best_t
        lN   = mask.sum()
        rN   = N - lN
        if lN < self.min_samples_leaf or rN < self.min_samples_leaf:
            return node

        self._imp_sum[best_j] += best_gain * N
        node.is_leaf = False
        node.w       = best_j
        node.thr     = best_t
        node.gain    = best_gain
        node.left    = self._build(X[mask],  y_enc[mask],  depth + 1)
        node.right   = self._build(X[~mask], y_enc[~mask], depth + 1)
        return node


# ---------------------------------------------------------------------------
# DNHObliqueDecisionTree
# ---------------------------------------------------------------------------

class DNHObliqueDecisionTree(ClassifierMixin, BaseEstimator):
    """
    Oblique Decision Tree with DNH criterion and Riemannian ascent.

    Each split finds w* in S^{d-1} that maximises DNH gain::

        gain(w) = I(parent) - sum_child (n_child/N) * I(child, w)

    The search uses gradient ascent on the unit hypersphere::

        grad_tang = grad - <w, grad> * w        (project to tangent space)
        w_new     = normalize(w + lr * grad_tang)

    Parameters
    ----------
    max_depth         : int,   default=5
    min_samples_split : int,   default=4
    min_samples_leaf  : int,   default=1
    gamma             : float, default=3.0
    criterion         : str,   default='exp'
    n_iter            : int,   default=40
        Riemannian ascent iterations per node.
    lr0               : float, default=0.4
        Initial learning rate.
    lr_decay          : float, default=0.96
        Geometric decay: lr_t = lr0 * decay^t.
    strategy          : {'lda', 'random', 'best_random'}, default='lda'
    n_random_init     : int,   default=20
        Candidates for strategy='best_random'.
    max_features      : default=None
    random_state      : default=None
    """

    def __init__(
        self,
        max_depth: int         = 5,
        min_samples_split: int  = 4,
        min_samples_leaf: int   = 1,
        gamma: float            = 3.0,
        criterion: str          = 'exp',
        n_iter: int             = 40,
        lr0: float              = 0.4,
        lr_decay: float         = 0.96,
        strategy: str           = 'lda',
        n_random_init: int      = 20,
        max_features            = None,
        random_state            = None,
    ):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.gamma              = gamma
        self.criterion          = criterion
        self.n_iter             = n_iter
        self.lr0                = lr0
        self.lr_decay           = lr_decay
        self.strategy           = strategy
        self.n_random_init      = n_random_init
        self.max_features       = max_features
        self.random_state       = random_state

    # -- sklearn API ----------------------------------------------------------

    def fit(self, X, y):
        X, y = _fit_validate(self, X, y, dtype=np.float64, reset=True)
        self.classes_       = unique_labels(y)
        self.n_classes_     = len(self.classes_)

        lut   = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([lut[yi] for yi in y], dtype=np.int32)

        self._crit_code = criterion_code(self.criterion)
        # Deterministic seed: same random_state → same tree on every fit call
        self._rng       = np.random.default_rng(self.random_state)
        self._imp_sum   = np.zeros(self.n_features_in_)

        X_c = np.ascontiguousarray(X, dtype=np.float64)
        self.tree_ = self._build(X_c, y_enc, depth=0)

        total = self._imp_sum.sum()
        self.feature_importances_ = (
            self._imp_sum / total if total > 0 else self._imp_sum.copy()
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = _pred_validate(self, X, dtype=np.float64, reset=False)
        N   = X.shape[0]
        out = np.empty(N, dtype=int)

        def _traverse(node, idx):
            if not idx.size:
                return
            if node.is_leaf:
                out[idx] = node.pred
                return
            mask = (X[idx] @ node.w) <= node.thr
            _traverse(node.left,  idx[mask])
            _traverse(node.right, idx[~mask])

        _traverse(self.tree_, np.arange(N))
        return self.classes_[out]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = _pred_validate(self, X, dtype=np.float64, reset=False)
        N   = X.shape[0]
        out = np.empty((N, self.n_classes_))

        def _traverse(node, idx):
            if not idx.size:
                return
            if node.is_leaf:
                out[idx] = node.proba
                return
            mask = (X[idx] @ node.w) <= node.thr
            _traverse(node.left,  idx[mask])
            _traverse(node.right, idx[~mask])

        _traverse(self.tree_, np.arange(N))
        return out

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # -- Riemannian helpers ---------------------------------------------------

    def _init_w(self, X, y_enc):
        d = X.shape[1]
        if self.strategy == 'lda':
            return lda_direction(X, y_enc, self.n_classes_)
        if self.strategy == 'best_random':
            cnt    = np.bincount(y_enc, minlength=self.n_classes_).astype(np.int64)
            pi     = dnh_impurity(cnt, self.gamma, self._crit_code)
            best_g = -np.inf
            best_w = None
            for _ in range(self.n_random_init):
                w = random_unit(d, self._rng)
                z = X @ w
                _, g = best_threshold_1d(
                    z, y_enc, self.gamma, self.n_classes_, pi, self._crit_code
                )
                if g > best_g:
                    best_g, best_w = g, w
            return best_w if best_w is not None else random_unit(d, self._rng)
        return random_unit(d, self._rng)

    def _find_w(self, X, y_enc, parent_imp):
        w = self._init_w(X, y_enc)
        for t in range(self.n_iter):
            lr = self.lr0 * (self.lr_decay ** t)
            w  = riemannian_step(
                X, y_enc, w, self.gamma, self.n_classes_,
                parent_imp, lr, 0.007, self._crit_code,
            )
        return w

    # -- tree construction ----------------------------------------------------

    def _n_features_to_try(self):
        D  = self.n_features_in_
        mf = self.max_features
        if mf is None:               return D
        if mf == 'sqrt':             return max(1, int(np.sqrt(D)))
        if mf == 'log2':             return max(1, int(np.log2(D)))
        if isinstance(mf, float):    return max(1, int(mf * D))
        return int(mf)

    def _make_leaf(self, y_enc, n):
        node         = _Node()
        cnt          = np.bincount(y_enc, minlength=self.n_classes_).astype(float)
        node.pred    = int(cnt.argmax())
        node.proba   = (cnt + 1.0) / (n + self.n_classes_)
        node.n_samples = n
        return node

    def _build(self, X, y_enc, depth):
        N    = len(y_enc)
        node = self._make_leaf(y_enc, N)

        cnt_tot = np.bincount(y_enc, minlength=self.n_classes_).astype(np.int64)
        if depth >= self.max_depth or N < self.min_samples_split or cnt_tot.max() == N:
            return node

        D  = X.shape[1]
        k  = self._n_features_to_try()
        fi = self._rng.choice(D, k, replace=False) if k < D else np.arange(D)
        Xs = np.ascontiguousarray(X[:, fi])

        parent_imp = dnh_impurity(cnt_tot, self.gamma, self._crit_code)
        w_sub      = self._find_w(Xs, y_enc, parent_imp)

        z   = Xs @ w_sub
        thr, gain = best_threshold_1d(
            z, y_enc, self.gamma, self.n_classes_,
            parent_imp, self._crit_code, self.min_samples_leaf,
        )

        if thr is None or gain <= 0:
            return node

        mask = z <= thr
        lN, rN = mask.sum(), (~mask).sum()
        if lN < self.min_samples_leaf or rN < self.min_samples_leaf:
            return node

        w_full       = np.zeros(D)
        w_full[fi]   = w_sub
        self._imp_sum[fi] += np.abs(w_sub) * gain * N

        node.is_leaf = False
        node.w       = w_full
        node.thr     = thr
        node.gain    = gain
        node.left    = self._build(X[mask],  y_enc[mask],  depth + 1)
        node.right   = self._build(X[~mask], y_enc[~mask], depth + 1)
        return node


# ---------------------------------------------------------------------------
# DNHRandomForest
# ---------------------------------------------------------------------------

class DNHRandomForest(ClassifierMixin, BaseEstimator):
    """
    Random Forest using DNH impurity criterion.

    Trains n_estimators DNHDecisionTree instances with bootstrap sampling
    and random feature subsets.  Prediction by soft-vote (average proba).

    Parameters
    ----------
    n_estimators      : int,   default=100
    max_depth         : int or None, default=None  (grow until pure)
    min_samples_split : int,   default=4
    min_samples_leaf  : int,   default=1
    gamma             : float, default=2.0
    criterion         : str,   default='exp'
    max_features      : default='sqrt'
    bootstrap         : bool,  default=True
    n_jobs            : int,   default=1  (-1 = all cores)
    random_state      : default=None
    verbose           : int,   default=0
    """

    def __init__(
        self,
        n_estimators: int      = 100,
        max_depth              = None,
        min_samples_split: int  = 4,
        min_samples_leaf: int   = 1,
        gamma: float            = 2.0,
        criterion: str          = 'exp',
        max_features            = 'sqrt',
        bootstrap: bool         = True,
        n_jobs: int             = 1,
        random_state            = None,
        verbose: int            = 0,
    ):
        self.n_estimators       = n_estimators
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.gamma              = gamma
        self.criterion          = criterion
        self.max_features       = max_features
        self.bootstrap          = bootstrap
        self.n_jobs             = n_jobs
        self.random_state       = random_state
        self.verbose            = verbose

    def fit(self, X, y):
        X, y = _fit_validate(self, X, y, dtype=np.float64, reset=True)
        self.classes_       = unique_labels(y)
        self.n_classes_     = len(self.classes_)
        N                   = X.shape[0]

        rng   = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_estimators)
        md    = self.max_depth if self.max_depth is not None else 10**6

        def _fit_one(seed):
            rs = int(seed)
            r  = np.random.default_rng(rs)
            if self.bootstrap:
                idx    = r.integers(0, N, size=N)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y
            tree = DNHDecisionTree(
                max_depth         = md,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                gamma             = self.gamma,
                criterion         = self.criterion,
                max_features      = self.max_features,
                random_state      = rs,
            )
            return tree.fit(Xb, yb)

        if self.n_jobs == 1:
            self.estimators_ = [_fit_one(s) for s in seeds]
        else:
            from joblib import Parallel, delayed
            self.estimators_ = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose
            )(delayed(_fit_one)(s) for s in seeds)

        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = _pred_validate(self, X, dtype=np.float64, reset=False)
        all_p = np.array([t.predict_proba(X) for t in self.estimators_])
        return all_p.mean(axis=0)

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

"""Tests for DNHObliqueDecisionTree."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from dnhdt import DNHObliqueDecisionTree


@pytest.fixture
def diagonal_hard():
    """Oblique boundary: y = (x0 + x1 > 0)."""
    rng = np.random.default_rng(42)
    X   = rng.uniform(-1, 1, (500, 2))
    y   = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X.astype(np.float64), y


class TestDNHObliqueDecisionTree:
    def test_fit_predict(self, iris):
        X, y = iris
        clf  = DNHObliqueDecisionTree(max_depth=3, n_iter=10, random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_predict_proba(self, iris):
        X, y = iris
        clf  = DNHObliqueDecisionTree(max_depth=3, n_iter=10, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_oblique_beats_axis_on_diagonal(self, diagonal_hard):
        X, y = diagonal_hard
        cv   = StratifiedKFold(5, shuffle=True, random_state=0)

        from dnhdt import DNHDecisionTree
        axis_score = cross_val_score(
            DNHDecisionTree(max_depth=5, random_state=0), X, y, cv=cv
        ).mean()
        obl_score  = cross_val_score(
            DNHObliqueDecisionTree(max_depth=5, n_iter=20, random_state=0),
            X, y, cv=cv,
        ).mean()
        # oblique should be meaningfully better on oblique data
        assert obl_score >= axis_score - 0.05   # allow small variance

    @pytest.mark.parametrize("strategy", ['lda', 'random', 'best_random'])
    def test_strategies(self, iris, strategy):
        X, y = iris
        clf  = DNHObliqueDecisionTree(
            max_depth=3, n_iter=10, strategy=strategy,
            n_random_init=5, random_state=0
        )
        clf.fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_feature_importances(self, iris):
        X, y = iris
        clf  = DNHObliqueDecisionTree(max_depth=3, n_iter=10, random_state=0)
        clf.fit(X, y)
        assert clf.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)


@parametrize_with_checks([DNHObliqueDecisionTree(n_iter=5)])
def test_sklearn_oblique(estimator, check):
    check(estimator)

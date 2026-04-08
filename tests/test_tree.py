"""Tests for DNHDecisionTree (axis-aligned)."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from dnhdt import DNHDecisionTree


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestDNHDecisionTreeAPI:
    def test_fit_predict(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=4, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset(set(np.unique(y)))

    def test_predict_proba_shape(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=4, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()

    def test_score_above_chance(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=5, gamma=3.0, random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.9

    def test_feature_importances_sum_to_one(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=5, random_state=0).fit(X, y)
        assert clf.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_get_set_params(self):
        clf = DNHDecisionTree(max_depth=3, gamma=1.5)
        p   = clf.get_params()
        assert p['max_depth'] == 3
        assert p['gamma']     == 1.5
        clf.set_params(gamma=2.0)
        assert clf.gamma == 2.0

    def test_clone_refit(self, iris):
        from sklearn.base import clone
        X, y = iris
        clf  = DNHDecisionTree(max_depth=3, random_state=0)
        clf.fit(X, y)
        clf2 = clone(clf)
        clf2.fit(X, y)
        assert clf2.score(X, y) > 0.85

    def test_cross_val_score(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=5, gamma=3.0, random_state=0)
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X, y, cv=cv)
        assert scores.mean() > 0.88


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

class TestCriteria:
    @pytest.mark.parametrize("criterion", ['exp', 'rational', 'exp_norm', 'rational_norm'])
    def test_all_criteria_fit_predict(self, iris, criterion):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=4, gamma=2.0, criterion=criterion,
                               random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.8

    def test_invalid_criterion_raises(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(criterion='gini')
        with pytest.raises(ValueError):
            clf.fit(X, y)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_class(self):
        X = np.random.randn(50, 3)
        y = np.zeros(50, dtype=int)
        clf = DNHDecisionTree(max_depth=3).fit(X, y)
        assert (clf.predict(X) == 0).all()

    def test_max_depth_zero(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(max_depth=0).fit(X, y)
        # depth=0 → root is a leaf → majority class
        preds = clf.predict(X)
        assert len(set(preds)) == 1

    def test_min_samples_split_large(self, iris):
        X, y = iris
        clf  = DNHDecisionTree(min_samples_split=10000).fit(X, y)
        preds = clf.predict(X)
        assert len(set(preds)) == 1  # no splits possible

    def test_max_features_sqrt(self, cancer):
        X, y = cancer
        clf  = DNHDecisionTree(max_depth=5, max_features='sqrt', random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.9


# ---------------------------------------------------------------------------
# sklearn estimator checks (subset)
# ---------------------------------------------------------------------------

@parametrize_with_checks([DNHDecisionTree()])
def test_sklearn_compatible(estimator, check):
    check(estimator)

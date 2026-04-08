"""Tests for DNHRandomForest."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from dnhdt import DNHRandomForest


class TestDNHRandomForest:
    def test_fit_predict(self, iris):
        X, y = iris
        rf   = DNHRandomForest(n_estimators=20, random_state=0)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba(self, iris):
        X, y = iris
        rf   = DNHRandomForest(n_estimators=20, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()

    def test_score_iris(self, iris):
        X, y = iris
        cv   = StratifiedKFold(5, shuffle=True, random_state=0)
        rf   = DNHRandomForest(n_estimators=50, gamma=2.0, random_state=0)
        scores = cross_val_score(rf, X, y, cv=cv)
        assert scores.mean() > 0.90

    def test_feature_importances(self, iris):
        X, y = iris
        rf   = DNHRandomForest(n_estimators=20, random_state=0).fit(X, y)
        fi   = rf.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert fi.sum()  == pytest.approx(1.0, abs=1e-6)

    def test_no_bootstrap(self, iris):
        X, y = iris
        rf   = DNHRandomForest(n_estimators=10, bootstrap=False,
                               random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.9

    def test_max_depth_none_grows_full(self, iris):
        X, y = iris
        rf   = DNHRandomForest(n_estimators=5, max_depth=None, random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.95

    @pytest.mark.parametrize("max_features", ['sqrt', 'log2', 0.5, 2])
    def test_max_features_variants(self, cancer, max_features):
        X, y = cancer
        rf   = DNHRandomForest(n_estimators=10, max_features=max_features,
                               random_state=0)
        rf.fit(X, y)
        assert rf.score(X, y) > 0.85

    def test_n_jobs_parallel(self, iris):
        X, y = iris
        rf1  = DNHRandomForest(n_estimators=20, n_jobs=1,  random_state=0).fit(X, y)
        rf2  = DNHRandomForest(n_estimators=20, n_jobs=-1, random_state=0).fit(X, y)
        # scores may differ slightly (different thread scheduling) but should be close
        assert abs(rf1.score(X, y) - rf2.score(X, y)) < 0.05


@parametrize_with_checks([DNHRandomForest(n_estimators=5)])
def test_sklearn_forest(estimator, check):
    check(estimator)

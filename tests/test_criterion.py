"""Unit tests for the criterion module (Cython or pure-Python)."""

import numpy as np
import pytest

from dnhdt import (
    criterion_code,
    dnh_impurity,
    best_threshold_1d,
    EXP, RATIONAL, EXP_NORM, RATIONAL_NORM,
)


# ---------------------------------------------------------------------------
# criterion_code
# ---------------------------------------------------------------------------

class TestCriterionCode:
    def test_valid_names(self):
        assert criterion_code('exp')           == EXP
        assert criterion_code('rational')      == RATIONAL
        assert criterion_code('exp_norm')      == EXP_NORM
        assert criterion_code('rational_norm') == RATIONAL_NORM

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            criterion_code('gini')


# ---------------------------------------------------------------------------
# dnh_impurity
# ---------------------------------------------------------------------------

class TestDnhImpurity:
    def test_uniform_node_is_one(self):
        # Uniform distribution: S=0, I=exp(0)=1 (maximum impurity)
        counts = np.array([5, 5], dtype=np.int64)
        assert dnh_impurity(counts, gamma=2.0, criterion=EXP) == pytest.approx(1.0)

    def test_uniform_three_class_is_one(self):
        counts = np.array([10, 10, 10], dtype=np.int64)
        assert dnh_impurity(counts, gamma=2.0, criterion=EXP) == pytest.approx(1.0)

    def test_pure_node_low_impurity(self):
        # Pure node [n, 0, 0]: S=K-1 → I = exp(-gamma*(K-1)) which is small
        counts = np.array([10, 0, 0], dtype=np.int64)
        val = dnh_impurity(counts, gamma=2.0, criterion=EXP)
        # K=3, S=2, I=exp(-4) ≈ 0.018
        assert val == pytest.approx(np.exp(-4.0), rel=1e-6)
        assert val < 0.1  # much less than 1

    def test_pure_node_rational(self):
        counts = np.array([10, 0, 0], dtype=np.int64)
        val = dnh_impurity(counts, gamma=2.0, criterion=RATIONAL)
        # K=3, S=2, I=1/(1+4)=0.2
        assert val == pytest.approx(1.0 / 5.0, rel=1e-6)

    def test_rational_range(self):
        counts = np.array([8, 2], dtype=np.int64)
        val = dnh_impurity(counts, gamma=2.0, criterion=RATIONAL)
        assert 0.0 < val < 1.0

    def test_empty_node(self):
        counts = np.array([0, 0], dtype=np.int64)
        assert dnh_impurity(counts, gamma=2.0, criterion=EXP) == 0.0

    def test_exp_norm_range_k2_unchanged(self):
        # K=2: K-1=1, normalisation S/1=S → identical to exp
        counts = np.array([7, 3], dtype=np.int64)
        v_exp      = dnh_impurity(counts, gamma=2.0, criterion=EXP)
        v_exp_norm = dnh_impurity(counts, gamma=2.0, criterion=EXP_NORM)
        assert v_exp == pytest.approx(v_exp_norm, rel=1e-9)

    def test_exp_norm_k5_higher_than_exp_pure(self):
        # Pure node K=5: S=4, exp gives exp(-8)≈0.00034, exp_norm gives exp(-2)≈0.135
        # exp_norm gives HIGHER impurity value (less extreme) for pure nodes
        counts = np.array([10, 0, 0, 0, 0], dtype=np.int64)
        v_exp      = dnh_impurity(counts, gamma=2.0, criterion=EXP)
        v_exp_norm = dnh_impurity(counts, gamma=2.0, criterion=EXP_NORM)
        assert v_exp_norm > v_exp  # normalisation makes the range K-independent

    def test_impurity_decreases_with_gamma(self):
        # Higher gamma → lower impurity for impure nodes (more sensitive)
        counts = np.array([8, 2], dtype=np.int64)
        v1 = dnh_impurity(counts, gamma=1.0, criterion=EXP)
        v2 = dnh_impurity(counts, gamma=5.0, criterion=EXP)
        assert v1 > v2  # higher gamma → more discriminating → lower impurity


# ---------------------------------------------------------------------------
# best_threshold_1d
# ---------------------------------------------------------------------------

class TestBestThreshold1d:
    def _make_parent_imp(self, y, n_classes, criterion):
        counts = np.bincount(y, minlength=n_classes).astype(np.int64)
        return dnh_impurity(counts, gamma=2.0, criterion=criterion)

    def test_perfect_split(self):
        z = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        parent = self._make_parent_imp(y, 2, EXP)
        t, gain = best_threshold_1d(z, y, 2.0, 2, parent, EXP, 1)
        assert t is not None
        assert gain > 0

    def test_no_valid_split_uniform_z(self):
        z = np.ones(10, dtype=np.float64)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        parent = self._make_parent_imp(y, 2, EXP)
        t, gain = best_threshold_1d(z, y, 2.0, 2, parent, EXP, 1)
        assert t is None

    def test_threshold_near_boundary(self):
        rng = np.random.default_rng(42)
        z   = rng.uniform(0, 10, 200).astype(np.float64)
        y   = (z > 5).astype(np.int32)
        parent = self._make_parent_imp(y, 2, EXP)
        t, gain = best_threshold_1d(z, y, 2.0, 2, parent, EXP, 1)
        assert t is not None
        assert 4.0 < t < 6.0
        assert gain > 0

    def test_all_criteria_positive_gain(self):
        rng = np.random.default_rng(0)
        z   = rng.uniform(-3, 3, 300).astype(np.float64)
        y   = (z > 0).astype(np.int32)
        for crit in [EXP, RATIONAL, EXP_NORM, RATIONAL_NORM]:
            parent = self._make_parent_imp(y, 2, crit)
            t, gain = best_threshold_1d(z, y, 2.0, 2, parent, crit, 1)
            assert t is not None, f"criterion={crit} returned no threshold"
            assert gain > 0,      f"criterion={crit} gain={gain} not positive"

    def test_min_samples_leaf_respected(self):
        z   = np.arange(20, dtype=np.float64)
        y   = np.array([0]*10 + [1]*10, dtype=np.int32)
        parent = self._make_parent_imp(y, 2, EXP)
        t, gain = best_threshold_1d(z, y, 2.0, 2, parent, EXP, min_samples_leaf=8)
        assert t is not None
        left_count  = int((z <= t).sum())
        right_count = len(z) - left_count
        assert left_count  >= 8
        assert right_count >= 8

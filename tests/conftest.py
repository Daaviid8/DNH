"""Shared fixtures for all test modules."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


@pytest.fixture(scope="session")
def iris():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture(scope="session")
def wine():
    X, y = load_wine(return_X_y=True)
    return X, y


@pytest.fixture(scope="session")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


@pytest.fixture(scope="session")
def diagonal_2d():
    """Linearly separable dataset with oblique boundary."""
    rng = np.random.default_rng(0)
    N   = 400
    X   = rng.uniform(-1, 1, (N, 2))
    y   = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X.astype(np.float64), y.astype(np.int32)

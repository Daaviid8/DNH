# DNH-DT

**Decision Trees with Non-Homogeneous Distribution (DNH) Criterion**

[![CI](https://github.com/DavidCortes/dnhdt/actions/workflows/ci.yml/badge.svg)](https://github.com/DavidCortes/dnhdt/actions)
[![PyPI](https://img.shields.io/pypi/v/dnhdt.svg)](https://pypi.org/project/dnhdt/)
[![Python](https://img.shields.io/pypi/pyversions/dnhdt)](https://pypi.org/project/dnhdt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Based on the paper **"Distribución No Homogénea (DNH) en Mezclas Líquidas"** — Cortés (2025)

---

## What is DNH-DT?

Classical impurity criteria (Gini, Entropy) measure node purity through
information theory.  The DNH criterion derives purity from physical chemistry:
the **normalised variance of class counts** in a mixture:

```
S = K · Σ nk² / N² − 1        (S ∈ [0, K−1])

I_exp(S) = exp(−γ · S)          (original DNH, Eq. 2 of the paper)
I_rat(S) = 1 / (1 + γ · S)     (H2: uniform gradient everywhere)
```

where `K` = number of classes, `nk` = count of class k, `N` = total samples.

### Key properties

| Property | DNH | Gini |
|----------|-----|------|
| Derived from physical chemistry | Yes | No |
| Tunable sensitivity `γ` | Yes | No |
| Closed-form error law `E = S/(1+S)` | Yes | No |
| Normalisation independent of K | H1 variant | No |
| Uniform gradient | H2 variant | Approximate |

---

## Installation

```bash
# From PyPI (pre-built wheels)
pip install dnhdt

# From source (requires Cython + C compiler)
git clone https://github.com/DavidCortes/dnhdt.git
cd dnhdt
pip install -e ".[dev]"
```

> **Windows**: install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) before building from source.

---

## Quick start

```python
from dnhdt import DNHDecisionTree, DNHObliqueDecisionTree, DNHRandomForest
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_iris(return_X_y=True)
cv   = StratifiedKFold(5, shuffle=True, random_state=0)

# Axis-aligned tree
clf = DNHDecisionTree(max_depth=5, gamma=3.0)
print(cross_val_score(clf, X, y, cv=cv).mean())  # ~0.96

# Oblique tree (Riemannian ascent on S^{d-1})
obl = DNHObliqueDecisionTree(max_depth=4, gamma=3.0, n_iter=30)
print(cross_val_score(obl, X, y, cv=cv).mean())  # ~0.95

# Random Forest
rf = DNHRandomForest(n_estimators=100, gamma=2.0)
print(cross_val_score(rf, X, y, cv=cv).mean())   # ~0.96
```

---

## API reference

### `DNHDecisionTree`

```python
DNHDecisionTree(
    max_depth          = 5,
    min_samples_split  = 4,
    min_samples_leaf   = 1,
    gamma              = 2.0,     # DNH sensitivity
    criterion          = 'exp',   # 'exp' | 'rational' | 'exp_norm' | 'rational_norm'
    max_features       = None,    # None | 'sqrt' | 'log2' | int | float
    random_state       = None,
)
```

### `DNHObliqueDecisionTree`

```python
DNHObliqueDecisionTree(
    max_depth          = 5,
    gamma              = 3.0,
    criterion          = 'exp',
    n_iter             = 40,      # Riemannian ascent iterations per node
    lr0                = 0.4,     # initial learning rate
    lr_decay           = 0.96,    # geometric decay: lr_t = lr0 * decay^t
    strategy           = 'lda',   # 'lda' | 'random' | 'best_random'
    n_random_init      = 20,
    max_features       = None,
    random_state       = None,
)
```

### `DNHRandomForest`

```python
DNHRandomForest(
    n_estimators       = 100,
    max_depth          = None,
    gamma              = 2.0,
    criterion          = 'exp',
    max_features       = 'sqrt',
    bootstrap          = True,
    n_jobs             = 1,       # -1 = all cores
    random_state       = None,
)
```

### Criterion variants

| `criterion` | Formula | Notes |
|-------------|---------|-------|
| `'exp'`           | `exp(−γ·S)`            | Original DNH (Eq. 2) |
| `'rational'`      | `1/(1+γ·S)`            | H2: uniform gradient |
| `'exp_norm'`      | `exp(−γ·S/(K−1))`      | H1: K-independent range |
| `'rational_norm'` | `1/(1+γ·S/(K−1))`      | H1+H2 combined |

---

## Criterion `gamma` guidelines

| Regime | Behaviour |
|--------|-----------|
| `γ → 0` | Linear (≈ Gini) |
| `γ = 1–3` | Moderate discrimination |
| `γ > 5` | Exponentially sensitive to purity |

Tune `gamma` with `GridSearchCV` or `cross_val_score` over `[0.5, 1, 2, 3, 5, 10]`.

---

## Performance

The Cython extension implements an **incremental O(1) update** for the
sum-of-squared class counts in the threshold sweep:

```
When sample of class k moves from right to left:
    ss_l += 2·lC[k] + 1          # (lC[k]+1)² − lC[k]²
    ss_r −= 2·rC[k] − 1          # (rC[k]−1)² − rC[k]²
```

This reduces the inner loop from O(N·K) to O(N) with O(K) working memory,
eliminating all temporary array allocations.

### Benchmark (N=2000, D=20, K=3)

| Model | Fit time |
|-------|----------|
| sklearn CART (gini, C) | ~2 ms |
| DNH-DT exp (Cython) | ~5 ms |
| DNH-DT rational (Cython) | ~5 ms |

---

## Repository layout

```
dnhdt/
├── src/dnhdt/
│   ├── _criterion.pyx    # Cython hot-path: threshold sweep, Riemannian step
│   ├── _criterion.pxd    # C-level declarations
│   ├── estimators.py     # DNHDecisionTree / DNHObliqueDecisionTree / DNHRandomForest
│   ├── __init__.py
│   └── _version.py
├── tests/                # pytest suite
├── benchmarks/           # timing scripts
├── examples/             # runnable demos
├── .github/workflows/    # CI + wheel builds
├── pyproject.toml
└── setup.py              # Cython extension compilation
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Rebuild Cython extension after edits to .pyx
python setup.py build_ext --inplace

# Run benchmarks
python benchmarks/bench_fit.py --n 5000 --d 30 --k 5
```

---

## Citation

If you use DNH-DT in research, please cite:

```bibtex
@article{cortes2025dnh,
  title   = {Distribucion No Homogenea (DNH) en Mezclas Liquidas},
  author  = {Cort\'es, David},
  year    = {2025},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

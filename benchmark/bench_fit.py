"""
benchmarks/bench_fit.py
=======================
Compare DNHDecisionTree (Cython) vs sklearn DecisionTreeClassifier.

Usage
-----
    python benchmarks/bench_fit.py
    python benchmarks/bench_fit.py --n 5000 --d 50 --repeat 10
"""

import argparse
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from dnhdt import DNHDecisionTree, DNHRandomForest


def bench(label, fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mu  = np.mean(times)
    std = np.std(times)
    print(f"  {label:<35s}  {mu*1000:8.2f} ms  (+/- {std*1000:.2f} ms)")
    return mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',      type=int, default=2000,  help='samples')
    parser.add_argument('--d',      type=int, default=20,    help='features')
    parser.add_argument('--k',      type=int, default=3,     help='classes')
    parser.add_argument('--repeat', type=int, default=5,     help='repetitions')
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.n, n_features=args.d,
        n_informative=max(2, args.d // 2),
        n_classes=args.k, n_clusters_per_class=1,
        random_state=0,
    )
    X = X.astype(np.float64)

    print(f"\nDataset: N={args.n}, D={args.d}, K={args.k}, repeat={args.repeat}")
    print("-" * 60)

    skl = DecisionTreeClassifier(max_depth=10, criterion='gini', random_state=0)
    dnh = DNHDecisionTree(max_depth=10, gamma=2.0, random_state=0)

    t_skl = bench("sklearn CART (gini)",            lambda: skl.fit(X, y), args.repeat)
    t_dnh = bench("DNH-DT  (exp,  Cython)",         lambda: dnh.fit(X, y), args.repeat)

    for crit in ['rational', 'exp_norm', 'rational_norm']:
        clf = DNHDecisionTree(max_depth=10, gamma=2.0, criterion=crit, random_state=0)
        bench(f"DNH-DT  ({crit})", lambda c=clf: c.fit(X, y), args.repeat)

    print(f"\n  Speedup DNH/SKL: {t_dnh/t_skl:.2f}x")

    # Random Forest benchmark
    print("\n" + "-" * 60)
    n_est = 50
    rf_skl = __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
        n_estimators=n_est, max_depth=8, random_state=0, n_jobs=1
    )
    rf_dnh = DNHRandomForest(n_estimators=n_est, max_depth=8, gamma=2.0, random_state=0, n_jobs=1)

    bench(f"sklearn RF  (n={n_est}, n_jobs=1)", lambda: rf_skl.fit(X, y), max(2, args.repeat // 2))
    bench(f"DNH-RF      (n={n_est}, n_jobs=1)", lambda: rf_dnh.fit(X, y), max(2, args.repeat // 2))


if __name__ == '__main__':
    main()

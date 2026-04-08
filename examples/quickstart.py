"""
examples/quickstart.py
======================
Minimal demo: DNHDecisionTree, DNHObliqueDecisionTree, DNHRandomForest
on the three sklearn toy datasets.

Run
---
    python examples/quickstart.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from dnhdt import DNHDecisionTree, DNHObliqueDecisionTree, DNHRandomForest

datasets = {
    'Iris   (150x4,  3c)': load_iris(return_X_y=True),
    'Wine   (178x13, 3c)': load_wine(return_X_y=True),
    'Cancer (569x30, 2c)': load_breast_cancer(return_X_y=True),
}

models = {
    'CART (gini)':     DecisionTreeClassifier(max_depth=5, random_state=0),
    'DNH-axis exp':    DNHDecisionTree(max_depth=5, gamma=2.0, random_state=0),
    'DNH-axis rat':    DNHDecisionTree(max_depth=5, gamma=2.0,
                                       criterion='rational', random_state=0),
    'DNH-oblique':     DNHObliqueDecisionTree(max_depth=4, gamma=3.0,
                                              n_iter=25, random_state=0),
    'DNH-RF (50 trees)': DNHRandomForest(n_estimators=50, gamma=2.0,
                                          random_state=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

header_names = list(models.keys())
col_w = 18
print(f"\n{'Dataset':<25}", end='')
for h in header_names:
    print(f"{h:>{col_w}}", end='')
print()
print('-' * (25 + col_w * len(header_names)))

for ds_name, (X, y) in datasets.items():
    print(f"{ds_name:<25}", end='')
    for model in models.values():
        scores = cross_val_score(model, X, y, cv=cv)
        print(f"{scores.mean():>{col_w}.4f}", end='')
    print()

print()
print('5-fold CV accuracy (mean over splits).')

# %% [markdown]
# # Lab 4: Trees & ensembles, by hand and with scikit-learn 🌳
#
# You'll compute Gini impurity and entropy by hand, write the split search a
# decision tree runs at every node, sweep max_depth to watch a tree overfit,
# compare bagging with a random forest using out-of-bag (OOB) scores, write
# gradient boosting yourself (each tree fits the residuals) and check it
# against scikit-learn, use early stopping, and blend two models with
# out-of-fold predictions.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase4_trees.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 4.1 (impurity, splits, depth), 4.2 (bagging,
# OOB), 4.3 (boosting, early stopping), 4.5 (out-of-fold blending)
# Everything runs on a laptop CPU in seconds, offline.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")  # keeps the histogram booster from oversubscribing small CPUs

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_friedman1, make_moons
from sklearn.ensemble import (BaggingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

rng = np.random.default_rng(0)  # one seeded random generator → reproducible results


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def show():
    """plt.show(), but quietly skipped when there is no screen (e.g. MPLBACKEND=Agg)."""
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


def close(a, b, tol=1e-3):
    """True if a and b (numbers or arrays) agree to within tol."""
    return bool(np.allclose(a, b, atol=tol))


# %% TODO 1: Gini impurity and entropy of a node   (Lesson 4.1)
def gini(counts):
    """counts: how many rows of each class sit in the node, e.g. [5, 5].
    Gini = 1 − Σ p², where p = the share of each class."""
    # TODO 1: p = counts / total, then 1 − Σ p². Hint: np.asarray(counts, dtype=float) / np.sum(counts)
    raise NotImplementedError("TODO 1: gini")


def entropy(counts):
    """Entropy in bits = − Σ p log₂ p (classes with p = 0 contribute nothing)."""
    # TODO 1: drop the zero shares (p = p[p > 0]), then −Σ p·log₂ p with np.log2
    raise NotImplementedError("TODO 1: entropy")


check(close(gini([5, 5]), 0.5) and close(entropy([5, 5]), 1.0), "a 50/50 node: Gini 0.5, entropy 1 bit")
check(close(gini([4, 0]), 0.0) and close(entropy([4, 0]), 0.0), "a pure node: 0 and 0")
check(close(gini([1, 5]), 0.2778) and close(entropy([1, 5]), 0.6500), "1 churner among 6: Gini 0.278, entropy 0.650")
check(close(gini([6, 2]), 0.375), "quiz 1 of Lesson 4.1: 6 vs 2 → Gini 0.375")

# %% TODO 2: find the best threshold on one feature   (Lesson 4.1)
# Lesson 4.1's ten customers: support calls in the last 3 months, and whether they churned.
calls = np.array([5, 0, 4, 1, 3, 0, 1, 6, 2, 0])
churned = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0])


def best_split(x, y, impurity=gini):
    """Try a threshold halfway between every pair of neighbouring distinct values of x.
    Return (threshold, gain) with the largest gain, where
    gain = impurity(parent) − (n_left/n · impurity(left) + n_right/n · impurity(right))
    and 'left' means x <= threshold. Classes are 0/1, so counts = np.bincount(y, minlength=2)."""
    # TODO 2: loop over t in (values[:-1] + values[1:]) / 2; left = y[x <= t]; keep the t with the largest parent − weighted children
    raise NotImplementedError("TODO 2: best_split")


t, g = best_split(calls, churned)
print(f"best question: calls <= {t}?  gain = {g:.3f}")
check(close(t, 2.5) and close(g, 0.3333), "best split is calls ≤ 2.5 with Gini gain 0.333 (Lesson 4.1's question B)")
t_e, g_e = best_split(calls, churned, impurity=entropy)
check(close(t_e, 2.5) and close(g_e, 0.6100), "entropy agrees: same threshold, information gain 0.610 bits")

# %% TODO 3: sweep max_depth and watch a tree overfit   (Lesson 4.1)
X_m, y_m = make_moons(n_samples=600, noise=0.4, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_m, y_m, test_size=0.5, random_state=0, stratify=y_m)


def depth_sweep(depths):
    """For each depth d (None = unlimited), fit DecisionTreeClassifier(max_depth=d, random_state=0)
    on (X_tr, y_tr). Return two lists: training accuracies and validation accuracies (on X_va, y_va)."""
    # TODO 3: for d in depths: fit the tree, append tree.score(X_tr, y_tr) and tree.score(X_va, y_va)
    raise NotImplementedError("TODO 3: depth_sweep")


depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, None]
tr_acc, va_acc = depth_sweep(depths)
for d, a, b in zip(depths, tr_acc, va_acc):
    print(f"max_depth={str(d):4s}  train {a:.3f}  validation {b:.3f}")
best_d = depths[int(np.argmax(va_acc))]
check(tr_acc[-1] == 1.0, "an unlimited tree gets 100% of the training data right")
check(max(va_acc) - va_acc[-1] > 0.02, f"…but the best depth ({best_d}) beats it on validation by more than 2 points")
check(tr_acc == sorted(tr_acc), "training accuracy never goes down as depth grows")

plt.figure(figsize=(6, 3.5))
xs = [str(d) for d in depths]
plt.plot(xs, tr_acc, "o-", label="train"); plt.plot(xs, va_acc, "o-", label="validation")
plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.title("Deeper trees: better on train, not on new data"); plt.legend(); plt.tight_layout(); show()

# %% TODO 4: bagging vs random forest, with out-of-bag scores   (Lesson 4.2)
X_bc, y_bc = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_bc, y_bc, test_size=0.3, random_state=0, stratify=y_bc)


def bag_and_forest():
    """Fit (1) BaggingClassifier of 200 full-depth trees (its default base model is a decision tree)
    and (2) RandomForestClassifier with 200 trees, both with oob_score=True and random_state=0.
    Return the two fitted models."""
    # TODO 4: BaggingClassifier(n_estimators=200, oob_score=True, random_state=0) and the same for RandomForestClassifier; .fit(Xb_tr, yb_tr)
    raise NotImplementedError("TODO 4: bag_and_forest")


bag, rf = bag_and_forest()
single = DecisionTreeClassifier(random_state=0).fit(Xb_tr, yb_tr)
print(f"one tree test {single.score(Xb_te, yb_te):.3f} | bagging OOB {bag.oob_score_:.3f} test {bag.score(Xb_te, yb_te):.3f}"
      f" | forest OOB {rf.oob_score_:.3f} test {rf.score(Xb_te, yb_te):.3f}")
check(rf.score(Xb_te, yb_te) > single.score(Xb_te, yb_te), "the forest beats a single deep tree on the test set")
check(abs(rf.oob_score_ - rf.score(Xb_te, yb_te)) < 0.05, "the free OOB estimate is within 5 points of the real test score")
n = len(yb_tr)
check(close((1 - 1 / n) ** n, np.exp(-1), tol=0.01), f"each tree leaves out (1 − 1/n)ⁿ ≈ {np.exp(-1):.3f} of the rows: those are its OOB rows")

# %% TODO 5: gradient boosting by hand   (Lesson 4.3)
X_f, y_f = make_friedman1(n_samples=800, noise=1.0, random_state=0)
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_f, y_f, test_size=0.25, random_state=0)


def boost(X, y, n_rounds, lr, max_depth):
    """Start from F = mean(y). Each round: residuals r = y − F; fit
    DecisionTreeRegressor(max_depth=max_depth, criterion="friedman_mse", random_state=0) to (X, r);
    update F ← F + lr · tree.predict(X). Return (f0, list_of_trees, train_mse_per_round)."""
    # TODO 5: F = np.full(len(y), mean); repeat: r = y − F, tree.fit(X, r), F = F + lr * tree.predict(X); record np.mean((y − F)**2)
    raise NotImplementedError("TODO 5: boost")


def predict(f0, trees, lr, X):
    return f0 + lr * sum(t.predict(X) for t in trees)


f0, trees, mse = boost(Xf_tr, yf_tr, n_rounds=100, lr=0.1, max_depth=3)
mine = predict(f0, trees, 0.1, Xf_te)
sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(Xf_tr, yf_tr)
print(f"train MSE: round 1 {mse[0]:.2f} → round 100 {mse[-1]:.2f};  test MSE mine {np.mean((yf_te - mine) ** 2):.3f}, "
      f"scikit-learn {np.mean((yf_te - sk.predict(Xf_te)) ** 2):.3f}")
check(all(b <= a + 1e-9 for a, b in zip(mse, mse[1:])), "training error falls every single round")
mse_mine, mse_sk = np.mean((yf_te - mine) ** 2), np.mean((yf_te - sk.predict(Xf_te)) ** 2)
check(abs(mse_mine - mse_sk) < 0.02 * mse_sk and np.corrcoef(mine, sk.predict(Xf_te))[0, 1] > 0.999,
      "your hand-written booster matches GradientBoostingRegressor (same test error within 2%; tiny differences come from tie-breaking)")

# %% TODO 6: early stopping   (Lesson 4.3)
def early_stopped_models():
    """(1) GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, max_depth=3,
           validation_fraction=0.1, n_iter_no_change=20, random_state=0) on (Xf_tr, yf_tr);
       (2) HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.1, early_stopping=True,
           n_iter_no_change=20, random_state=0) on (Xb_tr, yb_tr).
    Return both fitted models."""
    # TODO 6: build the two estimators exactly as described and .fit them on the data named in the docstring
    raise NotImplementedError("TODO 6: early_stopped_models")


gbr, hgb = early_stopped_models()
print(f"GBR stopped at {gbr.n_estimators_} of 2000 trees (test R² {gbr.score(Xf_te, yf_te):.3f}); "
      f"HGB stopped at {hgb.n_iter_} of 1000 rounds (test accuracy {hgb.score(Xb_te, yb_te):.3f})")
check(gbr.n_estimators_ < 2000 and hgb.n_iter_ < 1000, "both stopped long before their maximum: no guessing the number of rounds")
check(gbr.score(Xf_te, yf_te) > 0.85, "the early-stopped booster explains over 85% of the variance on new data")

# %% TODO 7: blend two models with out-of-fold predictions   (Lesson 4.5)
def oof_blend_weight(model_a, model_b, X, y):
    """Get 5-fold out-of-fold predictions of each model with cross_val_predict(model, X, y, cv=5).
    Try weights w = 0, 0.01, …, 1 for the blend w·A + (1 − w)·B and return the w with the
    lowest mean squared error, plus the OOF predictions of A and B."""
    # TODO 7: pa = cross_val_predict(model_a, X, y, cv=5), same for b; ws = np.linspace(0, 1, 101); pick the w with the smallest mean((w·pa + (1 − w)·pb − y)²)
    raise NotImplementedError("TODO 7: oof_blend_weight")


lin = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())  # a smooth model: degree-2 polynomial regression (Lesson 2.5)
gbm = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=2, random_state=0)
w, pa, pb = oof_blend_weight(lin, gbm, Xf_tr, yf_tr)
lin.fit(Xf_tr, yf_tr); gbm.fit(Xf_tr, yf_tr)
mse_a = np.mean((yf_te - lin.predict(Xf_te)) ** 2)
mse_b = np.mean((yf_te - gbm.predict(Xf_te)) ** 2)
mse_blend = np.mean((yf_te - (w * lin.predict(Xf_te) + (1 - w) * gbm.predict(Xf_te))) ** 2)
print(f"blend weight on the polynomial model: {w:.2f}   test MSE: polynomial {mse_a:.3f}, GBM {mse_b:.3f}, blend {mse_blend:.3f}")
print(f"out-of-fold error correlation between the two: {np.corrcoef(pa - yf_tr, pb - yf_tr)[0, 1]:.2f}")
check(0 < w < 1, "the best blend uses both models")
check(mse_blend < min(mse_a, mse_b), "on unseen test data, the blend beats both of its members")

# %% Optional: the same booster in XGBoost / LightGBM (skipped if not installed)
try:
    import xgboost as xgb

    xm = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=3, early_stopping_rounds=20)
    Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(Xf_tr, yf_tr, test_size=0.1, random_state=0)
    xm.fit(Xs_tr, ys_tr, eval_set=[(Xs_va, ys_va)], verbose=False)
    print(f"XGBoost {xgb.__version__}: best iteration {xm.best_iteration}, test R² {xm.score(Xf_te, yf_te):.3f}")
except Exception as e:  # ImportError, or a missing OpenMP runtime on macOS
    print("XGBoost not available, skipped:", type(e).__name__)
try:
    import lightgbm as lgb

    lm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=15, verbose=-1)
    Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(Xf_tr, yf_tr, test_size=0.1, random_state=0)
    lm.fit(Xs_tr, ys_tr, eval_set=[(Xs_va, ys_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    print(f"LightGBM {lgb.__version__}: best iteration {lm.best_iteration_}, test R² {lm.score(Xf_te, yf_te):.3f}")
except Exception as e:
    print("LightGBM not available, skipped:", type(e).__name__)

# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. In TODO 3, add `min_samples_leaf=10` to the unlimited tree. How close does it get to the best
#    depth's validation score? Then try `ccp_alpha` values from `cost_complexity_pruning_path`.
# 2. In TODO 4, set `max_features=1.0` on the random forest (every feature at every split). It's now
#    plain bagging: does the OOB score move toward the bagging model's?
# 3. In TODO 5, try lr = 1.0 and lr = 0.01 with 100 rounds. Plot the test MSE after every round
#    (`predict` with `trees[:k]`): which one overfits, which one is still underfitting?
# 4. Add 50 columns of random noise to X_f (np.hstack with rng.normal) and refit the GBM and a
#    k-NN regressor. Who suffers more? (Lesson 4.6, Playground 2.)

# %% [markdown]
# # Lab 5: Evaluation & tuning, done honestly 🎯
#
# You'll build the evaluation toolkit of Phase 5 and check every piece against
# scikit-learn: cross-validation that respects groups (and see shuffled CV lie
# on grouped data), a confusion matrix with precision / recall / F1 by hand,
# ROC AUC by its ranking definition, a threshold chosen by the cost of mistakes,
# calibration of an overconfident naive Bayes model, random vs grid search under
# the same budget, and a learning-curve diagnosis.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase5_evaluation.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Runs offline in well under a minute on a laptop CPU.
# Lessons to reread if stuck: 5.1 (cross-validation), 5.3 (confusion matrix,
# thresholds), 5.4 (ROC AUC), 5.6 (calibration), 5.7 (search), 5.8 (curves)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score, learning_curve, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import loguniform

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


# %% Data for TODO 1: 40 patients, 10 recordings each (Lesson 5.1's toy, in NumPy)
def make_patients(n_patients, n_records, spread, seed):
    """Each patient has a centre; their recordings scatter tightly around it. The label is per patient
    and depends only weakly on the features, so recognising the *patient* is much easier than the *disease*."""
    r = np.random.default_rng(seed)
    centres = r.normal(0, 1.5, size=(n_patients, 2))
    sick = (r.random(n_patients) < 1 / (1 + np.exp(-2 * centres[:, 0]))).astype(int)
    X = np.repeat(centres, n_records, axis=0) + r.normal(0, spread, size=(n_patients * n_records, 2))
    y = np.repeat(sick, n_records)
    groups = np.repeat(np.arange(n_patients), n_records)
    return X, y, groups


X_p, y_p, g_p = make_patients(40, 10, spread=0.1, seed=2)
X_new, y_new, _ = make_patients(400, 1, spread=0.1, seed=99)   # 400 brand-new patients: the truth
knn = KNeighborsClassifier(n_neighbors=1)


# %% TODO 1: cross-validation that respects groups   (Lesson 5.1)
def cv_mean(model, X, y, cv, groups=None):
    """Mean accuracy of `model` under the splitter `cv` (pass groups for group splitters)."""
    # TODO 1: one line with cross_val_score(model, X, y, cv=cv, groups=groups), then .mean()
    raise NotImplementedError("TODO 1: cv_mean")


shuffled = cv_mean(knn, X_p, y_p, KFold(5, shuffle=True, random_state=0))
stratified = cv_mean(knn, X_p, y_p, StratifiedKFold(5, shuffle=True, random_state=0))
grouped = cv_mean(knn, X_p, y_p, GroupKFold(n_splits=5), groups=g_p)
truth = knn.fit(X_p, y_p).score(X_new, y_new)
print(f"shuffled KFold {shuffled:.3f} | stratified {stratified:.3f} | GroupKFold {grouped:.3f} | new patients {truth:.3f}")
check(shuffled > truth + 0.10, "shuffled CV is inflated by more than 10 points on grouped data")
check(abs(grouped - truth) < abs(shuffled - truth), "GroupKFold is closer to the new-patient truth")

# %% TODO 2: the confusion matrix and its metrics, by hand   (Lesson 5.3)
def confusion_counts(y_true, y_pred):
    """Return (tp, fp, fn, tn) for 0/1 labels, where 1 is the positive class."""
    # TODO 2a: count with boolean masks, e.g. tp = np.sum((y_true == 1) & (y_pred == 1))
    raise NotImplementedError("TODO 2a: confusion_counts")


def precision_recall_f1(tp, fp, fn):
    """Precision = tp / (tp + fp); recall = tp / (tp + fn); F1 = their harmonic mean."""
    # TODO 2b: three lines (guard against dividing by zero with `if ... else 0.0`)
    raise NotImplementedError("TODO 2b: precision_recall_f1")


check(close(precision_recall_f1(28, 8, 27), (0.778, 0.509, 0.615)), "Lesson 5.3's matrix: precision 0.778, recall 0.509, F1 0.615")
Xb, yb = load_breast_cancer(return_X_y=True)
yb = 1 - yb                                         # make "malignant" the positive class (1)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xb_tr, yb_tr)
pred = lr.predict(Xb_te)
tp, fp, fn, tn = confusion_counts(yb_te, pred)
check((tn, fp, fn, tp) == tuple(confusion_matrix(yb_te, pred).ravel()), "same counts as sklearn's confusion_matrix")
check(close(precision_recall_f1(tp, fp, fn), (precision_score(yb_te, pred), recall_score(yb_te, pred), f1_score(yb_te, pred))),
      "same precision, recall and F1 as sklearn")

# %% TODO 3: ROC AUC by its ranking definition   (Lesson 5.4)
def auc_by_ranking(y_true, scores):
    """Share of (positive, negative) pairs where the positive scores higher; ties count half."""
    # TODO 3: split scores into pos and neg, compare every pair with broadcasting:
    #         wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    raise NotImplementedError("TODO 3: auc_by_ranking")


check(close(auc_by_ranking([1, 1, 1, 0, 0, 0], [0.9, 0.6, 0.4, 0.7, 0.3, 0.2]), 7 / 9), "Lesson 5.4's worked example: 7 of 9 pairs = 0.778")
s_te = lr.predict_proba(Xb_te)[:, 1]
check(close(auc_by_ranking(yb_te, s_te), roc_auc_score(yb_te, s_te), 1e-9), "matches roc_auc_score exactly")
check(close(auc_by_ranking(yb_te, 10 * s_te ** 3), roc_auc_score(yb_te, s_te), 1e-9), "unchanged by any order-preserving transform of the scores")

# %% TODO 4: choose the threshold by the cost of mistakes   (Lesson 5.3)
def best_threshold(y_true, p, cost_fp, cost_fn):
    """Try thresholds 0.01, 0.02, …, 0.99; return (best threshold, its total cost) where
    cost = cost_fp × false alarms + cost_fn × misses, and you flag when p >= threshold."""
    # TODO 4: loop over np.linspace(0.01, 0.99, 99); reuse confusion_counts((p >= t).astype(int))
    raise NotImplementedError("TODO 4: best_threshold")


p_cal = 1 / (1 + np.exp(-(-3 + 2.2 * rng.standard_normal(4000))))   # calibrated probabilities…
y_cal = (rng.random(4000) < p_cal).astype(int)                     # …and outcomes drawn from them
t_best, c_best = best_threshold(y_cal, p_cal, cost_fp=10, cost_fn=100)
_, fp5, fn5, _ = confusion_counts(y_cal, (p_cal >= 0.5).astype(int))
print(f"best threshold {t_best:.2f} costs ${c_best:,.0f}; 0.5 costs ${10 * fp5 + 100 * fn5:,.0f}; formula t* = {10 / 110:.3f}")
check(c_best < 0.75 * (10 * fp5 + 100 * fn5), "the cost-based threshold is at least 25% cheaper than 0.5")
check(abs(t_best - 10 / 110) < 0.06, "for calibrated probabilities it lands near t* = C_FP / (C_FP + C_FN) ≈ 0.09")

# %% TODO 5: calibrate an overconfident naive Bayes   (Lesson 5.6)
Xc, yc = make_classification(n_samples=6000, n_features=20, n_informative=4, n_redundant=10, random_state=0)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.5, random_state=0)
raw = GaussianNB().fit(Xc_tr, yc_tr)


def calibrated_nb(X, y):
    """Return GaussianNB wrapped in CalibratedClassifierCV with isotonic calibration and 5-fold CV, fitted."""
    # TODO 5: CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(X, y)
    raise NotImplementedError("TODO 5: calibrated_nb")


cal = calibrated_nb(Xc_tr, yc_tr)
p_raw, p_iso = raw.predict_proba(Xc_te)[:, 1], cal.predict_proba(Xc_te)[:, 1]
b_raw, b_iso = brier_score_loss(yc_te, p_raw), brier_score_loss(yc_te, p_iso)
print(f"Brier: raw {b_raw:.3f} → isotonic {b_iso:.3f};  AUC: raw {roc_auc_score(yc_te, p_raw):.3f} → {roc_auc_score(yc_te, p_iso):.3f}")
check(b_iso < b_raw - 0.01, "calibration lowers the Brier score by more than 0.01")
check(abs(roc_auc_score(yc_te, p_iso) - roc_auc_score(yc_te, p_raw)) < 0.03, "…while the ranking (AUC) barely moves")

plt.figure(figsize=(4.5, 4.5))
for name, p in [("naive Bayes", p_raw), ("+ isotonic", p_iso)]:
    frac, mean_pred = calibration_curve(yc_te, p, n_bins=10)
    plt.plot(mean_pred, frac, "o-", label=name)
plt.plot([0, 1], [0, 1], "k--", label="perfect"); plt.xlabel("predicted"); plt.ylabel("observed")
plt.title("Reliability diagram"); plt.legend(); plt.tight_layout(); show()

# %% TODO 6: random vs grid search, same budget   (Lesson 5.7)
Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=0)


def grid_and_random(X, y, budget=16):
    """Tune SVC's C and gamma with (a) a 4 × 4 grid and (b) RandomizedSearchCV with n_iter=budget drawn from
    loguniform(1e-2, 1e3) for C and loguniform(1e-5, 1e0) for gamma (cv=3, random_state=0).
    Return the two fitted search objects (grid, random)."""
    # TODO 6: GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": [1e-4, 1e-3, 1e-2, 1e-1]}, cv=3)
    #         and RandomizedSearchCV(SVC(), {...distributions...}, n_iter=budget, cv=3, random_state=0)
    raise NotImplementedError("TODO 6: grid_and_random")


grid, rand = grid_and_random(Xd_tr, yd_tr)
print(f"grid best CV {grid.best_score_:.4f} {grid.best_params_}\nrandom best CV {rand.best_score_:.4f}")
check(len(grid.cv_results_["params"]) == 16 and len(rand.cv_results_["params"]) == 16, "both searches tried exactly 16 settings")
check(rand.best_score_ > grid.best_score_ - 0.01, "random search is as good as the grid (within 1 point) for the same budget")
check(len({round(p["gamma"], 12) for p in rand.cv_results_["params"]}) == 16, "random search tried 16 different gammas; the grid only 4")
print(f"test accuracy of the random-search winner (the one honest look): {rand.score(Xd_te, yd_te):.3f}")

# %% TODO 7: read the learning curves   (Lesson 5.8)
def diagnose(train_score, val_score, target):
    """Scores are accuracies at the largest training size. Return "high variance" if the train–validation
    gap is above 0.05, else "high bias" if validation is below the target, else "fine"."""
    # TODO 7: three if-statements, in that order
    raise NotImplementedError("TODO 7: diagnose")


results = {}
for name, model in [("stump (depth 1)", DecisionTreeClassifier(max_depth=1, random_state=0)),
                    ("deep tree", DecisionTreeClassifier(random_state=0)),
                    ("SVC", SVC(gamma=0.001))]:
    sizes, tr, va = learning_curve(model, Xd, yd, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
    results[name] = diagnose(tr.mean(1)[-1], va.mean(1)[-1], target=0.95)
    print(f"{name:16s} train {tr.mean(1)[-1]:.3f}  validation {va.mean(1)[-1]:.3f}  → {results[name]}")
    plt.plot(sizes, va.mean(1), "o-", label=name + " (validation)")
plt.xlabel("training rows"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout(); show()
check(results["stump (depth 1)"] == "high bias", "a depth-1 tree underfits: high bias")
check(results["deep tree"] == "high variance", "an unpruned tree overfits: high variance")
check(results["SVC"] == "fine", "the SVC clears the 0.95 target with a small gap")

# %% 🔬 Try it yourself
# 1. TODO 1 with spread=1.0 (recordings vary as much within a patient as between): does the leak shrink?
# 2. TODO 4 with cost_fn=500: where does the threshold go? Compare with 10 / 510.
# 3. TODO 5 with method="sigmoid", and with only 300 training rows: when does isotonic lose?
# 4. TODO 6 with budget=9 vs a 3 × 3 grid, over several random_state values.
print("\n🎉 Lab 5 complete: honest folds, the right metric, a cost-based threshold, calibrated probabilities, a fair search and a diagnosis.")

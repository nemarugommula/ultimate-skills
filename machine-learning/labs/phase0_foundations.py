# %% [markdown]
# # Lab 0: Foundations, honestly measured 🌱
#
# You'll build the habits every later lab relies on: split data by hand, measure a
# do-nothing baseline, fit and score a real scikit-learn model, measure two kinds of
# leakage (scaling before the split: tiny; feature selection before the split: huge),
# choose a hyperparameter on a validation set while touching the test set only once,
# and watch "tuning on the test set" invent accuracy out of pure coin flips.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase0_foundations.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 0.1 (generalization), 0.2 (baselines),
# 0.5 (fit / predict / transform), 0.6 (splits and leakage)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


# The dataset for most of this lab: 569 breast tumours, 30 numeric measurements each,
# label 0 = malignant, 1 = benign. It ships with scikit-learn, so nothing is downloaded.
X, y = load_breast_cancer(return_X_y=True)
print(X.shape, "rows × columns;", np.bincount(y), "malignant / benign")


# %% TODO 1: a train/test split by hand   (Lesson 0.6)
def split_by_hand(n, test_frac, seed):
    """Return (train_idx, test_idx): two arrays of row numbers that together cover 0..n-1 once.
    Shuffle the row numbers with np.random.default_rng(seed), then put the first
    round(n * test_frac) of them in the test set and the rest in the training set."""
    # TODO 1: order = a shuffled copy of 0..n-1;  n_test = int(round(n * test_frac))
    # Hint: np.random.default_rng(seed).permutation(n). Test = order[:n_test], train = the rest.
    raise NotImplementedError("TODO 1: split_by_hand")


tr_idx, te_idx = split_by_hand(len(y), 0.25, seed=0)
check(len(tr_idx) == 427 and len(te_idx) == 142, "569 rows → 427 train + 142 test")
check(len(set(tr_idx) & set(te_idx)) == 0, "no row is in both sets")
check(sorted(np.concatenate([tr_idx, te_idx])) == list(range(len(y))), "every row is used exactly once")
check(not np.array_equal(te_idx, np.arange(142)), "the rows were shuffled (not just the first 142)")

# From here on we use scikit-learn's version, which does the same thing (plus stratify=y
# keeps the malignant/benign shares equal in both parts).
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# %% TODO 2: the do-nothing baseline   (Lesson 0.2)
def majority_baseline(y_train, y_test):
    """Accuracy on y_test of always predicting the most common class in y_train."""
    # TODO 2: find the most common label in y_train, then the share of y_test equal to it.
    # Hint: np.bincount(y_train).argmax() is the most common 0/1 label; np.mean(a == b) is a share.
    raise NotImplementedError("TODO 2: majority_baseline")


base = majority_baseline(y_tr, y_te)
print(f"baseline accuracy: {base:.3f}")
check(close(base, DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).score(X_te, y_te)),
      "matches scikit-learn's DummyClassifier")
check(0.6 < base < 0.65, "predicting 'benign' for everyone already scores about 63%")


# %% TODO 3: fit and score a real model   (Lesson 0.5)
def fit_and_score(X_train, y_train, X_test, y_test):
    """Build make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), fit it on the
    training data only, and return (the fitted pipeline, its accuracy on the test data)."""
    # TODO 3: build the pipeline, .fit it on the training data, .score it on the test data.
    # Hint: fit returns the pipeline itself; score returns accuracy for classifiers.
    raise NotImplementedError("TODO 3: fit_and_score")


model, acc = fit_and_score(X_tr, y_tr, X_te, y_te)
print(f"logistic regression: {acc:.3f} (baseline {base:.3f})")
check(acc > base + 0.25, "the model beats the baseline by more than 25 points")
check(hasattr(model[-1], "coef_") and model[-1].coef_.shape == (1, 30), "fitted: one learned weight per column (coef_)")
check(close(model[0].mean_, X_tr.mean(axis=0)), "the scaler's mean_ was learned from the TRAINING rows only")
p = model.predict_proba(X_te[:3])
check(p.shape == (3, 2) and close(p.sum(axis=1), 1), "predict_proba: one column per class, rows sum to 1")


# %% TODO 4: the scaling leak (small)   (Lesson 0.6)
def scaling_leak(X, y, seed):
    """Split with train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed), then
    score KNeighborsClassifier(n_neighbors=5) on the test part in two ways:
      leaky:  StandardScaler fitted on ALL of X (before the split), then used on both parts
      honest: StandardScaler fitted on the training part only
    Return (leaky_accuracy, honest_accuracy)."""
    # TODO 4: split; leaky = scaler.fit(X) (all rows); honest = scaler.fit(X_train_part).
    # In both cases: knn.fit(scaler.transform(train part), y) then knn.score(scaler.transform(test part), y).
    raise NotImplementedError("TODO 4: scaling_leak")


gaps = []
for seed in range(10):
    leaky, honest = scaling_leak(X, y, seed)
    gaps.append(leaky - honest)
print("leaky − honest, 10 splits:", np.round(gaps, 3))
check(abs(np.mean(gaps)) < 0.02, "scaling before the split barely changes the score here (a tiny leak)")
check(honest > 0.9, "and the honest k-NN is still above 90%")


# %% TODO 5: the feature-selection leak (huge)   (Lesson 0.6)
rng = np.random.default_rng(0)
X_noise = rng.normal(size=(200, 5000))     # 200 rows × 5,000 columns of pure noise
y_noise = rng.integers(0, 2, size=200)     # random labels: nothing can beat 50% honestly


def selection_leak(X, y, k=20):
    """Return (leaky, honest) mean 5-fold cross-validated accuracies of LogisticRegression():
      leaky:  SelectKBest(f_classif, k=k) fitted on ALL rows first, then cross_val_score on the result
      honest: make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression()) passed to cross_val_score,
              so the selection is redone inside every training fold."""
    # TODO 5: leaky: X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y), then
    #         cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean()
    # honest: cross_val_score(make_pipeline(SelectKBest(...), LogisticRegression()), X, y, cv=5).mean()
    raise NotImplementedError("TODO 5: selection_leak")


leaky, honest = selection_leak(X_noise, y_noise)
print(f"pure noise: leaky {leaky:.2f}, honest {honest:.2f}")
check(leaky > 0.7, "selecting columns on all rows makes pure noise look predictive (> 70%)")
check(honest < 0.62, "inside a Pipeline the honest score is near 50%")


# %% TODO 6: choose k on a validation set, touch the test set once   (Lesson 0.6)
def choose_k(X_train, y_train, X_test, y_test, ks, seed=0):
    """1. Split the TRAINING data again: train_test_split(X_train, y_train, test_size=0.25,
          stratify=y_train, random_state=seed) → (fit part, validation part).
       2. For each k in ks, fit make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
          on the fit part and score it on the validation part.
       3. Take the k with the best validation score (first one wins ties), refit that pipeline on
          ALL of X_train, and score it once on the test data.
    Return (best_k, dict of validation scores {k: score}, test_score)."""
    # TODO 6: split X_train into a fit part and a validation part; loop over ks filling a dict
    # {k: validation score}; pick best_k = the k with the highest score; refit on all of X_train;
    # score once on X_test. Hint: max(ks, key=lambda k: val[k]) returns the first best k.
    raise NotImplementedError("TODO 6: choose_k")


ks = [1, 3, 5, 7, 9, 15, 25, 51]
best_k, val_scores, test_score = choose_k(X_tr, y_tr, X_te, y_te, ks)
print("validation scores:", {k: round(v, 3) for k, v in val_scores.items()})
print(f"chosen k = {best_k}, test accuracy (measured once) = {test_score:.3f}")
check(best_k in ks and set(val_scores) == set(ks), "every k was scored on the validation set")
check(val_scores[best_k] == max(val_scores.values()), "the chosen k has the best validation score")
check(test_score > 0.9, "the chosen model scores above 90% on the untouched test set")

plt.figure(figsize=(5, 3))
plt.plot(ks, [val_scores[k] for k in ks], "o-", label="validation accuracy")
plt.axhline(test_score, color="teal", ls="--", label=f"test, chosen k = {best_k}")
plt.xscale("log"); plt.xlabel("k (neighbours)"); plt.legend(); plt.title("Choose on validation, report on test")
plt.tight_layout(); show()


# %% TODO 7: fool yourself with the test set   (Lesson 0.6, Playground 1)
def peek(n_models, n_test, seed=0):
    """Labels are coin flips. Each 'model' guesses at random. Score n_models random guessers
    on the same n_test-row test set, keep the best, and also score that winner's guesses on
    a FRESH n_test-row set it was never chosen on.
    Use rng = np.random.default_rng(seed); y_test = rng.integers(0, 2, n_test);
    each model's guesses = rng.integers(0, 2, n_test). For the fresh set, draw new labels and
    new guesses the same way (a random guesser stays random).
    Return (best_test_accuracy, fresh_accuracy)."""
    # TODO 7: rng = np.random.default_rng(seed); y_test = rng.integers(0, 2, n_test)
    # best = the max over n_models of np.mean(rng.integers(0, 2, n_test) == y_test)
    # fresh = np.mean(rng.integers(0, 2, n_test) == rng.integers(0, 2, n_test))
    raise NotImplementedError("TODO 7: peek")


bests = [peek(100, 100, seed)[0] for seed in range(50)]
fresh = [peek(100, 100, seed)[1] for seed in range(50)]
print(f"best of 100 coin flips on 100 rows: {np.mean(bests):.3f} on average; on fresh data: {np.mean(fresh):.3f}")
check(np.mean(bests) > 0.6, "picking the best of 100 random models 'finds' over 60% accuracy")
check(abs(np.mean(fresh) - 0.5) < 0.03, "on fresh data the winner is back to about 50%")
check(np.mean([peek(100, 2000, s)[0] for s in range(10)]) < 0.55, "a 2,000-row test set wobbles less, so the inflation shrinks")

# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. In TODO 5, change k from 20 to 2 and to 200. Which leaks more, and why? What happens to
#    the leaky score with 500 columns instead of 5,000?
# 2. In TODO 6, report the best *validation* score as if it were the final result. How far is it
#    from the test score? Repeat with 5 different seeds: which number is more stable?
# 3. Add a "target-leaking" column to the breast-cancer data: X_leak = np.column_stack([X, y + noise])
#    with noise = 0.1 * rng.normal(size=len(y)). Refit TODO 3's model. What accuracy do you get, and
#    why would it collapse in a real clinic?

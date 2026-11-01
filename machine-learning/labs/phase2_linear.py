# %% [markdown]
# # Lab 2: Linear models, from least squares to XOR 📈
#
# You'll solve least squares with NumPy and check it against scikit-learn,
# compute R² by hand, watch ridge shrink and lasso zero out the weights of the
# diabetes dataset, choose a decision threshold for a breast-cancer classifier
# and read its odds ratios, compare softmax with one-vs-rest on iris, and
# rescue logistic regression on XOR with one interaction feature.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase2_linear.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 2.1 (least squares, R²), 2.2 (ridge, lasso),
# 2.3 (logistic regression, thresholds), 2.4 (softmax vs one-vs-rest),
# 2.5 (polynomial features, XOR)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


X_dia, y_dia = load_diabetes(return_X_y=True)          # 442 patients, 10 features (pre-scaled), disease progression
print(X_dia.shape, y_dia[:3])


# %% TODO 1: least squares with NumPy   (Lesson 2.1)
def least_squares(X, y):
    """Return (w, b): the weights and intercept that minimize the sum of squared residuals.
    Add a column of ones to X for the intercept and solve with np.linalg.lstsq."""
    # TODO 1: A = X with an extra column of ones (np.column_stack), coef = np.linalg.lstsq(A, y, rcond=None)[0].
    # Hint: the last entry of coef multiplies the column of ones, so it's the intercept.
    raise NotImplementedError("TODO 1: least_squares")


w, b = least_squares(X_dia, y_dia)
sk = LinearRegression().fit(X_dia, y_dia)
check(close(w, sk.coef_) and close(b, sk.intercept_), "NumPy's lstsq gives the same weights and intercept as LinearRegression")
check(close(b, 152.133), "intercept = 152.13 = the mean of y (the features are centred)")

# The 4-point worked example from Lesson 2.1
w4, b4 = least_squares(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([2.0, 3.0, 5.0, 6.0]))
check(close(w4, [1.4]) and close(b4, 0.5), "worked example: slope 1.4, intercept 0.5")


# %% TODO 2: R² by hand   (Lesson 2.1)
def r2_by_hand(y, y_pred):
    """R² = 1 − SSE / SST, where SST is the squared spread around the mean of y."""
    # TODO 2: sse = sum of (y − y_pred)², sst = sum of (y − mean(y))², return 1 − sse/sst.
    raise NotImplementedError("TODO 2: r2_by_hand")


check(close(r2_by_hand(np.array([2.0, 3, 5, 6]), np.array([1.9, 3.3, 4.7, 6.1])), 0.98), "worked example: R² = 1 − 0.2/10 = 0.98")
X_tr, X_te, y_tr, y_te = train_test_split(X_dia, y_dia, test_size=0.25, random_state=0)
lin = LinearRegression().fit(X_tr, y_tr)
r2 = r2_by_hand(y_te, lin.predict(X_te))
check(close(r2, lin.score(X_te, y_te)), f"your R² matches scikit-learn's .score(): {r2:.3f} on the test set")
check(close(r2_by_hand(y_te, np.full(len(y_te), y_te.mean())), 0.0), "predicting the mean gives R² = 0")


# %% TODO 3: ridge shrinks every weight, smoothly   (Lesson 2.2)
def ridge_norms(X, y, alphas):
    """For each alpha, fit Ridge(alpha) on standardized X and return the L2 norm ‖w‖ of its weights."""
    # TODO 3: Z = StandardScaler().fit_transform(X); for each a: Ridge(alpha=a).fit(Z, y).coef_ → np.linalg.norm(...).
    raise NotImplementedError("TODO 3: ridge_norms")


alphas = [0.01, 1, 10, 100, 1000, 10000]
norms = ridge_norms(X_dia, y_dia, alphas)
print("‖w‖ by alpha:", [round(v, 1) for v in norms])
check(all(a > b for a, b in zip(norms, norms[1:])), "a bigger ridge penalty always gives smaller weights")
check(norms[-1] < 0.1 * norms[0], "with alpha = 10,000 the weights are shrunk by more than 90%")


# %% TODO 4: the lasso path and its zeros   (Lesson 2.2)
def lasso_zeros(X, y, alphas):
    """For each alpha, fit Lasso(alpha, max_iter=10_000) on standardized X; return how many weights are exactly 0."""
    # TODO 4: like TODO 3 with Lasso(alpha=a, max_iter=10_000); count the weights equal to 0: np.sum(coef_ == 0).
    raise NotImplementedError("TODO 4: lasso_zeros")


lasso_alphas = [0.01, 0.1, 1, 3, 10, 30, 100]
zeros = lasso_zeros(X_dia, y_dia, lasso_alphas)
print("zeros by alpha:", dict(zip(lasso_alphas, zeros)))
check(zeros[0] <= 1, "a tiny lasso penalty keeps (almost) every feature")
check(all(a <= b for a, b in zip(zeros, zeros[1:])), "more penalty → more weights exactly zero (on this path)")
check(zeros[-1] == 10, "alpha = 100 switches every feature off: the model predicts the mean")
ridge_zero = int(np.sum(Ridge(alpha=1000).fit(StandardScaler().fit_transform(X_dia), y_dia).coef_ == 0))
check(ridge_zero == 0, "ridge never makes exact zeros, even with alpha = 1000")

coefs = np.array([Lasso(alpha=a, max_iter=10_000).fit(StandardScaler().fit_transform(X_dia), y_dia).coef_ for a in np.logspace(-2, 2, 40)])
plt.figure(figsize=(6, 4))
plt.semilogx(np.logspace(-2, 2, 40), coefs)
plt.xlabel("alpha (lasso penalty)"); plt.ylabel("weight (standardized)"); plt.title("The lasso path: weights hit zero one by one")
plt.tight_layout(); show()


# %% TODO 5: logistic regression and the threshold   (Lesson 2.3)
X_bc, y_bc = load_breast_cancer(return_X_y=True)
y_bc = 1 - y_bc                                        # 1 = malignant: the class we must not miss
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_bc, y_bc, test_size=0.25, stratify=y_bc, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xb_tr, yb_tr)
p_mal = clf.predict_proba(Xb_te)[:, 1]


def precision_recall_at(y_true, p, threshold):
    """Flag every case with p >= threshold. Return (precision, recall) for the positive class.
    precision = TP / (TP + FP), recall = TP / (TP + FN)."""
    # TODO 5: flag = p >= threshold; count TP (flag & y==1), FP (flag & y==0), FN (~flag & y==1).
    raise NotImplementedError("TODO 5: precision_recall_at")


p5, r5 = precision_recall_at(yb_te, p_mal, 0.5)
p2, r2_ = precision_recall_at(yb_te, p_mal, 0.2)
print(f"threshold 0.5: precision {p5:.3f} recall {r5:.3f}   threshold 0.2: precision {p2:.3f} recall {r2_:.3f}")
check(close(p5, 1.0) and close(r5, 0.962), "threshold 0.5: precision 1.000, recall 0.962")
check(r2_ > r5 and p2 < p5, "lowering the threshold to 0.2 raises recall and lowers precision")


# %% TODO 6: coefficients as odds ratios   (Lesson 2.3)
def odds_ratios(pipeline):
    """Return e^w for every weight of the pipeline's last step (a fitted LogisticRegression)."""
    # TODO 6: pipeline[-1] is the LogisticRegression; its weights are .coef_[0]. Return np.exp of them.
    raise NotImplementedError("TODO 6: odds_ratios")


names = load_breast_cancer().feature_names
orat = odds_ratios(clf)
top = names[np.argmax(np.abs(np.log(orat)))]
print("largest effect:", top, f"odds × {orat.max():.2f} per +1 standard deviation")
check(orat.shape == (30,) and np.all(orat > 0), "30 odds ratios, all positive (e to any power is positive)")
check(top == "radius error" and close(orat.max(), 3.46, tol=0.01), "biggest effect: radius error, odds × 3.46 per +1 SD")
check(close(np.log(orat), clf[-1].coef_[0]), "log(odds ratio) is the weight itself")


# %% TODO 7: softmax vs one-vs-rest on iris   (Lesson 2.4)
X_ir, y_ir = load_iris(return_X_y=True)


def softmax_and_ovr():
    """Return two unfitted pipelines (StandardScaler + model): softmax = LogisticRegression(max_iter=1000)
    (multinomial for 3+ classes), and ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))."""
    # TODO 7: two make_pipeline(StandardScaler(), ...) objects. Don't fit them: cross_val_score does that.
    raise NotImplementedError("TODO 7: softmax_and_ovr")


softmax, ovr = softmax_and_ovr()
acc_s = cross_val_score(softmax, X_ir, y_ir, cv=5).mean()
acc_o = cross_val_score(ovr, X_ir, y_ir, cv=5).mean()
print(f"5-fold CV accuracy: softmax {acc_s:.3f}, one-vs-rest {acc_o:.3f}")
check(close(acc_s, 0.960) and close(acc_o, 0.927), "softmax 0.960 beats one-vs-rest 0.927")
proba = softmax.fit(X_ir, y_ir).predict_proba(X_ir)
check(close(proba.sum(axis=1), 1.0), "softmax probabilities sum to 1 for every flower")


# %% TODO 8: XOR needs one extra feature   (Lesson 2.5)
X_xor = rng.normal(size=(400, 2))
y_xor = (X_xor[:, 0] * X_xor[:, 1] > 0).astype(int)    # same sign → class 1


def xor_models():
    """Return (linear, crossed): LogisticRegression() on the raw features, and a pipeline
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) → StandardScaler → LogisticRegression()."""
    # TODO 8: interaction_only=True adds x1·x2 without squares; include_bias=False because the model has an intercept.
    raise NotImplementedError("TODO 8: xor_models")


linear, crossed = xor_models()
acc_lin = cross_val_score(linear, X_xor, y_xor, cv=5).mean()
acc_x = cross_val_score(crossed, X_xor, y_xor, cv=5).mean()
print(f"XOR, 5-fold CV accuracy: linear {acc_lin:.3f}, with x1·x2 {acc_x:.3f}")
check(acc_lin < 0.6, "a straight line can't do XOR: about a coin flip")
check(acc_x > 0.9, "one interaction feature x1·x2 makes it easy (> 90%)")
check(PolynomialFeatures(degree=3).fit(np.zeros((1, 100))).n_output_features_ == 176851, "100 features at degree 3 explode into 176,851 columns")

print("\n🎉 Lab 2 complete: least squares, R², ridge, lasso, thresholds, odds ratios, softmax and feature crosses.")

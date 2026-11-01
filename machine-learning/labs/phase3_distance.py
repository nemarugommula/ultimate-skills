# %% [markdown]
# # Lab 3: Distance & probability models 📏
#
# You'll build k-nearest neighbours by hand with NumPy and check it against
# scikit-learn, watch scaling rescue kNN on the wine dataset, measure the curse
# of dimensionality (nearest ÷ farthest distance → 1), write a word-count naive
# Bayes spam filter by hand and match MultinomialNB, show that an SVM depends
# only on its support vectors, verify the degree-2 kernel identity, and see an
# RBF SVM separate two rings that a straight line can't.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase3_distance.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 3.1 (kNN, scaling, curse of dimensionality),
# 3.2 (naive Bayes, smoothing), 3.3 (SVMs, support vectors, C), 3.4 (kernels)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, make_blobs, make_circles
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


# %% TODO 1: k-nearest neighbours by hand   (Lesson 3.1)
def knn_predict(X_train, y_train, X_query, k):
    """Predict a label for every row of X_query by majority vote of its k nearest
    training rows (Euclidean distance). Ties go to the smallest label."""
    # TODO 1: (a) a distance matrix d of shape (n_query, n_train): subtract with broadcasting,
    #   X_query[:, None, :] - X_train[None, :, :], square, sum over the last axis, square root.
    # (b) nearest = np.argsort(d, axis=1, kind="stable")[:, :k]   (c) votes = y_train[nearest]
    # (d) for each row of votes, np.bincount(row).argmax() is the majority label (ties → smallest).
    raise NotImplementedError("TODO 1: knn_predict")

X_wine, y_wine = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X_wine, y_wine, test_size=0.3, stratify=y_wine, random_state=0)
sc = StandardScaler().fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
mine = knn_predict(Ztr, ytr, Zte, k=5)
sk = KNeighborsClassifier(n_neighbors=5).fit(Ztr, ytr).predict(Zte)
check(mine.shape == yte.shape, "one prediction per query row")
check((mine == sk).mean() == 1.0, "your kNN agrees with KNeighborsClassifier on every test wine")
check(knn_predict(Ztr, ytr, Ztr, k=1).tolist() == ytr.tolist(), "k = 1 on the training set is always 100% right (each point is its own neighbour)")
print(f"test accuracy (k = 5, scaled): {(mine == yte).mean():.3f}")

# %% TODO 2: scaling decides what “near” means   (Lessons 1.4 and 3.1)
def raw_vs_scaled(X, y, k=5):
    """Return (raw, scaled): 5-fold cross-validated accuracy of kNN on raw features,
    and of a Pipeline StandardScaler → kNN (so each fold is scaled with its own training stats)."""
    # TODO 2: cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean() for raw,
    # and the same with make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)) for scaled.
    raise NotImplementedError("TODO 2: raw_vs_scaled")

raw, scaled = raw_vs_scaled(X_wine, y_wine)
print(f"raw {raw:.3f}  →  scaled {scaled:.3f}   (feature std ranges from {X_wine.std(0).min():.2f} to {X_wine.std(0).max():.0f})")
check(close(raw, 0.691, 0.01) and close(scaled, 0.949, 0.01), "wine: kNN goes from 69% to 95% just by scaling")

# %% TODO 3: the curse of dimensionality   (Lesson 3.1)
def distance_ratio(d, n=500, n_queries=20, seed=0):
    """Scatter n uniform random points in the d-dimensional unit cube. For n_queries random
    query points, compute (nearest distance) / (farthest distance); return the average."""
    # TODO 3: r = np.random.default_rng(seed); P = r.random((n, d)); Q = r.random((n_queries, d)).
    # Distance matrix as in TODO 1 (shape n_queries × n), then the mean of dist.min(axis=1) / dist.max(axis=1).
    # Return it as a float.
    raise NotImplementedError("TODO 3: distance_ratio")

dims = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
ratios = [distance_ratio(d) for d in dims]
for d, q in zip(dims, ratios):
    print(f"d = {d:5d}: nearest / farthest = {q:.3f}")
check(ratios[1] < 0.1, "in 2-D the nearest point is more than 10× closer than the farthest")
check(ratios[-1] > 0.8, "in 1,000-D the nearest point is over 80% as far as the farthest")
check(all(a < b for a, b in zip(ratios[1:], ratios[2:])), "the ratio keeps rising with the dimension")
plt.figure(figsize=(5, 3.5))
plt.semilogx(dims, ratios, "o-")
plt.xlabel("dimensions d"); plt.ylabel("nearest ÷ farthest"); plt.title("Distances concentrate"); plt.tight_layout(); show()

# %% TODO 4: a word-count naive Bayes spam filter by hand   (Lesson 3.2)
SPAM = ["win a free prize now", "free money click here", "claim your free gift card now", "you won a cash prize click to claim",
        "cheap meds online buy now", "urgent your account wins a prize", "click here for free money now", "limited offer buy cheap watches",
        "free entry win cash now", "congratulations you won click the link", "earn money from home free", "exclusive deal click now free"]
HAM = ["are we still meeting for lunch", "can you send the report today", "the meeting moved to friday", "thanks for the notes from the call",
       "lunch at noon works for me", "here is the draft of the report", "can we move our call to monday", "please review the slides before the meeting",
       "see you at the team lunch", "the project report is attached", "free for a call this afternoon", "happy birthday hope you have a great day"]


def train_word_nb(spam, ham, alpha=1.0):
    """Return (vocab, word_score) where word_score[w] = log P(w | spam) − log P(w | ham),
    with P(w | class) = (count(w, class) + alpha) / (total words in class + alpha * |vocab|)."""
    # TODO 4a: sp = " ".join(spam).split() and hm = the same for ham (all words, with repeats).
    # vocab = sorted(set(sp) | set(hm)); V = len(vocab).
    # For each word w: log((sp.count(w) + alpha) / (len(sp) + alpha * V)) minus the same for hm.
    raise NotImplementedError("TODO 4a: train_word_nb")

def nb_log_odds(message, word_score, prior_log_odds=0.0):
    """Base-rate score + the score of every known word in the message (repeats count again)."""
    # TODO 4b: start from prior_log_odds and add word_score[w] for every w in message.split() that is in word_score.
    raise NotImplementedError("TODO 4b: nb_log_odds")

vocab, score = train_word_nb(SPAM, HAM)
msg = "click now to claim your free prize"
lo = nb_log_odds(msg, score)
p_spam = 1 / (1 + np.exp(-lo))
print(f"|vocab| = {len(vocab)}, score(free) = {score['free']:.2f}, log-odds = {lo:.2f}, P(spam) = {p_spam:.5f}")
check(len(vocab) == 76 and close(score["free"], 1.45, 0.01), "“free” is 4.3× likelier in spam: score +1.45 (Lesson 3.2's table)")
check(close(lo, 8.78, 0.01), "the lesson's worked example: total log-odds +8.78")
cv_ = CountVectorizer(token_pattern=r"[a-z0-9£$€']+")
mnb = MultinomialNB(alpha=1.0).fit(cv_.fit_transform(SPAM + HAM), [1] * 12 + [0] * 12)
check(close(mnb.predict_proba(cv_.transform([msg]))[0, 1], p_spam, 1e-6), "your filter matches scikit-learn's MultinomialNB exactly")
_, score0 = train_word_nb(SPAM, HAM, alpha=1e-12)
check(score0["meeting"] < -20, "with (almost) no smoothing, a word never seen in spam gets a near −∞ score: one word vetoes the message")

# %% TODO 5: an SVM only needs its support vectors   (Lesson 3.3)
X_b, y_b = make_blobs(n_samples=60, centers=[[-2, -1.5], [2, 1.5]], cluster_std=0.9, random_state=3)


def refit_on_support_vectors(X, y, C=1000.0):
    """Fit a linear SVC; refit a second one on its support vectors only.
    Return (n_support_vectors, coef_full, coef_sv_only) where coef = [w1, w2, b]."""
    # TODO 5: full = SVC(kernel="linear", C=C).fit(X, y); sv = full.support_ (indices of support vectors).
    # small = the same model fitted on X[sv], y[sv]. coef of a model m: np.r_[m.coef_[0], m.intercept_[0]].
    raise NotImplementedError("TODO 5: refit_on_support_vectors")

n_sv, c_full, c_sv = refit_on_support_vectors(X_b, y_b)
print(f"{n_sv} support vectors of {len(X_b)};  w, b (all points) = {np.round(c_full, 3)};  (support vectors only) = {np.round(c_sv, 3)}")
check(n_sv < 10, "only a handful of the 60 points are support vectors")
check(close(c_full, c_sv, 1e-3), "retraining on the support vectors alone gives the same line")
X_ov, y_ov = make_blobs(n_samples=200, centers=[[-1, -0.7], [1, 0.7]], cluster_std=1.0, random_state=11)
n_by_C = {C: int(SVC(kernel="linear", C=C).fit(X_ov, y_ov).n_support_.sum()) for C in [0.01, 1, 100]}
print("support vectors by C on overlapping data:", n_by_C)
check(n_by_C[0.01] > n_by_C[1] >= n_by_C[100] - 2, "smaller C → wider street → more support vectors")

# %% TODO 6: the kernel trick, checked by hand   (Lesson 3.4)
def phi2(v):
    """Degree-2 features of a 2-D point (x1, x2): (x1², √2·x1·x2, x2²)."""
    # TODO 6: unpack x1, x2 = v and return np.array([x1², √2·x1·x2, x2²]) (np.sqrt(2)).
    raise NotImplementedError("TODO 6: phi2")

x, y = np.array([1.0, 2.0]), np.array([3.0, 1.0])
check(close(phi2(x) @ phi2(y), (x @ y) ** 2) and close((x @ y) ** 2, 25), "Lesson 3.4's example: φ(x)·φ(y) = (x·y)² = 25")
A, B = rng.normal(size=(50, 2)), rng.normal(size=(40, 2))
explicit = np.array([[phi2(a) @ phi2(b) for b in B] for a in A])
check(close(explicit, (A @ B.T) ** 2), "the identity holds for 2,000 random pairs")
check(close(explicit, polynomial_kernel(A, B, degree=2, gamma=1, coef0=0)), "…and equals scikit-learn's polynomial_kernel(degree=2, gamma=1, coef0=0)")

# %% TODO 7: an RBF SVM separates the rings   (Lesson 3.4)
X_c, y_c = make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=0)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.5, stratify=y_c, random_state=0)


def ring_scores(X_tr, y_tr, X_te, y_te):
    """Return {name: (train_acc, test_acc)} for three scaled SVCs:
    'linear' (kernel='linear'), 'rbf' (kernel='rbf', default gamma) and 'rbf_huge_gamma' (gamma=1000)."""
    # TODO 7: for each of SVC(kernel="linear"), SVC(kernel="rbf"), SVC(kernel="rbf", gamma=1000):
    # pipe = make_pipeline(StandardScaler(), model).fit(X_tr, y_tr); store (pipe.score(X_tr, y_tr), pipe.score(X_te, y_te)).
    raise NotImplementedError("TODO 7: ring_scores")

res = ring_scores(Xc_tr, yc_tr, Xc_te, yc_te)
for name, (a, b) in res.items():
    print(f"{name:15s} train {a:.3f}  test {b:.3f}")
check(res["linear"][1] < 0.7, "a straight line can't separate the rings")
check(res["rbf"][1] > 0.95, "the RBF kernel separates them (> 95% on test)")
check(res["rbf_huge_gamma"][0] == 1.0 and res["rbf_huge_gamma"][1] < res["rbf"][1] - 0.1, "huge gamma memorizes: 100% train, much worse test")

# See it: the RBF boundary
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(Xc_tr, yc_tr)
gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
zz = pipe.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
plt.figure(figsize=(4.5, 4.5))
plt.contourf(gx, gy, zz, levels=[-10, 0, 10], alpha=0.25, cmap="coolwarm")
plt.contour(gx, gy, zz, levels=[0], colors="k")
plt.scatter(X_c[:, 0], X_c[:, 1], c=y_c, cmap="coolwarm", s=10)
plt.title("RBF SVM: a circle from a linear method"); plt.tight_layout(); show()

# %% [markdown]
# ## 🔬 Try it yourself
# - TODO 1: add 10 columns of pure noise (`rng.normal(size=(len(X), 10))`) to the scaled wine data. How much does kNN lose? Why does a tree-based model care less?
# - TODO 3: replace uniform points with points on a 2-D plane embedded in 100-D (`P = rng.random((n, 2)) @ rng.normal(size=(2, 100))`). Does the ratio still approach 1? (Intrinsic dimension!)
# - TODO 4: add “meeting” to a spammy message. How many spammy words does it take to win back the vote?
# - TODO 7: grid-search C and gamma with `GridSearchCV` and plot test accuracy as a heat map.

print("\n🎉 Lab 3 complete: kNN, naive Bayes, SVMs and kernels, from scratch and in scikit-learn.")

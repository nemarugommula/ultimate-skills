# %% [markdown]
# # Lab 6: Unsupervised learning, by hand and with scikit-learn 🔮 (SOLUTION)
#
# You'll write one k-means round yourself and check it against scikit-learn,
# pick k with the silhouette score, watch k-means, single linkage and DBSCAN
# compete on two moons, choose the number of Gaussian-mixture components with
# BIC, compute PCA with a plain SVD on the digits and match scikit-learn's
# explained variance, and catch injected outliers with an isolation forest
# measured by precision at an alert budget.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase6_unsupervised.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 6.1 (k-means, silhouette), 6.2 (linkage),
# 6.3 (DBSCAN), 6.4 (GMM, BIC), 6.5 (PCA), 6.7 (isolation forest)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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


# %% TODO 1: one round of k-means by hand   (Lesson 6.1)
def kmeans_round(X, centers):
    """One assign + update step. Returns (labels, new_centers, inertia), where labels and
    inertia come from assigning X to the *given* centers, and new_centers are the means."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # n × k squared distances
    labels = d2.argmin(axis=1)
    inertia = d2[np.arange(len(X)), labels].sum()
    new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                            for k in range(len(centers))])
    return labels, new_centers, inertia


# Lesson 6.1's worked example: 1, 2, 4, 9, 10, 13 with starting centres 1 and 2
X1 = np.array([[1.0], [2], [4], [9], [10], [13]])
lab, c, inert = kmeans_round(X1, np.array([[1.0], [2.0]]))
check(inert == 238 and close(c.ravel(), [1.0, 7.6]), "first round: inertia 238, centres move to 1 and 7.6")
lab, c, inert = kmeans_round(X1, c)
lab, c, inert = kmeans_round(X1, c)
check(close(c.ravel(), [7 / 3, 32 / 3]) and close(inert, 13.333), "two more rounds: centres 2.33 and 10.67, inertia 13.33")

# Now on real blobs: iterate your round until nothing changes, compare with scikit-learn from the same start
Xb, yb = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=0)
start = Xb[[0, 1, 2, 3]].copy()
centers = start.copy()
for _ in range(100):
    lab, new, inert = kmeans_round(Xb, centers)
    if np.allclose(new, centers):
        break
    centers = new
sk = KMeans(n_clusters=4, init=start, n_init=1, algorithm="lloyd").fit(Xb)
check(abs(inert - sk.inertia_) < 1e-3 * sk.inertia_, f"your loop matches scikit-learn's inertia ({inert:.1f})")

# %% TODO 2: choose k with the silhouette score   (Lesson 6.1)
def best_k_by_silhouette(X, ks):
    """Fit KMeans(n_clusters=k, n_init=10, random_state=0) for each k and return
    (best_k, scores) where scores[k] = silhouette_score(X, labels)."""
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return max(scores, key=scores.get), scores


k_best, sil = best_k_by_silhouette(Xb, range(2, 9))
print({k: round(v, 3) for k, v in sil.items()})
check(k_best == 4, "the silhouette peaks at k = 4 on 4 blobs")
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xb).inertia_ for k in range(1, 9)]
check(all(a > b for a, b in zip(inertias, inertias[1:])), "inertia always falls as k grows, so it can't pick k by itself")

plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1); plt.plot(range(1, 9), inertias, "o-"); plt.title("Elbow: inertia vs k"); plt.xlabel("k")
plt.subplot(1, 2, 2); plt.plot(list(sil), list(sil.values()), "o-", color="teal"); plt.title("Silhouette vs k"); plt.xlabel("k")
plt.tight_layout(); show()

# %% TODO 3: k-means vs single linkage vs DBSCAN on two moons   (Lessons 6.2 and 6.3)
Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
Xm = StandardScaler().fit_transform(Xm)


def three_clusterings(X):
    """Return a dict of label arrays: 'kmeans' (KMeans k=2, n_init=10, random_state=0),
    'single' (AgglomerativeClustering, 2 clusters, single linkage) and
    'dbscan' (DBSCAN eps=0.3, min_samples=5)."""
    return {
        "kmeans": KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X),
        "single": AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X),
        "dbscan": DBSCAN(eps=0.3, min_samples=5).fit_predict(X),
    }


labs = three_clusterings(Xm)
ari = {name: adjusted_rand_score(ym, l) for name, l in labs.items()}
print({k: round(v, 3) for k, v in ari.items()})
check(ari["kmeans"] < 0.6, "k-means cuts the moons with a straight line (ARI well below 1)")
check(ari["single"] > 0.99 and ari["dbscan"] > 0.99, "single linkage and DBSCAN both trace the moons (ARI ≈ 1)")
check(labs["dbscan"].max() + 1 == 2, "DBSCAN found 2 clusters without being told k")

# %% TODO 4: choose the number of Gaussian-mixture components with BIC   (Lesson 6.4)
Xg, yg = make_blobs(n_samples=600, centers=3, random_state=170)
Xg = Xg @ np.array([[0.6, -0.64], [-0.41, 0.85]])        # stretch and tilt: k-means' worst case


def bic_curve(X, ks, covariance_type="full"):
    """Return {k: BIC} for GaussianMixture(n_components=k, covariance_type=..., n_init=3, random_state=0)."""
    return {k: GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=3, random_state=0).fit(X).bic(X)
            for k in ks}


bics = bic_curve(Xg, range(1, 7))
check(min(bics, key=bics.get) == 3, "BIC picks 3 components for the 3 stretched blobs")
gm = GaussianMixture(n_components=3, n_init=3, random_state=0).fit(Xg)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xg)
ari_gm, ari_km = adjusted_rand_score(yg, gm.predict(Xg)), adjusted_rand_score(yg, km.labels_)
print(f"ARI: GMM {ari_gm:.3f}, k-means {ari_km:.3f}")
check(ari_gm > ari_km + 0.2, "the full-covariance GMM follows the stretched clusters far better than k-means")
check(close(gm.predict_proba(Xg).sum(axis=1), 1.0), "responsibilities sum to 1 for every point")

# %% TODO 5: PCA with a plain SVD   (Lesson 6.5)
Xd, yd = load_digits(return_X_y=True)


def pca_svd(X, k):
    """Centre X, take the SVD, and return (components, explained_variance_ratio) for the first k
    components. components has shape (k, n_features); ratios use ALL singular values in the total."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return Vt[:k], (var / var.sum())[:k]


comps, ratio = pca_svd(Xd, 10)
sk = PCA(n_components=10).fit(Xd)
check(close(ratio, sk.explained_variance_ratio_, 1e-6), "your explained variance ratios match scikit-learn's")
check(close(np.abs((comps * sk.components_).sum(axis=1)), 1.0, 1e-6), "your components match scikit-learn's (up to sign)")
_, all_ratio = pca_svd(Xd, 64)
n95 = int(np.searchsorted(np.cumsum(all_ratio), 0.95) + 1)
check(n95 == 29, f"29 of 64 components keep 95% of the digits' variance (first two: {all_ratio[:2].sum():.1%})")

Z = sk.transform(Xd)
plt.figure(figsize=(5, 4.5)); plt.scatter(Z[:, 0], Z[:, 1], c=yd, cmap="tab10", s=6)
plt.colorbar(label="digit"); plt.title("Digits on PC1 and PC2"); plt.tight_layout(); show()

# %% TODO 6: isolation forest at an alert budget   (Lesson 6.7)
normal = rng.normal(0, 1, size=(1000, 2)) * [1.5, 0.8]
outliers = rng.uniform(-7, 7, size=(40, 2))
outliers = outliers[np.abs(outliers).max(axis=1) > 4][:15]   # keep 15 points clearly outside the cloud
Xa = np.vstack([normal, outliers])
ya = np.r_[np.zeros(len(normal)), np.ones(len(outliers))]


def precision_at_k(scores, y, k):
    """Share of real anomalies (y == 1) among the k highest scores."""
    return y[np.argsort(-scores)[:k]].mean()


iso = IsolationForest(n_estimators=100, random_state=0).fit(Xa)
scores = -iso.score_samples(Xa)            # scikit-learn: higher score_samples = more normal, so negate
p15 = precision_at_k(scores, ya, 15)
print(f"precision@15 = {p15:.2f}")
check(p15 >= 0.8, "an isolation forest puts at least 12 of the 15 injected outliers in its top 15")
random_p = precision_at_k(rng.random(len(ya)), ya, 15)
check(p15 > random_p + 0.5, "far better than random alerts")
flags = IsolationForest(contamination=0.05, random_state=0).fit_predict(Xa)
check(abs((flags == -1).sum() - 0.05 * len(Xa)) <= 1, "contamination=0.05 flags about 5% of the points: it only moves the threshold")

print("\n🎉 Lab 6 complete: you ran, chose and checked every main unsupervised tool.")

# %% [markdown]
# # Lab 5: Statistics from scratch 📊
#
# (SOLUTION file: every TODO is filled in. Try labs/phase5_statistics.py first!)
#
# You'll write the numbers a data scientist (or an ML engineer reading an eval) uses every day:
# mean, median, std and z-scores, correlation, maximum likelihood for a coin and a bell curve,
# an A/B test with a 95% confidence interval (and why peeking breaks it), overfitting with a
# train/test split, k-means clustering, and precision / recall / F1 at different thresholds.
#
# Setup (once, from the math-course folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase5_statistics.py`.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 5.1 (describing data), 5.2 (correlation), 5.2b (k-means), 5.3 (MLE),
# 5.5 (A/B tests), 5.6 (overfitting), 5.6b (precision, recall, F1)

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)  # one seeded random generator → reproducible results


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def close(a, b, tol=1e-6):
    """True when every number in a is within tol of the matching number in b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) < tol))


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: describe a pile of numbers (Lesson 5.1)
def mean(x):
    """x̄: add everything up, divide by how many."""
    # TODO 1a: return sum(x) / len(x)
    return sum(x) / len(x)


def median(x):
    """The middle value once sorted (the average of the two middle ones when there's an even count)."""
    s = sorted(x)
    n = len(s)
    mid = n // 2  # // divides and rounds down: 5 // 2 = 2, the index of the middle of 5 values
    # TODO 1b:
    #   if n is odd (n % 2 == 1): return s[mid]
    #   otherwise:                return (s[mid − 1] + s[mid]) / 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def std(x):
    """s: the square root of the average squared distance from the mean (÷ n, as in Lesson 5.1)."""
    # TODO 1c:
    #   m = mean(x)
    #   squares = [(v − m) ** 2 for v in x]
    #   return np.sqrt(mean(squares))
    m = mean(x)
    squares = [(v - m) ** 2 for v in x]
    return np.sqrt(mean(squares))


waits = [2, 3, 3, 4, 8]  # minutes waited for coffee
check(mean(waits) == 4 and median(waits) == 3, "wait times: mean 4, median 3 (Lesson 5.1's worked example)")
check(close(std(waits), np.sqrt(4.4)) and close(std(waits), np.std(waits)), "std = √4.4 ≈ 2.10, same as np.std")
friends = [40, 45, 50, 50, 55, 60, 60, 65, 70]  # salaries in $1000s
check(mean(friends) == 55 and median(friends) == 55, "nine friends: mean 55, median 55")
with_billionaire = friends + [1_000_000]
check(mean(with_billionaire) == 100049.5 and median(with_billionaire) == 57.5,
      "a billionaire walks in: the mean jumps to 100,049.5, the median barely moves to 57.5")


# %% See it: z-scores and the slow tail of an API (the Lesson 5.1 hook)
def z_scores(x):
    """How many standard deviations each value is from the mean."""
    return (np.array(x) - mean(x)) / std(x)


check(close(z_scores(waits)[-1], 4 / np.sqrt(4.4)), "the 8-minute wait is (8 − 4) / 2.10 ≈ 1.9 std above the mean")

# 1,000 API response times: most are fast, 5% hit a slow path.
latency = np.where(rng.random(1000) < 0.95, rng.normal(120, 10, 1000), rng.normal(330, 40, 1000))
print(f"latency: mean {mean(latency):.0f} ms, median {median(latency):.0f} ms, std {std(latency):.0f} ms, "
      f"99th percentile {np.percentile(latency, 99):.0f} ms")
z = z_scores(latency)
print(f"{np.sum(z > 3)} requests are more than 3 std above the mean: the 'sometimes hangs' complaints")
check(mean(latency) > median(latency) + 5, "a slow tail drags the mean above the median: report both (and p99)")


# %% TODO 2: correlation r by hand (Lesson 5.2)
def correlation(x, y):
    """Pearson's r: the covariance divided by both standard deviations."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # TODO:
    #   cov = mean((x − mean(x)) * (y − mean(y)))     (average product of the deviations)
    #   return cov / (std(x) * std(y))
    cov = mean((x - mean(x)) * (y - mean(y)))
    return cov / (std(x) * std(y))


hours = [1, 2, 3, 4, 5]
scores = [52, 55, 61, 64, 68]
check(close(correlation(hours, scores), 8.2 / (np.sqrt(2) * np.sqrt(34))), "study hours vs score: r = 8.2 / (1.414 × 5.831) ≈ 0.994")
check(close(correlation(hours, scores), np.corrcoef(hours, scores)[0, 1]), "matches np.corrcoef")
minutes = [60 * h for h in hours]
check(close(correlation(minutes, scores), correlation(hours, scores)), "hours → minutes doesn't change r (no units)")
check(close(correlation(hours, [-s for s in scores]), -correlation(hours, scores)), "flip y and r flips sign")


# %% See it: Anscombe's quartet, four datasets with the same r ≈ 0.816 (Lesson 5.2)
ax_ = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
anscombe = {
    "I": (ax_, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (ax_, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (ax_, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for axis, (name, (xs, ys)) in zip(axes, anscombe.items()):
    r = correlation(xs, ys)
    axis.scatter(xs, ys)
    axis.set_title(f"{name}: r = {r:.3f}")
    check(abs(r - 0.816) < 0.001, f"Anscombe {name}: r = 0.816")
fig.suptitle("Same r, totally different stories: always plot your data"); plt.tight_layout(); show()


# %% TODO 3: maximum likelihood for a coin (Lesson 5.3)
def coin_log_likelihood(p, heads, tails):
    """ℓ(p) = log P(these flips | heads-chance p) = heads · ln p + tails · ln(1 − p)."""
    # TODO: return heads * np.log(p) + tails * np.log(1 − p)
    return heads * np.log(p) + tails * np.log(1 - p)


check(close(coin_log_likelihood(0.7, 7, 3), -6.108643), "7 heads, 3 tails at p = 0.7: ℓ = 7 ln 0.7 + 3 ln 0.3 ≈ −6.11")
check(close(coin_log_likelihood(0.5, 7, 3), 10 * np.log(0.5)), "at p = 0.5: ℓ = 10 ln 0.5 ≈ −6.93 (less likely)")
grid = np.linspace(0.01, 0.99, 99)  # candidate p values 0.01, 0.02, ..., 0.99
best = grid[np.argmax([coin_log_likelihood(p, 7, 3) for p in grid])]
check(close(best, 0.7), "the maximum-likelihood p̂ is 7/10 = 0.7: just the fraction of heads")


# %% See it: more data, same peak, sharper curve
plt.figure(figsize=(7, 3.5))
for h, t in [(7, 3), (70, 30)]:
    ll = np.array([coin_log_likelihood(p, h, t) for p in grid])
    plt.plot(grid, np.exp(ll - ll.max()), label=f"{h} heads, {t} tails")  # scaled so the peak is 1
plt.axvline(0.7, color="gray", ls="--")
plt.xlabel("guess p"); plt.ylabel("likelihood (peak = 1)"); plt.title("Maximum likelihood: p̂ = 0.7 both times")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); show()
# Flip the sign and ℓ becomes the NLL loss. An LLM's training loss is exactly this, averaged per token.


# %% TODO 4: maximum likelihood for a bell curve (Lesson 5.3)
def gaussian_log_likelihood(x, mu, sigma):
    """ℓ(μ, σ): add up the log of the bell-curve height N(x | μ, σ²) at every data point."""
    x = np.array(x, dtype=float)
    # TODO: for one point, log N(x | μ, σ²) = −½ · ln(2πσ²) − (x − μ)² / (2σ²). Add them all up:
    #   return np.sum(−0.5 * np.log(2 * np.pi * sigma ** 2) − (x − mu) ** 2 / (2 * sigma ** 2))
    return np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2))


check(close(gaussian_log_likelihood([0.0], 0.0, 1.0), -0.5 * np.log(2 * np.pi)), "one point at the centre of N(0, 1): ℓ = −½ ln 2π ≈ −0.919")
check(close(gaussian_log_likelihood(waits, 4.0, np.sqrt(4.4)), -2.5 * np.log(2 * np.pi * 4.4) - 2.5),
      "the wait times at μ = 4, σ = √4.4: ℓ = −2.5 ln(2π · 4.4) − 22 / 8.8 ≈ −10.80")


# %% See it: the best bell curve is centred on the mean, with the ÷ n standard deviation
mus = np.arange(2.0, 6.0, 0.01)
best_mu = mus[np.argmax([gaussian_log_likelihood(waits, m, 2.0) for m in mus])]
sigmas = np.arange(1.0, 4.0, 0.01)
best_sigma = sigmas[np.argmax([gaussian_log_likelihood(waits, best_mu, s) for s in sigmas])]
print(f"best μ = {best_mu:.2f} (mean = {mean(waits)}), best σ = {best_sigma:.2f} (std ÷ n = {std(waits):.2f})")
check(abs(best_mu - 4.0) < 0.006 and abs(best_sigma - 2.10) < 0.006, "MLE of a Gaussian = the mean and the ÷ n std")
# Why regression minimizes squared error: maximizing a Gaussian likelihood IS minimizing Σ (x − μ)².


# %% TODO 5: an A/B test with a 95% confidence interval (Lesson 5.5)
def ab_test(conv_a, n_a, conv_b, n_b):
    """Return (B − A difference in conversion rate, its standard error, (low, high) of the 95% CI)."""
    # TODO:
    #   pa = conv_a / n_a ;  pb = conv_b / n_b                    (the two measured rates)
    #   se_a = np.sqrt(pa * (1 − pa) / n_a) ;  se_b = the same for B
    #   se_diff = np.sqrt(se_a ** 2 + se_b ** 2)                  (variances add)
    #   diff = pb − pa
    #   return diff, se_diff, (diff − 1.96 * se_diff, diff + 1.96 * se_diff)
    pa = conv_a / n_a
    pb = conv_b / n_b
    se_a = np.sqrt(pa * (1 - pa) / n_a)
    se_b = np.sqrt(pb * (1 - pb) / n_b)
    se_diff = np.sqrt(se_a ** 2 + se_b ** 2)
    diff = pb - pa
    return diff, se_diff, (diff - 1.96 * se_diff, diff + 1.96 * se_diff)


diff, se, (lo, hi) = ab_test(100, 1000, 120, 1000)
check(close(diff, 0.02) and close(se, 0.0139857), "10% vs 12% on 1,000 each: gap 0.02, SE 0.0140 (Lesson 5.5)")
check(close([lo, hi], [-0.0074120, 0.0474120]), "95% CI [−0.7, +4.7] points: it includes 0, so not significant")
diff, se, (lo, hi) = ab_test(1000, 10000, 1200, 10000)
check(close([lo, hi], [0.0113316, 0.0286684]), "same rates on 10,000 each: CI [+1.1, +2.9] points, now it excludes 0")


# %% See it: A/A tests and the danger of peeking
# 1,000 fake experiments where A and B are IDENTICAL (both 10%). Any "significant" result is a false positive.
n_experiments, n_visitors = 1000, 2000
a_clicks = (rng.random((n_experiments, n_visitors)) < 0.10).cumsum(axis=1)  # running totals of sign-ups
b_clicks = (rng.random((n_experiments, n_visitors)) < 0.10).cumsum(axis=1)


def significant(conv_a, conv_b, n):
    diff, se, (lo, hi) = ab_test(conv_a, n, conv_b, n)
    return lo > 0 or hi < 0  # the 95% CI excludes 0


honest = 0  # look once, at the end
peeker = 0  # look after every 100 visitors and stop the first time it looks significant
for e in range(n_experiments):
    if significant(a_clicks[e, -1], b_clicks[e, -1], n_visitors):
        honest += 1
    for n in range(100, n_visitors + 1, 100):
        if significant(a_clicks[e, n - 1], b_clicks[e, n - 1], n):
            peeker += 1
            break
print(f"false positives, one look at the end:    {honest / n_experiments:.1%}  (the promised 5%)")
print(f"false positives, peeking 20 times:       {peeker / n_experiments:.1%}")
check(0.03 < honest / n_experiments < 0.07, "one look: about 5% false positives, as promised")
check(peeker > 2 * honest, "peeking and stopping early more than doubles the false positives")


# %% TODO 6: overfitting, measured with a train/test split (Lesson 5.6)
def train_and_test_error(x_train, y_train, x_test, y_test, degree):
    """Fit a polynomial of this degree on the training points only. Return (training MSE, test MSE)."""
    coefs = np.polyfit(x_train, y_train, degree)  # least-squares fit, done for you by NumPy
    # TODO: predict with np.polyval(coefs, x), then take the mean squared error on each set:
    #   train_mse = np.mean((y_train − np.polyval(coefs, x_train)) ** 2)
    #   test_mse = np.mean((y_test − np.polyval(coefs, x_test)) ** 2)
    #   return train_mse, test_mse
    train_mse = np.mean((np.array(y_train) - np.polyval(coefs, x_train)) ** 2)
    test_mse = np.mean((np.array(y_test) - np.polyval(coefs, x_test)) ** 2)
    return train_mse, test_mse


x_tr, y_tr = [1, 2, 3, 4], [2.5, 3.5, 6.5, 7.5]  # Lesson 5.6's four training points
x_te, y_te = [1.5, 2.5, 3.5], [3, 5, 7]  # three fresh points
check(close(train_and_test_error(x_tr, y_tr, x_te, y_te, 1), [0.20, 0.08 / 3]), "the line: train MSE 0.20, test MSE 0.027")
train_mse, test_mse = train_and_test_error(x_tr, y_tr, x_te, y_te, 3)
check(train_mse < 1e-12 and close(test_mse, 0.5 / 3), "the cubic: train MSE 0 (looks perfect!), test MSE 0.17, six times worse")


# %% See it: the U-curve. Training error only falls; test error falls, then shoots up
hidden = lambda x: np.sin(2 * np.pi * x)  # the true pattern behind the data
x_train = np.sort(rng.random(12))
y_train = hidden(x_train) + 0.2 * rng.standard_normal(12)
x_test = rng.random(100)
y_test = hidden(x_test) + 0.2 * rng.standard_normal(100)
degrees = list(range(10))
errors = [train_and_test_error(x_train, y_train, x_test, y_test, d) for d in degrees]
for d, (tr, te) in zip(degrees, errors):
    print(f"degree {d}: train MSE {tr:.3f}   test MSE {te:.3f}")
best_degree = degrees[int(np.argmin([te for tr, te in errors]))]
plt.figure(figsize=(7, 3.5))
plt.semilogy(degrees, [tr for tr, te in errors], "o-", label="training error")
plt.semilogy(degrees, [te for tr, te in errors], "o-", label="test error")
plt.xlabel("polynomial degree (knobs − 1)"); plt.ylabel("MSE (log scale)")
plt.title(f"12 training points: the sweet spot is degree {best_degree}"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); show()
check(all(errors[d + 1][0] <= errors[d][0] + 1e-9 for d in range(9)), "training error never goes up as the model grows")
check(2 <= best_degree <= 7 and errors[9][1] > 3 * errors[best_degree][1], "test error bottoms out, then degree 9 overfits")


# %% TODO 7: k-means from scratch (Lesson 5.2b)
def assign(X, centres):
    """For every point (a row of X), the index of its nearest centre."""
    labels = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        # TODO 7a: squared distance from point i to every centre, then pick the closest.
        #   d2 = np.sum((centres − X[i]) ** 2, axis=1)     (one number per centre)
        #   labels[i] = np.argmin(d2)
        d2 = np.sum((centres - X[i]) ** 2, axis=1)
        labels[i] = np.argmin(d2)
    return labels


def update(X, labels, k):
    """Move each centre to the average of the points assigned to it."""
    centres = np.zeros((k, X.shape[1]))
    for j in range(k):
        # TODO 7b: centres[j] = X[labels == j].mean(axis=0)
        # Hint: labels == j is a True/False array, so X[labels == j] keeps only group j's points.
        centres[j] = X[labels == j].mean(axis=0)
    return centres


def kmeans_loss(X, centres, labels):
    """J: the total squared distance from every point to its own centre."""
    return np.sum((X - centres[labels]) ** 2)


ages = np.array([[18], [20], [22], [60], [64], [68]], dtype=float)  # the café's six regulars (one column)
centres = np.array([[18.0], [30.0]])  # first guess
labels = assign(ages, centres)
check(list(labels) == [0, 0, 0, 1, 1, 1], "assign: 18, 20, 22 join centre 18; 60, 64, 68 join centre 30")
check(kmeans_loss(ages, centres, labels) == 3520, "loss with the first guess: 0 + 4 + 16 + 900 + 1156 + 1444 = 3520")
centres = update(ages, labels, 2)
check(close(centres, [[20], [64]]), "update: the centres move to the group averages, 20 and 64")
check(kmeans_loss(ages, centres, assign(ages, centres)) == 40, "loss after one update: 40")


# %% See it: k-means on three blobs. The loss never goes up
true_centres = np.array([[0, 0], [5, 1], [2, 5]])
X = np.vstack([c + rng.standard_normal((50, 2)) for c in true_centres])  # 150 unlabelled points
centres = X[rng.choice(len(X), size=3, replace=False)]  # start at 3 random data points
losses = []
for step in range(10):
    labels = assign(X, centres)
    losses.append(kmeans_loss(X, centres, labels))
    centres = update(X, labels, 3)
print("loss per step:", [round(float(L), 1) for L in losses])
check(all(losses[i + 1] <= losses[i] + 1e-9 for i in range(len(losses) - 1)), "the k-means loss never rises")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15)
ax1.scatter(centres[:, 0], centres[:, 1], c="red", marker="x", s=150)
ax1.set_title("k-means found 3 groups, no labels needed")
ax2.plot(losses, "o-"); ax2.set_xlabel("step"); ax2.set_ylabel("loss J"); ax2.set_title("never goes up")
plt.tight_layout(); show()
# FAISS uses exactly this loop (on pieces of vectors) to compress a billion embeddings (Lesson 5.2b).


# %% TODO 8: precision, recall and F1 (Lesson 5.6b)
def confusion(y_true, scores, threshold):
    """Flag everything with score ≥ threshold. Return the four boxes: (TP, FP, FN, TN)."""
    y_true = np.array(y_true)
    flagged = np.array(scores) >= threshold
    # TODO 8a: count each box (& = "and", ~ = "not"):
    #   tp = np.sum(flagged & (y_true == 1))      flagged and really positive
    #   fp = np.sum(flagged & (y_true == 0))      flagged but really negative (false alarm)
    #   fn = np.sum(~flagged & (y_true == 1))     missed
    #   tn = np.sum(~flagged & (y_true == 0))
    #   return tp, fp, fn, tn
    tp = np.sum(flagged & (y_true == 1))
    fp = np.sum(flagged & (y_true == 0))
    fn = np.sum(~flagged & (y_true == 1))
    tn = np.sum(~flagged & (y_true == 0))
    return tp, fp, fn, tn


def precision_recall_f1(tp, fp, fn):
    # TODO 8b:
    #   precision = tp / (tp + fp)        can I trust an alarm?
    #   recall = tp / (tp + fn)           how much did I catch?
    #   f1 = 2 * precision * recall / (precision + recall)     (the harmonic mean)
    #   return precision, recall, f1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


y = [1, 1, 1, 0, 0, 0, 0, 0]  # 3 real positives, 5 negatives
s = [0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.4, 0.05]  # the model's scores
check(confusion(y, s, 0.5) == (2, 1, 1, 4), "threshold 0.5: TP 2, FP 1, FN 1, TN 4")
check(confusion(y, s, 0.25) == (3, 2, 0, 3), "threshold 0.25: catch all 3 positives, but 2 false alarms")
check(close(precision_recall_f1(8, 4, 2), [2 / 3, 0.8, 16 / 22]), "Lesson 5.6b's spam filter: precision 0.67, recall 0.80, F1 0.73")


# %% See it: the threshold trade-off on rare positives (like fraud)
n = 10_000
y_true = (rng.random(n) < 0.02).astype(int)  # 2% positives
model_scores = np.where(y_true == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n))
print("lazy model that flags nothing: accuracy", f"{np.mean(y_true == 0):.1%}", "(and recall 0)")
print("threshold  precision  recall   F1     accuracy")
recalls = []
for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    tp, fp, fn, tn = confusion(y_true, model_scores, t)
    P, R, F = precision_recall_f1(tp, fp, fn)
    recalls.append(R)
    print(f"{t:9}  {P:9.3f}  {R:6.3f}  {F:.3f}  {(tp + tn) / n:8.1%}")
check(all(recalls[i + 1] <= recalls[i] for i in range(len(recalls) - 1)), "raising the threshold can only lower recall")
# Accuracy is high at every threshold and tells you almost nothing here. Pick the threshold from
# the cost of each mistake: spam filters lean to precision, medical screening to recall.


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. In the A/A cell, peek after every 10 visitors instead of 100. How high do false positives go?
# 2. Change the U-curve cell to 100 training points. Where is the sweet spot now, and why?
# 3. k-means with k = 2 and k = 5 on the three blobs: how does the final loss change? Why can't the
#    loss alone tell you the "right" k?
# 4. Bayesian twist (Lesson 5.4): with a Beta(1, 1) prior, the posterior mean after 7 heads and 3
#    tails is (7 + 1) / (10 + 2). Compare it with the MLE 0.7 for 2 flips, 10 flips, 1,000 flips.

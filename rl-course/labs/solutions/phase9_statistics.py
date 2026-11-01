# %% [markdown]
# # Lab 9 — SOLUTION: Statistics for RL research
# Try the exercise file first!
#
# You'll build the tools from Lesson 9.3 and use them to answer real research questions:
#   1. bootstrap confidence intervals      2. IQM (interquartile mean)
#   3. the unbiased pass@k estimator (8.8) 4. a power analysis: "how many seeds do I need?"
# numpy + matplotlib only; runs in seconds.

# %%
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# %% TODO 1 — bootstrap confidence interval
def bootstrap_ci(scores, stat=np.mean, B=10_000, level=0.95, seed=0):
    """Resample `scores` with replacement B times, apply `stat`, return the (low, high) percentiles."""
    r = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    idx = r.integers(0, len(scores), size=(B, len(scores)))
    boots = np.array([stat(scores[i]) for i in idx])
    alpha = (1 - level) / 2
    return np.percentile(boots, 100 * alpha), np.percentile(boots, 100 * (1 - alpha))


lo, hi = bootstrap_ci([5.0] * 10)
check(lo == hi == 5.0, "constant scores → zero-width interval")
lo, hi = bootstrap_ci(rng.normal(100, 15, size=400))
check(lo < 100 < hi and 1.5 < hi - lo < 4.5, "400 samples of N(100, 15): CI contains 100 and is about ±1.96·15/√400 wide")


# %% TODO 2 — IQM: mean of the middle 50% of runs
def iqm(scores):
    """Sort, drop the bottom 25% and top 25%, average the rest."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return s[lo:hi].mean()


check(iqm([1, 2, 3, 4, 5, 6, 7, 8]) == 4.5, "IQM of 1..8 = mean of 3,4,5,6 = 4.5")
check(abs(iqm([100, 101, 99, 100, -1000, 100, 102, 98]) - 100) < 1.5, "one diverged seed barely moves the IQM")


# %% TODO 3 — unbiased pass@k (Chen et al., 2021; Lesson 8.8)
def pass_at_k(n, c, k):
    """n samples, c correct. Return 1 - C(n-c, k) / C(n, k), computed stably."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


check(abs(pass_at_k(10, 1, 1) - 0.1) < 1e-12, "pass@1 = c/n")
check(pass_at_k(10, 0, 5) == 0.0 and pass_at_k(10, 6, 5) == 1.0, "no correct → 0; too few wrong to fill k → 1")
check(abs(pass_at_k(100, 10, 10) - (1 - np.prod([(90 - i) / (100 - i) for i in range(10)]))) < 1e-12, "matches the combinatorial definition")


# %% TODO 4 — power: how often does an experiment with n seeds detect a real effect?
def detection_rate(effect, std, n_seeds, trials=2000, seed=0):
    """Simulate `trials` experiments: A ~ N(0, std), B ~ N(effect, std), n_seeds each.
    'Detected' = mean(B) - mean(A) > 1.96 * standard error of the difference."""
    r = np.random.default_rng(seed)
    A = r.normal(0, std, size=(trials, n_seeds))
    B = r.normal(effect, std, size=(trials, n_seeds))
    se = np.sqrt(A.var(axis=1, ddof=1) / n_seeds + B.var(axis=1, ddof=1) / n_seeds)
    return np.mean((B.mean(axis=1) - A.mean(axis=1)) > 1.96 * se)


check(detection_rate(0, 15, 10) < 0.06, "no real effect → 'detected' only ~2.5% of the time (false positives)")
check(detection_rate(5, 15, 3) < detection_rate(5, 15, 30), "more seeds → more power")


# %% Experiment 1 — Henderson et al.'s seed split, with synthetic DQN-like runs
# Same algorithm, 10 seeds. Like Lab 4, some seeds fail badly (heavy-tailed results).
def dqn_like_runs(n, r, p_fail=0.25):
    ok = r.random(n) > p_fail
    return np.where(ok, r.normal(420, 40, n), r.normal(130, 60, n))


fooled = 0
for t in range(1000):
    runs = dqn_like_runs(10, np.random.default_rng(t))
    g1, g2 = runs[:5], runs[5:]
    se = np.sqrt(g1.var(ddof=1) / 5 + g2.var(ddof=1) / 5)
    fooled += abs(g1.mean() - g2.mean()) > 1.96 * se
print(f"Experiment 1: identical algorithm, split 5 vs 5 seeds → 'significant' difference in {fooled / 10:.1f}% of splits")
print("  A test at the 95% level should be fooled only 5% of the time. It's fooled more because 1.96 assumes")
print("  many seeds (with 5 per group the right cutoff is ≈ 2.3), and the failed seeds make the results heavy-tailed.")

# %% Experiment 2 — mean vs median vs IQM: which estimate is most stable across repeated 10-seed studies?
spread = {}
for p_fail in [0.05, 0.25]:
    est = {"mean": [], "median": [], "IQM": []}
    for t in range(2000):
        runs = dqn_like_runs(10, np.random.default_rng(10_000 + t), p_fail)
        est["mean"].append(runs.mean()); est["median"].append(np.median(runs)); est["IQM"].append(iqm(runs))
    spread[p_fail] = {k: np.std(v) for k, v in est.items()}
    print(f"Experiment 2 ({p_fail:.0%} of seeds fail): " + ", ".join(f"{k} {np.mean(v):5.1f} ± {np.std(v):4.1f}" for k, v in est.items()))
check(spread[0.05]["IQM"] < 0.7 * spread[0.05]["mean"], "rare failures (outliers): IQM is much steadier than the mean")
print("  With 25% failures, failing is part of the algorithm's behavior, not an outlier: no single number")
print("  summarizes it well. Report the failure rate (or plot all seeds) as well as an aggregate.")

# %% Experiment 3 — how many seeds to detect an effect? (power curves)
seeds = [2, 3, 5, 10, 20, 30, 50]
plt.figure(figsize=(7, 4))
for effect in [2.5, 5, 10, 15]:
    plt.plot(seeds, [detection_rate(effect, 15, n) for n in seeds], marker="o", label=f"true effect {effect} (std 15)")
plt.axhline(0.8, ls="--", c="gray", label="80% power (common target)")
plt.xscale("log"); plt.xticks(seeds, seeds); plt.ylim(0, 1)
plt.xlabel("seeds per algorithm"); plt.ylabel("chance of detecting the effect")
plt.title("Power: small effects need many seeds"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()
n80 = next(n for n in range(2, 200) if detection_rate(5, 15, n) >= 0.8)
print(f"Experiment 3: to detect a 5-point effect (std 15) 80% of the time you need ≈ {n80} seeds per algorithm")

# %% Experiment 4 — report YOUR Lab 4 ablation properly.
# Replace these placeholder lists with the per-seed evaluation scores your Lab 4 printed.
# PLACEHOLDER — replace with your Lab 4 numbers
lab4 = {"DQN": [109.0, 125.0, 141.0], "Double DQN": [291.0, 412.0, 500.0], "no target net": [9.4, 9.3, 9.5]}
for name, s in lab4.items():
    lo, hi = bootstrap_ci(s)
    print(f"Experiment 4: {name:14s} mean {np.mean(s):6.1f}  IQM {iqm(s):6.1f}  95% bootstrap CI [{lo:6.1f}, {hi:6.1f}]  (n={len(s)})")
print("  With only 3 seeds, bootstrap CIs are optimistic (too narrow). Treat them as rough, and add seeds for claims.")

# %% Experiment 5 — pass@k from samples, e.g. Lab 8 before/after (replace with real counts).
# The SAME 64 problems before and after (comparing different problem sets would measure the problems, not RL).
# p = each problem's true chance of a correct sample. Simulated RL "sharpens": problems the model can already
# solve (p > 0.2) get more reliable (p ** 0.3), while hard ones, where correct samples are rare, get starved (p * 0.2).
# For each problem: number of correct answers out of n=16 samples.
p = rng.beta(0.6, 0.9, size=64)
before = rng.binomial(16, p)
after = rng.binomial(16, np.where(p > 0.2, p ** 0.3, p * 0.2))
pk = {}
for k in [1, 4, 16]:
    pb = np.mean([pass_at_k(16, c, k) for c in before]); pa = np.mean([pass_at_k(16, c, k) for c in after])
    pk[k] = (pb, pa)
    print(f"Experiment 5: pass@{k:<2d} before {pb:.2f}  after {pa:.2f}")
check(pk[1][1] > pk[1][0] and pk[16][1] < pk[16][0],
      "sharpening: pass@1 goes up while pass@16 goes down (coverage shrinks, Lesson 8.8)")

# %% [markdown]
# ## Research questions
# 1. In Experiment 1, what fraction of splits fooled the naive test? How does it change if you pass p_fail=0.05 instead of the default 25%?
# 2. Experiment 3: how many seeds would you need for a 2.5-point effect? What does that imply for 1-seed LLM papers?
# 3. Replace Experiment 4's placeholders with your real Lab 4 numbers, run 5+ seeds, and write a 3-sentence result with CIs.

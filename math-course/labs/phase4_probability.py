# %% [markdown]
# # Lab 4: Probability by simulation 🎲
#
# You'll compute chances two ways, with a formula and by simulating thousands of random
# worlds, and watch them agree: the birthday problem, a medical test (Bayes by counting),
# how an LLM picks its next token (sampling, temperature, top-p), expectation and variance,
# Monte Carlo π, the bell curve appearing out of dice (CLT), and importance sampling,
# the reweighting trick inside PPO.
#
# Setup (once, from the math-course folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase4_probability.py`.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 4.1 (counting chances), 4.2 (Bayes), 4.4 (expectation, variance),
# 4.6 (sampling, temperature, Monte Carlo), 4.6b (importance sampling), 4.7 (LLN, CLT)

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


# %% TODO 1: the birthday problem, by formula (Lesson 4.1)
def birthday_formula(n):
    """P(at least two of n people share a birthday), with 365 equally likely days."""
    p_all_different = 1.0
    for i in range(n):
        # TODO: person number i + 1 must avoid the i birthdays already taken: a chance of (365 − i) / 365.
        #   p_all_different = p_all_different * (365 − i) / 365        ("and" → multiply)
        raise NotImplementedError("TODO 1: birthday_formula")
    return 1 - p_all_different  # "at least one shared" = 1 − "nobody shares"


check(close(birthday_formula(2), 1 / 365), "2 people: 1/365 ≈ 0.0027")
check(abs(birthday_formula(23) - 0.5073) < 1e-4, "23 people: 50.7%, already more likely than not!")
check(birthday_formula(57) > 0.99, "57 people: over 99%")


# %% See it: simulate thousands of rooms and count
def room_has_shared_birthday(n):
    birthdays = rng.integers(0, 365, size=n)  # n random days, numbered 0..364
    return len(np.unique(birthdays)) < n  # fewer different days than people → someone repeats


for n in [10, 23, 50]:
    simulated = np.mean([room_has_shared_birthday(n) for _ in range(5000)])
    print(f"{n:3} people: formula {birthday_formula(n):.3f}   simulated over 5,000 rooms {simulated:.3f}")
    check(abs(simulated - birthday_formula(n)) < 0.03, f"simulation agrees with the formula for {n} people")
# The same "and → multiply" rule: an agent that gets each of 100 steps right 99% of the time...
print(f"...finishes a 100-step task without a mistake only {0.99 ** 100:.1%} of the time.")


# %% TODO 2: conditional probability by counting (Lesson 4.2)
def prob_given(a, b):
    """P(A | B) from two True/False arrays: among the cases where B is True, the fraction where A is True too."""
    # TODO: keep only the B cases, then count:
    #   both = np.sum(a & b)          (A and B both happened; & means "and" for True/False arrays)
    #   return both / np.sum(b)       (... out of all the B cases)
    raise NotImplementedError("TODO 2: prob_given")


sick = np.array([True, True, False, False, False])
positive = np.array([True, False, True, True, False])
check(close(prob_given(sick, positive), 1 / 3), "3 people test positive and 1 of them is sick: P(sick | +) = 1/3")
check(close(prob_given(positive, sick), 1 / 2), "P(+ | sick) = 1/2 is a different number: don't flip the bar!")


# %% See it: the medical test from Lesson 4.2, on a simulated population of one million
n = 1_000_000
sick = rng.random(n) < 0.01  # 1 in 100 is sick
caught = rng.random(n) < 0.90  # the test catches 90% of sick people...
false_alarm = rng.random(n) < 0.09  # ...and wrongly flags 9% of healthy people
positive = np.where(sick, caught, false_alarm)  # sick people use 'caught', healthy ones 'false_alarm'

bayes = 0.90 * 0.01 / (0.90 * 0.01 + 0.09 * 0.99)  # Bayes' rule: P(+|sick)P(sick) / P(+)
print(f"P(+ | sick)  = {prob_given(positive, sick):.3f}  (the test's sensitivity)")
print(f"P(sick | +)  = {prob_given(sick, positive):.4f}  by counting   vs {bayes:.4f} by Bayes' rule")
check(abs(prob_given(sick, positive) - bayes) < 0.005, "counting a simulated population matches Bayes' rule (about 9%)")
# Fraud alerts, spam filters and anomaly detectors all face this: rare targets → most alarms are false.


# %% TODO 3: sample a categorical with one uniform number (Lesson 4.6)
def sample_categorical(p, u):
    """Turn one uniform number u in [0, 1) into a choice: lay the probabilities end to end, see where u lands."""
    edge = 0.0
    for i in range(len(p)):
        # TODO: option i covers the piece [edge, edge + p[i]).
        #   edge = edge + p[i]
        #   if u < edge: return i
        raise NotImplementedError("TODO 3: sample_categorical")
    return len(p) - 1  # only reached through rounding, when u is extremely close to 1


words = ["mat", "sofa", "moon"]
p = [0.5, 0.3, 0.2]
check(sample_categorical(p, 0.62) == 1, "u = 0.62 lands in sofa's piece [0.5, 0.8) (Lesson 4.6)")
check(sample_categorical(p, 0.10) == 0 and sample_categorical(p, 0.95) == 2, "u = 0.10 → mat, u = 0.95 → moon")
check(sample_categorical(p, 0.5) == 1, "u = 0.5 exactly is the start of sofa's piece")


# %% See it: 10,000 rolls of the loaded die
picks = [sample_categorical(p, rng.random()) for _ in range(10000)]
freq = np.bincount(picks, minlength=3) / 10000
for word, prob, f in zip(words, p, freq):
    print(f"{word:>5}: probability {prob:.2f}   picked {f:.3f} of the time")
check(close(freq, p, 0.02), "each word is picked about as often as its probability")


# %% TODO 4: temperature and top-p, the knobs on every LLM API (Lesson 4.6)
def softmax_T(z, T=1.0):
    """Scores (logits) → probabilities, with temperature T."""
    # TODO 4a:
    #   z = np.array(z, dtype=float) / T     (temperature divides the scores first)
    #   z = z − np.max(z)                    (doesn't change the answer, but stops e^1000 overflowing)
    #   e = np.exp(z)
    #   return e / np.sum(e)                 (make them add up to 1)
    raise NotImplementedError("TODO 4a: softmax_T")


check(close(softmax_T([2, 1, 0], 1.0), [0.6652, 0.2447, 0.0900], 1e-4), "T = 1: [2, 1, 0] → [0.665, 0.245, 0.090]")
check(close(softmax_T([2, 1, 0], 2.0), [0.5065, 0.3072, 0.1863], 1e-4), "T = 2 (hotter): flatter, [0.506, 0.307, 0.186]")
check(close(softmax_T([2, 1, 0], 0.5), [0.8668, 0.1173, 0.0159], 1e-4), "T = 0.5 (colder): sharper, [0.867, 0.117, 0.016]")
check(close(softmax_T([1002, 1001, 1000]), softmax_T([2, 1, 0])), "huge scores don't overflow (only the gaps matter)")


def top_p(probs, p_keep):
    """Nucleus sampling: keep the most likely options until they add up to at least p_keep, then rescale."""
    probs = np.array(probs, dtype=float)
    order = np.argsort(probs)[::-1]  # indices from most to least likely
    kept = np.zeros_like(probs)
    total = 0.0
    for i in order:
        # TODO 4b: keep option i, add its probability to the total, and stop once total ≥ p_keep.
        #   kept[i] = probs[i]
        #   total = total + probs[i]
        #   if total >= p_keep: break
        raise NotImplementedError("TODO 4b: top_p")
    return kept / np.sum(kept)  # rescale the survivors so they add up to 1


check(close(top_p([0.5, 0.3, 0.15, 0.05], 0.9), [0.526316, 0.315789, 0.157895, 0.0]),
      "top-p 0.9 on [0.5, 0.3, 0.15, 0.05]: 0.5 + 0.3 + 0.15 ≥ 0.9, drop the rest, rescale (Lesson 4.6)")
check(close(top_p([0.05, 0.5, 0.15, 0.3], 0.9), [0.0, 0.526316, 0.157895, 0.315789]), "works in any order")
check(close(top_p([0.9, 0.06, 0.04], 0.5), [1.0, 0.0, 0.0]), "a confident model keeps just 1 token")


# %% See it: "The cat sat on the ..." with different decoding settings
tokens = ["mat", "sofa", "floor", "roof", "moon"]
logits = np.log([0.40, 0.25, 0.20, 0.10, 0.05])  # scores whose T = 1 softmax is [0.40, 0.25, 0.20, 0.10, 0.05]
for T in [0.2, 1.0, 2.0]:
    probs = softmax_T(logits, T)
    story = [tokens[sample_categorical(probs, rng.random())] for _ in range(12)]
    print(f"T = {T}: probs {np.round(probs, 3)} → {' '.join(story)}")
nucleus = top_p(softmax_T(logits, 1.0), 0.8)
story = [tokens[sample_categorical(nucleus, rng.random())] for _ in range(12)]
print(f"T = 1, top-p 0.8: probs {np.round(nucleus, 3)} → {' '.join(story)}  (the moon is gone)")
check(softmax_T(logits, 0.2)[0] > 0.85, "a cold T = 0.2 lifts 'mat' from 40% to about 89% (T → 0 is greedy decoding)")


# %% TODO 5: expectation and variance (Lesson 4.4)
def expectation(values, probs):
    """𝔼[X] = Σ value × probability."""
    # TODO 5a: return np.sum(np.array(values) * np.array(probs))
    raise NotImplementedError("TODO 5a: expectation")


def variance(values, probs):
    """Var(X) = 𝔼[(X − μ)²], the expected squared distance from the mean."""
    # TODO 5b:
    #   mu = expectation(values, probs)
    #   return expectation((np.array(values) − mu) ** 2, probs)
    raise NotImplementedError("TODO 5b: variance")


die = [1, 2, 3, 4, 5, 6]
sixth = [1 / 6] * 6
check(close(expectation(die, sixth), 3.5), "a die's expected value is 3.5 (you can never roll it!)")
check(close(variance(die, sixth), 35 / 12), "a die's variance is 17.5 / 6 ≈ 2.92 (std ≈ 1.71)")
check(close(expectation([500, 50, 0], [1 / 1000, 10 / 1000, 989 / 1000]), 1.0), "the raffle ticket is worth $1.00 on average")
check(close(expectation([100, 0], [0.01, 0.99]), 1.0) and close(variance([100, 0], [0.01, 0.99]), 99.0),
      "game B: 𝔼 = $1 like game A, but variance 99")


# %% See it: roulette, same average, very different swings (simulated vs formula)
bets = {"$1 on red": ([1, -1], [18 / 38, 20 / 38]), "$1 on number 17": ([35, -1], [1 / 38, 37 / 38])}
for name, (values, probs) in bets.items():
    spins = np.array(values)[rng.choice(2, size=200_000, p=probs)]  # 200,000 simulated spins
    print(f"{name:>16}: formula 𝔼 = {expectation(values, probs):+.4f}, std = {np.sqrt(variance(values, probs)):.2f}"
          f"   simulated mean = {spins.mean():+.4f}, std = {spins.std():.2f}")
    check(abs(spins.std() - np.sqrt(variance(values, probs))) < 0.05, f"{name}: simulated swing matches the formula")
# Same 𝔼 (−5.3 cents per dollar), but the single number swings almost 6× more. In RL, a noisy
# reward with the same mean means a noisier gradient and slower learning.


# %% TODO 6: Monte Carlo π (Lesson 4.6)
def monte_carlo_pi(n):
    """Estimate π by throwing n random darts at the unit square and counting the hits in the quarter circle."""
    # TODO:
    #   x = rng.random(n)
    #   y = rng.random(n)
    #   inside = x ** 2 + y ** 2 < 1      (True when the dart is within distance 1 of the corner)
    #   return 4 * np.mean(inside)        (fraction inside ≈ area π/4, so multiply by 4)
    raise NotImplementedError("TODO 6: monte_carlo_pi")


check(abs(monte_carlo_pi(1_000_000) - np.pi) < 0.01, "a million darts give π to about 2 decimals")
hits = monte_carlo_pi(10) * 10 / 4  # turn the estimate back into a number of hits
check(close(hits, round(hits)), "with 10 darts the answer can only be 4 × (hits / 10): 0, 0.4, 0.8, ...")


# %% See it: the Monte Carlo error shrinks like 1/√n (Lesson 4.7)
spread = {}
for n in [100, 10_000]:
    estimates = [monte_carlo_pi(n) for _ in range(200)]
    spread[n] = np.std(estimates)
    print(f"n = {n:6}: 200 estimates of π have a typical miss (std) of {spread[n]:.4f}")
ratio = spread[100] / spread[10_000]
print(f"100× more darts → {ratio:.1f}× smaller error (√100 = 10)")
check(7 < ratio < 13, "100× more samples ≈ 10× more precise")


# %% TODO 7: the bell curve out of dice (Lesson 4.7)
def dice_averages(n_dice, n_samples):
    """n_samples numbers, each the average of n_dice fair dice."""
    # TODO:
    #   rolls = rng.integers(1, 7, size=(n_samples, n_dice))   (one row per sample, values 1..6)
    #   return rolls.mean(axis=1)                              (the average of each row)
    raise NotImplementedError("TODO 7: dice_averages")


avgs = dice_averages(4, 10)
check(avgs.shape == (10,) and close(avgs * 4, np.round(avgs * 4)), "10 averages of 4 dice, each a multiple of 0.25")
sigma = np.sqrt(35 / 12)  # one die's standard deviation, ≈ 1.708 (TODO 5)
for n in [1, 4, 25, 100]:
    a = dice_averages(n, 20000)
    check(abs(a.mean() - 3.5) < 0.05 and abs(a.std() / (sigma / np.sqrt(n)) - 1) < 0.05,
          f"averages of {n:3} dice: centre 3.5, spread σ/√n = {sigma / np.sqrt(n):.3f} (measured {a.std():.3f})")


# %% See it: the histogram turns into a bell (the central limit theorem)
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, n in zip(axes, [1, 2, 5, 25]):
    a = dice_averages(n, 20000)
    se = sigma / np.sqrt(n)
    ax.hist(a, bins=np.arange(0.5, 6.6, 1 / n) if n <= 5 else 40, density=True, alpha=0.6)
    xx = np.linspace(1, 6, 300)
    ax.plot(xx, np.exp(-(xx - 3.5) ** 2 / (2 * se ** 2)) / (se * np.sqrt(2 * np.pi)), color="purple")
    ax.set_title(f"average of {n} dice"); ax.set_yticks([])
fig.suptitle("Purple: the CLT's prediction N(3.5, σ²/n)"); plt.tight_layout(); show()
a = dice_averages(100, 20000)
inside = np.mean(np.abs(a - 3.5) < 1.96 * sigma / 10)
print(f"averages of 100 dice within ±1.96 SE of 3.5: {inside:.1%}")
check(0.93 < inside < 0.97, "about 95% land within 1.96 standard errors: the basis of every 95% interval (Lesson 5.5)")


# %% TODO 8: importance sampling (Lesson 4.6b)
def importance_estimate(xs, f, p, q):
    """Estimate 𝔼_p[f(X)] from samples xs that were drawn from q. Also return the effective sample size."""
    # TODO:
    #   w = p(xs) / q(xs)                       (importance weights, one per sample)
    #   estimate = np.mean(w * f(xs))
    #   ess = np.sum(w) ** 2 / np.sum(w ** 2)   (effective sample size: n if all weights are equal)
    #   return estimate, ess
    raise NotImplementedError("TODO 8: importance_estimate")


fair = lambda x: np.full(len(x), 1 / 6)  # q: the die your friend rolled
loaded = lambda x: np.where(x == 6, 0.5, 0.1)  # p: the die you care about (6 half the time)
roll_value = lambda x: x  # f: we want the average roll
est, ess = importance_estimate(np.array([4, 6, 2, 5, 1, 3]), roll_value, loaded, fair)
check(close(est, 4.5) and close(ess, 36 / 10.8), "rolls 4, 6, 2, 5, 1, 3 → 4.5, spot on (ESS 3.3 of 6)")
est, ess = importance_estimate(np.array([6, 6, 5, 6, 2, 1]), roll_value, loaded, fair)
check(close(est, 9.8), "lucky sixes 6, 6, 5, 6, 2, 1 → 9.8, way too high (each 6 counts 3×)")
est, ess = importance_estimate(np.array([3, 1, 5, 2, 4, 2]), roll_value, loaded, fair)
check(close(est, 1.7) and close(ess, 6), "no sixes 3, 1, 5, 2, 4, 2 → 1.7, too low (equal weights, ESS = 6)")


# %% See it: reweighting works on average, but ESS collapses when p and q drift apart
friend_rolls = rng.integers(1, 7, size=600)
est, ess = importance_estimate(friend_rolls, roll_value, loaded, fair)
print(f"600 fair rolls: naive average {friend_rolls.mean():.2f}, reweighted {est:.2f}, truth 4.5, ESS {ess:.0f} of 600")
check(abs(est - 4.5) < 0.4, "600 borrowed rolls, reweighted, estimate the loaded die's average")

# Samples from q = N(0, 1). Target p = N(μ, 1): its true mean is μ. Move p away and watch ESS.
normal = lambda mu: (lambda x: np.exp(-(x - mu) ** 2 / 2) / np.sqrt(2 * np.pi))
xs = rng.standard_normal(1000)
for mu in [0.5, 2.0, 4.0]:
    est, ess = importance_estimate(xs, lambda x: x, normal(mu), normal(0.0))
    print(f"p = N({mu}, 1): estimate of its mean {est:5.2f} (truth {mu}), ESS = {ess:6.1f} of 1000")
    if mu == 0.5:
        check(ess > 600, "p close to q: most samples still count")
    if mu == 4.0:
        check(ess < 20, "p far from q: 1000 samples are worth only a handful")
# PPO does exactly this with new-policy / old-policy probabilities, and clips the ratio to
# [0.8, 1.2] so a few huge weights can't hijack the update (Lesson 4.6b).


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. How many people do you need for a 99.9% chance of a shared birthday? Guess first.
# 2. What goes wrong in `softmax_T` at T = 0? How would you implement "T = 0" (greedy) instead?
# 3. Importance sampling with p = N(4, 1): try q = N(0, 3) (a WIDER proposal). Does ESS
#    recover? Why does a q that covers p's region keep the weights tame?
# 4. The "weird die" of Lesson 4.7 (1: 50%, 2: 30%, 6: 20%) is lopsided. How many dice must
#    you average before the histogram looks like a bell?

# %% [markdown]
# # Lab 0: The language of math, in code 🌱
#
# You'll turn Phase 0's math into small Python functions: a straight line y = w·x + b,
# a ReLU and a "bump" built from ReLU pieces (a tiny neural network!), Σ and Π as plain
# loops, log-probabilities that don't underflow, softmax, and an RL discounted return.
#
# How to run: open this file in VS Code and click "Run Cell" above each `# %%`,
# or run the whole thing with `python3 phase0_language.py`. Setup (once):
# `pip install -r requirements.txt`.
# ✅ SOLUTION FILE: every TODO is filled in. Try labs/phase0_language.py yourself first!
# Lessons to reread if stuck: 0.2 (functions, ReLU), 0.3 (logs), 0.4 (e, exp, softmax),
# 0.5 (Σ and Π).

# %%
import math
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*non-interactive.*")  # quiet when there's no screen

rng = np.random.default_rng(0)  # one seeded random generator → the same "random" numbers every run


def check(cond, msg):
    """Stop with ❌ if cond is False, otherwise print ✅ and the message."""
    assert cond, "❌ " + msg
    print("✅", msg)


def close(a, b, tol=1e-3):
    """True if a and b (numbers or arrays) are equal up to a small rounding tolerance."""
    return np.allclose(a, b, atol=tol)


# %% TODO 1: a function = a machine. The straight line y = w·x + b
def line(x, w, b):
    # TODO: return w times x, plus b.
    # Hint: plain * and + work for single numbers AND for whole NumPy arrays.
    return w * x + b


check(line(10, 1.8, 32) == 50, "line(10, w=1.8, b=32) = 50: 10 °C is 50 °F")
check(close(line(np.array([0, 100]), 1.8, 32), [32, 212]), "works on a whole array: 0 °C → 32 °F, 100 °C → 212 °F")
check(line(0, 3, -2) == -2, "at x = 0 the line is at height b")


# %% See it: w tilts the line, b lifts it. One neuron starts with exactly this w·x + b.
xs = np.linspace(-3, 3, 50)
plt.figure(figsize=(6, 4))
for w, b in [(1, 0), (2, 0), (-1, 0), (1, 2)]:
    plt.plot(xs, line(xs, w, b), label=f"w = {w}, b = {b}")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.title("y = w·x + b"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# %% TODO 2: ReLU, the bend that every modern neural network uses
def relu(x):
    # TODO: return x where x is positive, and 0 where it isn't.
    # Hint: np.maximum(0, x) compares 0 with every number of x, keeping the bigger one.
    return np.maximum(0, x)


check(close(relu(np.array([-2, 0.5, 3])), [0, 0.5, 3]), "relu([-2, 0.5, 3]) = [0, 0.5, 3]")
check(relu(-4) + relu(3) == 3, "ReLU(−4) + ReLU(3) = 0 + 3 = 3")
check(close(relu(line(np.array([-1, 0, 1, 2]), 2, -1)), [0, 0, 1, 3]), "the neuron ReLU(2x − 1) on −1, 0, 1, 2 gives 0, 0, 1, 3")


# %% TODO 3: build a bump out of three ReLU pieces (Lesson 0.2's curve builder)
def bump(x):
    # TODO: add three hockey sticks, each switching on at a different place:
    #   +1 × relu(x + 1)   switches on at x = −1 and climbs
    #   −2 × relu(x)       switches on at x =  0 and turns the climb into a fall
    #   +1 × relu(x − 1)   switches on at x = +1 and makes it flat again
    # Hint: one line, relu(x + 1) - 2 * relu(x) + relu(x - 1)
    return relu(x + 1) - 2 * relu(x) + relu(x - 1)


check(close(bump(0), 1), "the bump's peak: bump(0) = 1 − 0 + 0 = 1")
check(close(bump(np.array([-1, -0.5, 0.5, 1])), [0, 0.5, 0.5, 0]), "it climbs from −1 to 0 and falls from 0 to 1")
check(close(bump(np.array([-5, 3])), [0, 0]), "flat zero far away: bump(3) = 4 − 6 + 2 = 0")


# %% See it: two bumps make a mountain range. A hidden layer of ReLUs does exactly this.
xs = np.linspace(-3, 5, 400)
plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1)
plt.plot(xs, relu(xs + 1), "--", label="relu(x + 1)")
plt.plot(xs, -2 * relu(xs), "--", label="−2·relu(x)")
plt.plot(xs, relu(xs - 1), "--", label="relu(x − 1)")
plt.plot(xs, bump(xs), "k", lw=3, label="sum = bump")
plt.ylim(-3, 3); plt.title("3 hockey sticks → 1 bump"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.subplot(1, 2, 2)
mountain = 2 * bump(xs) + 1 * bump(xs - 3)  # a tall bump at 0, a smaller one moved to x = 3
plt.plot(xs, mountain, "g", lw=3)
plt.title("2·bump(x) + bump(x − 3): a mountain range"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("A neural network = many w·x + b lines, each bent by ReLU, then added up with strengths.")
print("With enough pieces it can draw almost any curve. Training just finds the strengths.")


# %% TODO 4: Σ and Π are just for-loops
def sum_loop(xs):
    # TODO: Σ x_i, "add up every x". Start a running total at 0, add each x to it, return it.
    # Hint:
    #   total = 0
    #   for x in xs:
    #       total = total + x
    total = 0
    for x in xs:
        total = total + x
    return total


def prod_loop(xs):
    # TODO: Π x_i, "multiply every x together". Same loop, but start at 1 and multiply.
    # (Why start at 1? Because 1 is "multiplying by nothing", like 0 is "adding nothing".)
    total = 1
    for x in xs:
        total = total * x
    return total


x = np.array([3, 1, 4, 1, 5])
check(sum_loop(x) == 14, "Σ x = 3 + 1 + 4 + 1 + 5 = 14")
check(prod_loop(x) == 60, "Π x = 3 × 1 × 4 × 1 × 5 = 60")
check(sum_loop(x) == x.sum() and prod_loop(x) == x.prod(), "your loops agree with NumPy's x.sum() and x.prod()")
check(close(sum_loop(x) / len(x), 2.8), "the mean x̄ = (1/n) Σ x = 14 / 5 = 2.8")


# %% See it: same answer, very different speed. This is why AI code is written with NumPy.
big = rng.random(1_000_000)
t0 = time.perf_counter(); s1 = sum_loop(big); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); s2 = big.sum(); t_np = time.perf_counter() - t0
print(f"Python loop: {s1:.3f} in {t_loop * 1000:.1f} ms")
print(f"NumPy sum:   {s2:.3f} in {t_np * 1000:.2f} ms  (about {t_loop / max(t_np, 1e-9):.0f}× faster)")
check(close(s1, s2, tol=1e-6), "loop and NumPy give the same Σ of a million numbers")


# %% TODO 5: the log of a product = the sum of the logs
def log_prob(ps):
    # TODO: return Σ ln p_i, the log of the probability of the whole sequence.
    # Hint: np.log(ps) takes ln of every number at once; then .sum() (or your sum_loop) adds them.
    return np.log(ps).sum()


p = np.array([0.5, 0.2, 0.1])
check(close(log_prob(p), -4.605), "Σ ln p for [0.5, 0.2, 0.1] = −4.605")
check(close(log_prob(p), math.log(0.5 * 0.2 * 0.1)), "same as ln(0.5 × 0.2 × 0.1) = ln(0.01)")
check(close(math.exp(log_prob(p)), 0.01), "exp undoes ln: e^(−4.605) = 0.01")


# %% See it: the log rules, checked with numbers, and why LLMs live in "log-land"
a, b = 8.0, 4.0
check(close(math.log(a * b), math.log(a) + math.log(b)), "ln(a·b) = ln a + ln b   (multiply → add)")
check(close(math.log(a / b), math.log(a) - math.log(b)), "ln(a/b) = ln a − ln b   (divide → subtract)")
check(close(math.log(a ** 3), 3 * math.log(a)), "ln(a³) = 3·ln a          (power → multiply)")
check(close(math.log2(8), 3), "log₂ 8 = 3: 'how many 2s multiply to 8?'")

# 100 tokens, each with probability 0.1, stored as 32-bit floats (common in AI)
p100 = np.full(100, 0.1, dtype=np.float32)
direct = np.prod(p100)   # the true answer is 10^−100, far below float32's floor (about 10^−45)
in_logs = log_prob(p100)
print("direct product of 100 probabilities:", direct)
print("sum of their logs:                  ", round(float(in_logs), 2), " (= 100 × ln 0.1)")
check(direct == 0.0, "the direct product underflows to exactly 0 (all information lost)")
check(close(in_logs, -230.26, tol=0.01), "the sum of logs is a perfectly normal number: −230.26")
check(prod_loop([0.1] * 1000) == 0.0, "even 64-bit floats give up at 1,000 tokens (10^−1000 → 0)")
print("That's why language models add log-probabilities instead of multiplying probabilities.")


# %% TODO 6: softmax turns any scores into probabilities
def softmax(z):
    # TODO: three steps.
    #   1. e = np.exp(z - np.max(z))   e to every score (subtracting the max changes nothing
    #                                   in the answer, but stops exp from overflowing)
    #   2. divide e by its total, e.sum()
    #   3. return that
    e = np.exp(z - np.max(z))
    return e / e.sum()


check(close(softmax(np.array([2.0, 1.0, 0.0])), [0.665, 0.245, 0.090]), "softmax([2, 1, 0]) = [0.665, 0.245, 0.090]")
check(close(softmax(np.array([5.0, -1.0, 3.0, 0.0])).sum(), 1.0), "the probabilities always add up to 1")
check(close(softmax(np.array([12.0, 11.0, 10.0])), softmax(np.array([2.0, 1.0, 0.0]))), "adding 10 to every score changes nothing")
check(close(softmax(np.array([1000.0, 999.0])), [0.731, 0.269]), "huge scores (1000, 999) don't overflow")


# %% See it: grade a tiny language model (the Phase 0 recap problem) + temperature
# After each prefix the model scores 3 candidate words. The first word in each row is the right one.
rows = [
    ("The ___",            ["cat", "dog", "car"],     [2.0, 1.0, 0.0]),
    ("The cat ___",        ["sat", "ran", "flew"],    [1.0, 1.0, -1.0]),
    ("The cat sat on the", ["mat", "sofa", "moon"],   [0.0, 2.0, -2.0]),
]
p_right = []
for prefix, cands, scores in rows:
    probs = softmax(np.array(scores))
    p_right.append(probs[0])
    print(f"{prefix:<20} p({cands[0]}) = {probs[0]:.3f}   loss term −ln p = {-math.log(probs[0]):.3f}")
loss = -log_prob(np.array(p_right)) / len(p_right)   # L = −(1/N) Σ ln p
print(f"average loss L = {loss:.3f}   (the row with 'sofa' hurts most)")
check(close(loss, 1.103), "the tiny model's loss is 1.103, exactly as in the recap")

z = np.array([2.0, 1.0, 0.0])
for T in [0.5, 1.0, 2.0]:
    print(f"temperature T = {T}: softmax(z / T) = {np.round(softmax(z / T), 3)}")
print("Low T → sharper (more confident), high T → flatter (more random). Chatbots expose this knob.")


# %% TODO 7: an RL agent's discounted return  G = Σ γ^(t−1) · r_t
def discounted_return(rewards, gamma):
    # TODO: rewards[0] is r₁ (counts fully), rewards[1] is r₂ (shrunk by γ once),
    #       rewards[2] is r₃ (shrunk by γ twice), and so on. Add them all up.
    # Hint (Python counts from 0, so reward number i is shrunk by gamma ** i):
    #   G = 0
    #   for i in range(len(rewards)):
    #       G = G + gamma ** i * rewards[i]
    G = 0
    for i in range(len(rewards)):
        G = G + gamma ** i * rewards[i]
    return G


check(close(discounted_return([1, 1, 1], 0.9), 2.71), "[1, 1, 1] with γ = 0.9: 1 + 0.9 + 0.81 = 2.71")
check(close(discounted_return([0, 0, 1], 0.9), 0.81), "a reward 2 steps late is worth 0.9² = 0.81")
check(close(discounted_return([5, 0, 0], 0.5), 5), "a reward right now isn't shrunk at all")
late = np.zeros(101); late[100] = 1.0
check(close(discounted_return(late, 0.99), 0.366), "a reward 100 steps away with γ = 0.99 is worth 0.99¹⁰⁰ = 0.366")


# %% See it: how fast does the future fade? (γ is exp's cousin: repeated multiplying)
steps = np.arange(0, 200)
plt.figure(figsize=(6, 4))
for g in [0.5, 0.9, 0.99]:
    plt.plot(steps, g ** steps, label=f"γ = {g}")
plt.xlabel("steps into the future"); plt.ylabel("weight γ^steps"); plt.title("Discounting: later rewards count less")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
for g in [0.9, 0.99]:
    print(f"γ = {g}: an endless stream of +1 rewards is worth about {discounted_return(np.ones(2000), g):.1f} = 1 / (1 − γ)")


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Make a "staircase": use relu pieces so the curve is 0, then climbs to 1, stays flat, then
#    climbs to 2. How many pieces do you need? (Hint: each change of slope costs one piece.)
# 2. The tiny model's worst row was "The cat sat on the ___" (it liked "sofa"). Change only that
#    row's scores so the average loss drops below 0.6. What's the smallest change that works?
# 3. An agent gets +1 every step for 50 steps, then nothing. With γ = 0.9 and γ = 0.99, how much of
#    the return comes from the first 10 steps? Which γ makes the agent more "far-sighted"?

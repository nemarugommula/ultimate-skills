# %% [markdown]
# # Lab 7: Optimization from scratch ⛰️
#
# You'll build the training toolbox of Phase 7 with NumPy: MSE vs MAE with an
# outlier, mini-batch SGD on a line fit (and how batch size changes the noise),
# momentum and Adam written by hand and raced down a narrow valley, the
# warmup + cosine learning-rate schedule LLMs use, L2 regularization calming a
# wiggly polynomial, and early stopping on a validation curve.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase7_optimization.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 7.1 (losses), 7.2 (SGD & mini-batches),
# 7.3 (momentum, Adam, schedules), 7.4 (regularization, early stopping)

# %%
import numpy as np
import matplotlib.pyplot as plt

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


# %% TODO 1: two loss functions   MSE = mean(e²),  MAE = mean(|e|)   (Lesson 7.1)
def mse(e):
    """Mean squared error of an array of errors e = prediction − truth."""
    e = np.asarray(e, dtype=float)
    # TODO 1a: square every error, then average. Hint: e ** 2 and .mean()
    raise NotImplementedError("TODO 1a: mse")


def mae(e):
    """Mean absolute error: the average size of the errors, ignoring their sign."""
    e = np.asarray(e, dtype=float)
    # TODO 1b: take the absolute value of every error, then average. Hint: np.abs
    raise NotImplementedError("TODO 1b: mae")


errors = [1, -2, 0.5, 10]  # one outlier: 10
check(close(mse(errors), 26.3125), "MSE of [1, −2, 0.5, 10] = (1 + 4 + 0.25 + 100) / 4 = 26.3125")
check(close(mae(errors), 3.375), "MAE of the same errors = (1 + 2 + 0.5 + 10) / 4 = 3.375")
check(mse(errors) / mse([1, -2, 0.5]) > 15, "squaring lets one outlier dominate MSE")

# %% See it: the best single guess under MSE is the MEAN; under MAE it's the MEDIAN
# Five house prices (in $100k), one of them a mansion. Which single number c "predicts" them best?
prices = np.array([2.0, 3.0, 3.0, 4.0, 30.0])
cs = np.linspace(0, 31, 3101)  # candidate guesses 0.00, 0.01, ..., 31.00
mse_curve = [mse(c - prices) for c in cs]
mae_curve = [mae(c - prices) for c in cs]
best_mse, best_mae = cs[np.argmin(mse_curve)], cs[np.argmin(mae_curve)]
print(f"MSE's best guess = {best_mse:.2f} (the mean is {prices.mean():.2f}): dragged up by the mansion")
print(f"MAE's best guess = {best_mae:.2f} (the median is {np.median(prices):.2f}): ignores how far away the mansion is")
check(close(best_mse, prices.mean(), 0.01) and close(best_mae, np.median(prices), 0.01), "MSE → mean, MAE → median")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(cs, mse_curve); ax[0].axvline(best_mse, ls="--"); ax[0].set_title("MSE: smooth bowl, min at the mean")
ax[1].plot(cs, mae_curve); ax[1].axvline(best_mae, ls="--"); ax[1].set_title("MAE: V-shaped, min at the median")
for a in ax:
    a.set_xlabel("guess c"); a.grid(alpha=0.3)
plt.tight_layout(); show()


# %% TODO 2: the gradient of MSE for a line  ŷ = w·x + b   (Lessons 7.2 and 8.1)
def grad_linear(w, b, x, y):
    """Gradient of the loss mean((w·x + b − y)²) with respect to w and to b."""
    err = w * x + b - y  # one error per example
    # TODO 2: grad_w = average of 2·err·x,  grad_b = average of 2·err.  Return both.
    # Hint: (2 * err * x).mean()
    raise NotImplementedError("TODO 2: grad_linear")


gw, gb = grad_linear(0.0, 0.0, np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0]))
check(close(gw, -15.333) and close(gb, -6.667), "Lesson 8.1's worked example: ∂L/∂w = −15.33, ∂L/∂b = −6.67")
gw, gb = grad_linear(1.5, 1 / 3, np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 5.0]))
check(close(gw, 0.0) and close(gb, 0.0), "at the best line (w = 1.5, b = 0.33) the gradient is 0: the bottom of the bowl")

# %% Data for the rest of the SGD part: 1,000 points around the line y = 3x + 2
N = 1000
X = rng.standard_normal(N)
Y = 3 * X + 2 + 0.5 * rng.standard_normal(N)


def full_loss(w, b):
    return mse(w * X + b - Y)


# %% TODO 3: mini-batch SGD   (Lesson 7.2)
def sgd(B, eta, epochs):
    """Train w, b on (X, Y) with mini-batches of B examples. Returns w, b and the full-data loss after every step."""
    w, b = 0.0, 0.0
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(N)  # shuffle once per epoch
        for s in range(0, N, B):
            i = order[s:s + B]  # the row numbers in this mini-batch
            # TODO 3: 1. gradient on this mini-batch only: grad_linear(w, b, X[i], Y[i])
            #         2. step downhill: w ← w − eta·grad_w and b ← b − eta·grad_b
            raise NotImplementedError("TODO 3: the SGD step")
            losses.append(full_loss(w, b))
    return w, b, losses


w, b, losses = sgd(B=10, eta=0.05, epochs=5)
check(abs(w - 3) < 0.1 and abs(b - 2) < 0.1, f"SGD finds the line: w = {w:.2f} (true 3), b = {b:.2f} (true 2)")
check(len(losses) == 5 * N // 10, "5 epochs of 1,000 examples in batches of 10 = 500 steps")
check(losses[-1] < 0.3, f"final loss {losses[-1]:.3f} is near the noise floor 0.5² = 0.25")

# %% See it: bigger batches → less noisy gradients, but noise only shrinks like 1/√B
# At a fixed (bad) w, b = 0, 0, draw 2,000 random mini-batches of each size and look at how much grad_w scatters.
for B in [1, 4, 16, 64]:
    g = [grad_linear(0.0, 0.0, X[j], Y[j])[0] for j in [rng.choice(N, B, replace=False) for _ in range(2000)]]
    print(f"B = {B:<3} spread (std) of the mini-batch gradient = {np.std(g):.2f}")
print("Each 4× bigger batch only halves the noise. That's why many cheap steps usually win.")

plt.figure(figsize=(7, 3.5))
for B in [1, 10, 100]:
    _, _, ls = sgd(B=B, eta=0.05, epochs=1)
    plt.plot(np.arange(1, len(ls) + 1) * B, ls, label=f"B = {B} ({len(ls)} steps)")
plt.xscale("log"); plt.yscale("log"); plt.xlabel("examples read"); plt.ylabel("full-data loss")
plt.title("Same data budget (1 epoch), different batch sizes"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); show()


# %% TODO 4: momentum (the heavy ball)   m ← β·m + g,  w ← w − η·m   (Lesson 7.3)
def momentum_step(w, m, g, eta, beta=0.9):
    """One momentum step. w = weights, m = velocity (running sum of gradients), g = gradient now."""
    # TODO 4: update m first, then w. Return the new w and the new m.
    raise NotImplementedError("TODO 4: momentum_step")


w, m = momentum_step(0.0, 0.0, g=1.0, eta=0.1)
check(close([w, m], [-0.1, 1.0]), "step 1 with g = 1: m = 1, w = −0.1")
w, m = momentum_step(w, m, g=1.0, eta=0.1)
check(close([w, m], [-0.29, 1.9]), "step 2: m = 0.9·1 + 1 = 1.9, w = −0.1 − 0.19 = −0.29 (it speeds up)")


# %% TODO 5: Adam   (Lesson 7.3)
def adam_step(w, m, v, t, g, eta, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step at step number t (t starts at 1). Works on single numbers or whole arrays."""
    # TODO 5: 1. m = b1·m + (1 − b1)·g          (average gradient)
    #         2. v = b2·v + (1 − b2)·g²         (average squared gradient)
    #         3. m_hat = m / (1 − b1**t),  v_hat = v / (1 − b2**t)   (fix the start-at-zero bias)
    #         4. w = w − eta · m_hat / (√v_hat + eps)
    #         Return w, m, v.
    raise NotImplementedError("TODO 5: adam_step")


w, m, v = adam_step(0.0, 0.0, 0.0, t=1, g=0.5, eta=0.001)
check(close(m, 0.05, 1e-9) and close(v, 0.00025, 1e-9), "Lesson 7.3's first step: m = 0.05, v = 0.00025")
check(close(w, -0.001, 1e-6), "and the first step moves exactly η = 0.001")
check(close(adam_step(0.0, 0.0, 0.0, t=1, g=50.0, eta=0.001)[0], -0.001, 1e-6), "a 100× louder gradient: still exactly η")

# %% See it: race plain gradient descent, momentum and Adam down a narrow valley
# L = 0.02·w₁² + 2·w₂²: a ravine 100× steeper across (w₂) than along (w₁), like real network losses.
def ravine_loss(w):
    return 0.02 * w[0] ** 2 + 2 * w[1] ** 2


def ravine_grad(w):
    return np.array([0.04 * w[0], 4 * w[1]])


start, eta, steps = np.array([-9.0, 1.5]), 0.4, 100
paths = {"gradient descent": [start], "momentum β = 0.9": [start], "Adam": [start]}
w_gd, w_mom, w_adam = start.copy(), start.copy(), start.copy()
m_mom, m_adam, v_adam = np.zeros(2), np.zeros(2), np.zeros(2)
for t in range(1, steps + 1):
    w_gd = w_gd - eta * ravine_grad(w_gd)
    w_mom, m_mom = momentum_step(w_mom, m_mom, ravine_grad(w_mom), eta)
    w_adam, m_adam, v_adam = adam_step(w_adam, m_adam, v_adam, t, ravine_grad(w_adam), eta)
    paths["gradient descent"].append(w_gd); paths["momentum β = 0.9"].append(w_mom); paths["Adam"].append(w_adam)
for name, p in paths.items():
    print(f"{name:<18} loss after {steps} steps = {ravine_loss(p[-1]):.5f}")
check(ravine_loss(w_mom) < ravine_loss(w_gd) / 10 and ravine_loss(w_adam) < ravine_loss(w_gd) / 10,
      "momentum and Adam both beat plain gradient descent by more than 10× on the ravine")

g1, g2 = np.meshgrid(np.linspace(-10, 3, 200), np.linspace(-2, 2, 200))
plt.figure(figsize=(8, 3.8))
plt.contour(g1, g2, 0.02 * g1 ** 2 + 2 * g2 ** 2, levels=15, alpha=0.4)
for name, p in paths.items():
    p = np.array(p)
    plt.plot(p[:, 0], p[:, 1], ".-", ms=3, label=name)
plt.plot(0, 0, "k*", ms=12); plt.xlabel("w₁ (gentle direction)"); plt.ylabel("w₂ (steep direction)")
plt.title(f"{steps} steps, same η = {eta}"); plt.legend(loc="lower right", fontsize=8); plt.tight_layout(); show()


# %% TODO 6: warmup + cosine learning-rate schedule   (Lesson 7.3)
def lr_schedule(t, peak, floor, warmup, total):
    """Learning rate at step t: a straight ramp 0 → peak over `warmup` steps, then a cosine curve peak → floor."""
    # TODO 6: if t < warmup: return peak · t / warmup
    #         otherwise:  progress = (t − warmup) / (total − warmup)      (0 at the top, 1 at the end)
    #                     return floor + 0.5 · (peak − floor) · (1 + cos(π · progress))
    # Hint: np.cos and np.pi
    raise NotImplementedError("TODO 6: lr_schedule")


sched = dict(peak=1e-3, floor=1e-4, warmup=300, total=3000)
check(close(lr_schedule(150, **sched), 5e-4, 1e-9), "step 150 is halfway up the warmup: η = 5 × 10⁻⁴")
check(close(lr_schedule(300, **sched), 1e-3, 1e-9), "step 300: the peak, η = 10⁻³")
check(close(lr_schedule(1650, **sched), 5.5e-4, 1e-9), "step 1,650 is halfway down the cosine: the average of peak and floor")
check(close(lr_schedule(3000, **sched), 1e-4, 1e-9), "the last step lands on the floor, 10⁻⁴")

# %% See it: the schedule GPT-3 used (linear warmup, then cosine decay to 10% of the peak)
ts = np.arange(3001)
plt.figure(figsize=(7, 3))
plt.plot(ts, [lr_schedule(t, **sched) for t in ts])
plt.xlabel("step"); plt.ylabel("learning rate η"); plt.title("Warmup + cosine (peak 1e-3, floor 1e-4)")
plt.grid(alpha=0.3); plt.tight_layout(); show()
print("Warmup: early gradients are wild and Adam's averages are still young, so start gently.")
print("Cosine: big steps to travel, small steps at the end to settle into the bottom of the valley.")


# %% TODO 7: L2 regularization (ridge)   minimize Σ err² + λ·Σ w²   (Lesson 7.4)
def ridge_fit(F, y, lam):
    """F = feature matrix (one row per example, one column per feature). Returns the weights w that
    minimize Σ (F w − y)² + lam · Σ w². Setting the slope to 0 gives (FᵀF + λI) w = Fᵀy."""
    d = F.shape[1]  # number of weights
    # TODO 7: solve (FᵀF + lam·I) w = Fᵀy for w and return it.
    # Hint: np.linalg.solve(A, c) finds the w with A @ w = c.  I = np.eye(d).  Fᵀ is F.T
    raise NotImplementedError("TODO 7: ridge_fit")


F_small, y_small = np.array([[1.0], [2.0]]), np.array([1.0, 2.0])
check(close(ridge_fit(F_small, y_small, 0.0), [1.0]), "no penalty: y = x fits perfectly, w = 5/5 = 1")
check(close(ridge_fit(F_small, y_small, 5.0), [0.5]), "λ = 5: w = 5 / (5 + 5) = 0.5, pulled halfway to 0")

# %% See it: λ tames a wiggly polynomial
# 10 noisy points from a sine wave, fitted with a degree-9 polynomial (10 weights: it can pass through every point).
# Features: Legendre polynomials P₀(x) … P₉(x), a better-behaved version of 1, x, x², …, x⁹ (same curves they can draw).
def poly_features(x):
    return np.polynomial.legendre.legvander(x, 9)


x_train = np.linspace(-1, 1, 10)
y_train = np.sin(np.pi * x_train) + 0.3 * rng.standard_normal(10)
x_val = np.linspace(-1, 1, 101)  # validation points: new data the model never trains on
y_val = np.sin(np.pi * x_val) + 0.3 * rng.standard_normal(101)
F_train, F_val = poly_features(x_train), poly_features(x_val)

results = {}
for lam in [0.0, 0.01, 0.1, 1.0]:
    w = ridge_fit(F_train, y_train, lam)
    results[lam] = (np.linalg.norm(w), mse(F_train @ w - y_train), mse(F_val @ w - y_val))
    print(f"λ = {lam:<5} ‖w‖ = {results[lam][0]:.2f}  train MSE = {results[lam][1]:.3f}  validation MSE = {results[lam][2]:.3f}")
check(results[0.0][1] < 1e-6, "λ = 0: train error 0. It memorized the noise")
check(results[0.0][0] > results[0.1][0] > results[1.0][0], "bigger λ → smaller weights")
check(results[0.1][2] < results[0.0][2], "λ = 0.1 does better on new data than λ = 0")

xs = np.linspace(-1, 1, 300)
plt.figure(figsize=(7, 3.8))
plt.plot(x_train, y_train, "ko", label="training points")
plt.plot(xs, np.sin(np.pi * xs), "k:", label="truth: sin(πx)")
for lam in [0.0, 0.1]:
    plt.plot(xs, poly_features(xs) @ ridge_fit(F_train, y_train, lam), label=f"λ = {lam}")
plt.ylim(-2.5, 2.5); plt.legend(); plt.title("L2 penalty: smaller weights, smoother curve")
plt.grid(alpha=0.3); plt.tight_layout(); show()


# %% TODO 8: early stopping   (Lesson 7.4)
def early_stop(val_losses, patience):
    """Walk through the validation losses in order. Stop once `patience` checks in a row fail to beat the best.
    Returns (best_index, stop_index)."""
    best, best_i, bad = np.inf, 0, 0  # best loss so far, where it happened, checks in a row without improvement
    for i, v in enumerate(val_losses):
        # TODO 8: if v < best: remember it (best = v, best_i = i) and reset bad to 0
        #         else: bad += 1, and if bad == patience: return best_i, i
        raise NotImplementedError("TODO 8: early_stop")
    return best_i, len(val_losses) - 1  # never ran out of patience: stop at the end


curve = [5.0, 4.0, 3.0, 3.5, 3.2, 3.1, 3.3, 3.4, 3.6]
check(early_stop(curve, patience=3) == (2, 5), "patience 3: best at index 2 (loss 3.0), stop at index 5")
check(early_stop(curve, patience=10) == (2, 8), "patience 10: never runs out, best is still index 2")
check(early_stop([3.0, 2.0, 1.0], patience=2) == (2, 2), "always improving: best and stop are the last check")

# %% See it: early stopping on a real validation curve
# Train the same degree-9 polynomial by gradient descent (no penalty) and check the validation loss every 100 steps.
w = np.zeros(10)
train_curve, val_curve = [], []
for step in range(20000):
    w = w - 0.05 * (2 * F_train.T @ (F_train @ w - y_train) / len(y_train))  # gradient of the train MSE
    if step % 100 == 0:
        train_curve.append(mse(F_train @ w - y_train)); val_curve.append(mse(F_val @ w - y_val))
best_i, stop_i = early_stop(val_curve, patience=10)
print(f"best validation MSE {val_curve[best_i]:.3f} at step {best_i * 100}; stopped at step {stop_i * 100}")
print(f"training all 20,000 steps instead: train MSE {train_curve[-1]:.4f}, validation MSE {val_curve[-1]:.3f}")
check(val_curve[best_i] < val_curve[-1], "the early-stopped model beats the fully trained one on new data")

plt.figure(figsize=(7, 3.5))
plt.plot(np.arange(len(train_curve)) * 100, train_curve, label="train")
plt.plot(np.arange(len(val_curve)) * 100, val_curve, label="validation")
plt.axvline(best_i * 100, ls="--", c="k", label="early stop picks this")
plt.yscale("log"); plt.xlabel("step"); plt.ylabel("MSE"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Train keeps falling, validation turns back up"); plt.tight_layout(); show()

# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. In the ravine race, push η up to 0.55. Plain gradient descent blows up (why? the
#    steep direction's slope is 4·w₂, so each step multiplies w₂ by 1 − 4η). Does Adam?
# 2. Re-run the batch-size plot with the "linear scaling rule": η = 0.005 × B (capped
#    at 0.5). Does B = 100 now keep up with B = 10 for the same number of examples?
# 3. Early stopping and λ both fight overfitting. Train the polynomial by gradient descent
#    with the penalty added to the gradient (+ 2λw). Does the validation curve still turn up?

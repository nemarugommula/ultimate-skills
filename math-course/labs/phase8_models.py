# %% [markdown]
# # Lab 8: The models, from the math up 🧠
#
# You'll build the core of every Phase 8 model with NumPy only: logistic
# regression trained with the (p − y)·x gradient, a gradient-boosting round,
# a numerically safe softmax (log-sum-exp), a 2-layer neural network with
# backprop that learns to separate two rings of points, a convolution done
# with loops, causal self-attention softmax(QKᵀ/√d)V on 4 toy tokens, and
# value iteration (the Bellman equation) on a small gridworld.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase8_models.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 8.1 (logistic regression), 8.1b (boosting),
# 8.2 (softmax), 8.3 (neural network), 8.3b (convolution), 8.4 (attention),
# 8.5 (log-sum-exp), 8.6 (Bellman)

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


def sigmoid(z):
    """σ(z) = 1 / (1 + e^(−z)): squashes any score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))


# %% TODO 1: one gradient-descent step of logistic regression   (Lesson 8.1)
def logistic_step(w, b, X, y, eta):
    """X: one row per example. y: 0/1 labels. Returns the new w and b after one step on the
    average log-loss, whose gradient is the average of (p − y)·x."""
    p = sigmoid(X @ w + b)  # predicted probability of "yes" for every example
    # TODO 1: grad_w = Xᵀ(p − y) / n,  grad_b = average of (p − y). Then step: w − eta·grad_w, b − eta·grad_b.
    # Hint: n = len(y);  X.T @ (p - y) adds up (p − y)·x over all examples.
    raise NotImplementedError("TODO 1: logistic_step")


# Lesson 8.1's worked example: one ad, x = [2, 1], it was clicked (y = 1).
w, b = logistic_step(np.array([1.5, -1.0]), -0.5, np.array([[2.0, 1.0]]), np.array([1.0]), eta=0.5)
check(close(w, [1.682, -0.909]) and close(b, -0.409), "one click: w = [1.682, −0.909], b = −0.409")
check(close(sigmoid(np.array([2.0, 1.0]) @ w + b), 0.886), "the model is now more confident: p = 0.886 (was 0.818)")

# %% See it: train it on two clouds of points
n_half = 100
X_lr = np.vstack([rng.normal([-1, -1], 0.8, (n_half, 2)), rng.normal([1.5, 1], 0.8, (n_half, 2))])
y_lr = np.r_[np.zeros(n_half), np.ones(n_half)]
w, b = np.zeros(2), 0.0
for step in range(300):
    w, b = logistic_step(w, b, X_lr, y_lr, eta=0.5)
acc = ((sigmoid(X_lr @ w + b) > 0.5) == y_lr).mean()
print(f"w = {np.round(w, 2)}, b = {b:.2f}, training accuracy = {acc:.0%}")
check(acc > 0.9, "logistic regression separates the two clouds (> 90% right)")

plt.figure(figsize=(5, 4.5))
plt.scatter(X_lr[:, 0], X_lr[:, 1], c=y_lr, cmap="coolwarm", s=12)
xs = np.linspace(-4, 4, 2)
plt.plot(xs, -(w[0] * xs + b) / w[1], "k--", label="boundary: w·x + b = 0 (p = 0.5)")
plt.ylim(-4, 4); plt.legend(); plt.title("Logistic regression draws a straight line"); plt.tight_layout(); show()


# %% TODO 2: one round of gradient boosting   (Lesson 8.1b)
def fit_stump(x, r):
    """Ready-made: the best one-question tree ("x < threshold?") for targets r. Returns its predictions:
    the average r on each side of the best threshold (best = smallest squared error)."""
    best_sse, best_pred = np.inf, None
    xs = np.sort(np.unique(x))
    for t in (xs[:-1] + xs[1:]) / 2:  # try a cut halfway between every two neighbouring x values
        left = x < t
        pred = np.where(left, r[left].mean(), r[~left].mean())
        sse = ((r - pred) ** 2).sum()
        if sse < best_sse:
            best_sse, best_pred = sse, pred
    return best_pred


def boost_round(x, y, F, eta):
    """F = the current predictions. Returns the improved predictions F + eta · (stump fitted to the residuals)."""
    # TODO 2: 1. residuals = y − F (what the model still gets wrong)
    #         2. h = fit_stump(x, residuals)
    #         3. return F + eta · h
    raise NotImplementedError("TODO 2: boost_round")


x_h = np.array([1.0, 2.0, 3.0, 4.0])  # house sizes
y_h = np.array([1.0, 3.0, 4.0, 8.0])  # prices
F = np.full(4, y_h.mean())            # round 0: predict the average, 4, for every house
check(close(((y_h - F) ** 2).mean(), 6.5), "round 0: MSE = 6.5")
F = boost_round(x_h, y_h, F, eta=0.5)
check(close(F, [3.333, 3.333, 3.333, 6.0]) and close(((y_h - F) ** 2).mean(), 2.5), "round 1: F = [3.33, 3.33, 3.33, 6], MSE = 2.5")
F = boost_round(x_h, y_h, F, eta=0.5)
check(close(((y_h - F) ** 2).mean(), 1.139), "round 2: MSE = 1.14")
F = boost_round(x_h, y_h, F, eta=0.5)
check(close(((y_h - F) ** 2).mean(), 0.470), "round 3: MSE = 0.47. Weak stumps, strong sum (XGBoost does this with bigger trees)")


# %% TODO 3: a numerically safe softmax via log-sum-exp   (Lessons 8.2 and 8.5)
def logsumexp(z, axis=-1):
    """ln(Σ e^z), computed safely: pull out the biggest number first so e^(...) never overflows."""
    z = np.asarray(z, dtype=float)
    m = z.max(axis=axis, keepdims=True)
    # TODO 3a: return m + ln( Σ e^(z − m) ). Keep keepdims=True in the sum, like the line above.
    raise NotImplementedError("TODO 3a: logsumexp")


def softmax(z, axis=-1):
    """Probabilities e^z / Σ e^z, which is the same as e^(z − logsumexp(z))."""
    z = np.asarray(z, dtype=float)
    # TODO 3b: return np.exp(z − logsumexp(z, axis))
    raise NotImplementedError("TODO 3b: softmax")


check(close(logsumexp([1000.0, 999.0]), 1000.313), "log-sum-exp of [1000, 999] = 1000.313, no overflow")
check(close(softmax([2.0, 1.0, 0.0]), [0.665, 0.245, 0.090]), "softmax([2, 1, 0]) = [0.665, 0.245, 0.090]")
check(close(softmax([1000.0, 999.0]), [0.731, 0.269]), "huge logits are fine: [0.731, 0.269]")
check(close(softmax([[1.0, 1.0], [0.0, 50.0]], axis=1), [[0.5, 0.5], [0.0, 1.0]]), "works row by row on a matrix")

# %% See it: the naive formula breaks, and the cross-entropy gradient is just p − y
z = np.array([1000.0, 999.0])
with np.errstate(over="ignore", invalid="ignore"):  # hide NumPy's overflow warning, we want to see the nan
    print("naive e^z / Σ e^z on [1000, 999]:", np.exp(z) / np.exp(z).sum(), "← e^1000 = inf, and inf / inf = nan")
print("safe softmax on [1000, 999]:       ", np.round(softmax(z), 3))

z = np.array([2.0, 1.0, 0.0]); y_onehot = np.array([1.0, 0.0, 0.0])  # the true class is class 0 ("cat")
loss = lambda zz: -np.log(softmax(zz)[0])  # cross-entropy: −ln(probability of the true class)
numeric = np.array([(loss(z + 1e-6 * e) - loss(z - 1e-6 * e)) / 2e-6 for e in np.eye(3)])  # nudge each logit
print("gradient by nudging each logit:", np.round(numeric, 3), "   p − y:", np.round(softmax(z) - y_onehot, 3))
check(close(numeric, softmax(z) - y_onehot, 1e-5), "the softmax + cross-entropy gradient is exactly p − y")


# %% TODO 4: a 2-layer neural network, forward pass   (Lesson 8.3)
# Shapes: X is (n, 2): one point per row.  W1 is (H, 2), b1 is (H,): H hidden neurons.  W2 is (H,), b2 a number.
# Vectors are columns in the lessons (W x). With one point per ROW, "W x for every row" is X @ W1.T.
def forward(X, W1, b1, W2, b2):
    # TODO 4: z1 = X @ W1.T + b1    (each hidden neuron's weighted sum, for every point)
    #         h  = ReLU(z1) = np.maximum(0, z1)    (the bend)
    #         p  = sigmoid(h @ W2 + b2)    (probability of class 1 for every point)
    #         return z1, h, p    (backprop needs z1 and h too)
    raise NotImplementedError("TODO 4: forward")


# Lesson 8.3's hand-made XOR network
W1_x, b1_x = np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.0, -1.0])
W2_x, b2_x = np.array([4.0, -8.0]), -2.0
X_xor = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_xor = np.array([0.0, 1.0, 1.0, 0.0])
_, h, p = forward(X_xor, W1_x, b1_x, W2_x, b2_x)
check(close(h, [[0, 0], [1, 0], [1, 0], [2, 1]]), "hidden layer h = [0,0], [1,0], [1,0], [2,1] (as in the table)")
check(close(p, [0.119, 0.881, 0.881, 0.119]), "p = [0.119, 0.881, 0.881, 0.119]: XOR, which no straight line can do")


# %% TODO 5: backpropagation   (Lesson 8.3)
def backward(X, y, z1, h, p, W2):
    """Gradients of the average log-loss with respect to every weight. Returns gW1, gb1, gW2, gb2."""
    n = len(y)
    d2 = (p - y) / n  # output error for every point (÷ n because the loss is an average)
    # TODO 5: gW2 = hᵀ d2 = h.T @ d2         gb2 = d2.sum()
    #         d1  = (each point's d2 sent back through W2) times ReLU's slope:
    #               np.outer(d2, W2) * (z1 > 0)          (shape (n, H))
    #         gW1 = d1.T @ X                 gb1 = d1.sum(axis=0)
    raise NotImplementedError("TODO 5: backward")
    return gW1, gb1, gW2, gb2


# Lesson 8.3's backward pass by hand: x = [1, 1], target y = 0
x1, y1 = X_xor[3:], y_xor[3:]
z1, h, p = forward(x1, W1_x, b1_x, W2_x, b2_x)
gW1, gb1, gW2, gb2 = backward(x1, y1, z1, h, p, W2_x)
check(close(gW2, [0.238, 0.119]), "∇W₂ = 0.119 × h = [0.238, 0.119]")
check(close(gW1, [[0.477, 0.477], [-0.954, -0.954]]), "∇W₁ = δ₁ xᵀ = [[0.477, 0.477], [−0.954, −0.954]]")


# Gradient check: nudge one weight a tiny bit and see how the loss changes. Must match backprop.
def avg_log_loss(X, y, W1, b1, W2, b2):
    p = forward(X, W1, b1, W2, b2)[2]
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


Wt1, bt1, Wt2, bt2 = rng.standard_normal((3, 2)), rng.standard_normal(3), rng.standard_normal(3), 0.1
z1, h, p = forward(X_xor, Wt1, bt1, Wt2, bt2)
gW1 = backward(X_xor, y_xor, z1, h, p, Wt2)[0]
bumped = Wt1.copy(); bumped[1, 0] += 1e-6
numeric = (avg_log_loss(X_xor, y_xor, bumped, bt1, Wt2, bt2) - avg_log_loss(X_xor, y_xor, Wt1, bt1, Wt2, bt2)) / 1e-6
check(close(numeric, gW1[1, 0], 1e-4), f"backprop's ∂L/∂W₁[1,0] = {gW1[1, 0]:.5f} matches the nudge test {numeric:.5f}")

# %% See it: train the network on two rings (no straight line can separate them)
n_ring = 100
angles = rng.uniform(0, 2 * np.pi, 2 * n_ring)
radius = np.r_[np.full(n_ring, 0.5), np.full(n_ring, 1.0)] + 0.1 * rng.standard_normal(2 * n_ring)
X_c = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
y_c = np.r_[np.zeros(n_ring), np.ones(n_ring)]  # inner ring = 0, outer ring = 1

H = 16
W1, b1 = rng.standard_normal((H, 2)), np.zeros(H)
W2, b2 = 0.5 * rng.standard_normal(H), 0.0
eta = 0.5
for step in range(3000):
    z1, h, p = forward(X_c, W1, b1, W2, b2)
    gW1, gb1, gW2, gb2 = backward(X_c, y_c, z1, h, p, W2)
    W1, b1, W2, b2 = W1 - eta * gW1, b1 - eta * gb1, W2 - eta * gW2, b2 - eta * gb2
    if step % 1000 == 0:
        print(f"step {step:4d}  loss = {avg_log_loss(X_c, y_c, W1, b1, W2, b2):.3f}")
acc = ((forward(X_c, W1, b1, W2, b2)[2] > 0.5) == y_c).mean()
print(f"after 3000 steps: loss = {avg_log_loss(X_c, y_c, W1, b1, W2, b2):.3f}, accuracy = {acc:.0%}")
check(acc > 0.95, "the 2-layer network separates the rings (> 95% right)")

g1, g2 = np.meshgrid(np.linspace(-1.5, 1.5, 150), np.linspace(-1.5, 1.5, 150))
grid_p = forward(np.c_[g1.ravel(), g2.ravel()], W1, b1, W2, b2)[2].reshape(g1.shape)
plt.figure(figsize=(5, 4.5))
plt.contourf(g1, g2, grid_p, levels=20, cmap="coolwarm", alpha=0.4)
plt.contour(g1, g2, grid_p, levels=[0.5], colors="k")
plt.scatter(X_c[:, 0], X_c[:, 1], c=y_c, cmap="coolwarm", s=10)
plt.gca().set_aspect("equal"); plt.title(f"{H} ReLU neurons bend the boundary into a closed loop"); plt.tight_layout(); show()


# %% TODO 6: 2D convolution with loops   (Lesson 8.3b)
def conv2d(img, K):
    """Slide the small filter K over img. Each output number = (window of img) · K, added up.
    ("valid" convolution: the filter never hangs off the edge, so the output is a bit smaller.)"""
    kh, kw = K.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    # TODO 6: for every output position (i, j):
    #             out[i, j] = sum of ( img[i:i + kh, j:j + kw] * K )
    # Hint: two nested for loops over range(out.shape[0]) and range(out.shape[1]).
    raise NotImplementedError("TODO 6: conv2d")
    return out


img = np.zeros((6, 6)); img[:, 3:] = 9  # dark left half, bright right half
edge = np.array([[-1, 0, 1]] * 3)        # "right minus left": a vertical-edge detector
blur = np.ones((3, 3)) / 9               # the average of the 3 × 3 window
check(conv2d(img, edge).shape == (4, 4), "a 6 × 6 image and a 3 × 3 filter give a 4 × 4 feature map")
check(close(conv2d(img, edge), np.tile([0, 27, 27, 0], (4, 1))), "edge filter: every row is [0, 27, 27, 0]")
check(close(conv2d(img, blur), np.tile([0, 3, 6, 9], (4, 1))), "blur filter: the hard 0 → 9 jump becomes a ramp [0, 3, 6, 9]")

# %% See it: one filter finds edges everywhere in a picture (weight sharing)
pic = np.zeros((16, 16)); pic[4:12, 4:12] = 1.0  # a bright square
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for a, (title, im) in zip(ax, [("image", pic), ("vertical-edge filter", conv2d(pic, edge)),
                               ("horizontal-edge filter", conv2d(pic, edge.T))]):
    a.imshow(im, cmap="gray"); a.set_title(title); a.axis("off")
plt.tight_layout(); show()
print("The same 9 weights are reused at every position. A dense layer from 16×16 inputs to a 14×14 map")
print(f"would need {16 * 16 * 14 * 14:,} weights; the convolution needs 9. That's why CNNs (and ViT's patch layer) use them.")


# %% TODO 7: causal self-attention   softmax(QKᵀ/√d) V   (Lesson 8.4)
def causal_attention(Q, K, V):
    """Q, K, V: one row per token. Each token may only look at itself and EARLIER tokens.
    Returns (output, A) where A[i, j] = how much token i attends to token j."""
    n, d = K.shape
    future = np.triu(np.ones((n, n), dtype=bool), k=1)  # True above the diagonal: j > i is the future
    # TODO 7: 1. S = Q Kᵀ / √d        (every query · every key, scaled)
    #         2. set S[future] = -np.inf   (e^(−∞) = 0, so the future gets weight 0)
    #         3. A = softmax(S, axis=1)    (each row adds up to 1; your TODO 3 softmax)
    #         4. return A @ V, A
    raise NotImplementedError("TODO 7: causal_attention")


tokens = ["the", "cat", "sat", "on"]
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [1.0, 1.0]])  # "sat" asks with q = [1, 2], as in Lesson 8.4
K = np.array([[0.0, 1.0], [2.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
out, A = causal_attention(Q, K, V)
check(close(A.sum(axis=1), np.ones(4)), "every row of attention weights adds up to 1")
check(close(A[0], [1, 0, 0, 0]), "the first token can only look at itself")
check(close(np.triu(A, k=1), 0), "no token looks at the future (everything above the diagonal is 0)")
check(close(A[2], [0.178, 0.734, 0.088, 0.0]), "'sat' attends 18% the, 73% cat, 9% sat (Lesson 8.4's numbers)")
check(close(out[2], [0.734, 0.088]), "so the new vector for 'sat' is [0.73, 0.09]: mostly 'cat'")

# %% See it: the attention pattern
print("attention weights (rows = the token asking, columns = the token looked at):")
print("        " + "".join(f"{t:>7}" for t in tokens))
for t, row in zip(tokens, A):
    print(f"{t:>7} " + "".join(f"{a:7.2f}" for a in row))
print("A GPT does exactly this in every layer and every head, with d = 64 or 128 and thousands of tokens.")
print("The causal mask is why it can be trained on every position of a text at once without cheating.")


# %% TODO 8: value iteration, one Bellman sweep   V(s) = max over a of [ r(s, a) + γ·V(next state) ]   (Lesson 8.6)
def bellman_sweep(V, next_state, reward, terminal, gamma):
    """next_state[s, a] = where action a takes you from s;  reward[s, a] = the reward for doing it.
    Terminal states (the end of the game) keep value 0. Returns the new value table."""
    n_states, n_actions = next_state.shape
    V_new = np.zeros(n_states)
    for s in range(n_states):
        if terminal[s]:
            continue
        # TODO 8: V_new[s] = the biggest of  reward[s, a] + gamma * V[next_state[s, a]]  over all actions a.
        # Hint: max(... for a in range(n_actions)). Use the OLD table V on the right-hand side.
        raise NotImplementedError("TODO 8: bellman_sweep")
    return V_new


# The corridor A → B → C → 🏁 from Lesson 8.6. Actions: 0 = left, 1 = right. Reaching 🏁 gives +1.
next_c = np.array([[0, 1], [0, 2], [1, 3], [3, 3]])
reward_c = np.array([[0, 0], [0, 0], [0, 1], [0, 0]], dtype=float)
terminal_c = np.array([False, False, False, True])
V = np.zeros(4)
V = bellman_sweep(V, next_c, reward_c, terminal_c, 0.9)
check(close(V, [0, 0, 1, 0]), "sweep 1: only C knows it's next to the goal")
V = bellman_sweep(V, next_c, reward_c, terminal_c, 0.9)
V = bellman_sweep(V, next_c, reward_c, terminal_c, 0.9)
check(close(V, [0.81, 0.9, 1.0, 0.0]), "sweep 3: V = [0.81, 0.9, 1], value ripples back with γ = 0.9")

# %% See it: a 4 × 4 gridworld. 🏆 top-right gives +1, 🔥 just below it gives −1 (both end the game).
size = 4
goal, pit = (0, 3), (1, 3)
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
arrows = "↑↓←→"
n_s = size * size
next_g = np.zeros((n_s, 4), dtype=int)
reward_g = np.zeros((n_s, 4))
terminal_g = np.zeros(n_s, dtype=bool)
for r in range(size):
    for c in range(size):
        s = r * size + c
        terminal_g[s] = (r, c) in (goal, pit)
        for a, (dr, dc) in enumerate(moves):
            nr, nc = min(max(r + dr, 0), size - 1), min(max(c + dc, 0), size - 1)  # bumping a wall = stay put
            next_g[s, a] = nr * size + nc
            reward_g[s, a] = 1.0 if (nr, nc) == goal else (-1.0 if (nr, nc) == pit else 0.0)

V = np.zeros(n_s)
for sweep in range(100):
    V_next = bellman_sweep(V, next_g, reward_g, terminal_g, 0.9)
    if np.abs(V_next - V).max() < 1e-9:  # nothing changes any more: the Bellman equation holds everywhere
        break
    V = V_next
print(f"converged after {sweep} sweeps. Values and the greedy policy:")
for r in range(size):
    row = ""
    for c in range(size):
        s = r * size + c
        if (r, c) == goal:
            row += "    🏆   "
        elif (r, c) == pit:
            row += "    🔥   "
        else:
            best_a = int(np.argmax(reward_g[s] + 0.9 * V[next_g[s]]))
            row += f" {V[s]:5.3f} {arrows[best_a]} "
    print(row)
check(close(V[3 * size + 0], 0.9 ** 5), "bottom-left is 6 moves from 🏆, so V = 0.9⁵ = 0.590")
check(V[1 * size + 2] > 0, "the cell next to 🔥 still has positive value: the best move avoids the fire")

# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Shrink the network to H = 2 hidden neurons and retrain on the rings. Why can't
#    2 ReLU bends draw a closed loop? What is the smallest H that gets above 95%?
# 2. In the attention cell, remove the causal mask (skip step 2). Which rows of A
#    change, and why would that be cheating when training a model to predict the next token?
# 3. In the gridworld, add a cost of −0.04 for every move that doesn't end the game,
#    and try γ = 0.5. How do the values and arrows change? (Lesson 8.6's playground has both knobs.)

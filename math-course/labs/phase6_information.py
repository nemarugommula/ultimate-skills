# %% [markdown]
# # Lab 6: Information theory from scratch 📡
#
# You'll code the whole of Phase 6 with NumPy: surprise in bits, entropy, the
# cross-entropy loss and perplexity every LLM training log prints, a tiny
# character-level language model (a "bigram" model) trained on a short text,
# KL divergence in both directions, mutual information from a table, and
# CLIP's matching-game loss (InfoNCE).
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase6_information.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 6.1 (surprise), 6.2 (entropy),
# 6.3 (cross-entropy & perplexity), 6.4 (KL), 6.5 (mutual information, InfoNCE)

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


# %% TODO 1: surprise in bits   surprise = −log₂ p   (Lesson 6.1)
def surprise_bits(p):
    """How surprised you are (in bits) when something with probability p happens."""
    # TODO 1: return minus the base-2 log of p.
    # Hint: np.log2 works on single numbers and on whole arrays.
    raise NotImplementedError("TODO 1: surprise_bits")


check(close(surprise_bits(0.5), 1.0), "a fair coin flip is 1 bit of surprise")
check(close(surprise_bits(0.25), 2.0), "probability 1/4 is 2 bits")
check(close(surprise_bits(1 / 1024), 10.0), "1 in 1024 is 10 bits (ten halvings)")
check(close(surprise_bits(np.array([0.5, 0.25, 0.001])), [1.0, 2.0, 9.966]), "works on arrays too")

# %% See it: the LLM's surprise at the next word (Lesson 6.1's worked example)
# The model's guesses after "the cat sat on the ...". The real text decides the surprise.
guesses = {"mat": 0.25, "floor": 0.25, "sofa": 0.125, "moon": 0.001}
for word, p in guesses.items():
    bits = surprise_bits(p)
    nats = bits * np.log(2)  # PyTorch losses use nats (natural log): 1 bit = ln 2 ≈ 0.693 nats
    print(f"real next word = {word:<6} p = {p:<6} surprise = {bits:5.2f} bits = {nats:5.2f} nats")
print("A surprised model gets a big loss, so it learns the most from that token.")


# %% TODO 2: entropy = average surprise   H(p) = Σ p · (−log₂ p)   (Lesson 6.2)
def entropy_bits(p):
    """Entropy of a probability distribution p (a list or array that sums to 1), in bits."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # outcomes with probability 0 never happen, so they add nothing (0 · log 0 counts as 0)
    # TODO 2: multiply each probability by its surprise, then add them all up.
    # Hint: use surprise_bits(p) from TODO 1 and .sum()
    raise NotImplementedError("TODO 2: entropy_bits")


check(close(entropy_bits([0.5, 0.5]), 1.0), "fair coin: H = 1 bit")
check(close(entropy_bits([1 / 6] * 6), np.log2(6)), "fair die: H = log₂ 6 ≈ 2.585 bits (the maximum for 6 outcomes)")
check(close(entropy_bits([0.5, 0.25, 0.125, 0.125]), 1.75), "the A/B/C/D alphabet: H = 1.75 bits")
check(close(entropy_bits([1.0, 0.0, 0.0]), 0.0), "a sure thing has 0 entropy")

# %% See it: temperature changes the entropy of an LLM's next-token choice
# An LLM turns scores (logits) into probabilities with softmax(z / T). T is the "temperature" knob.
logits = np.array([2.0, 1.0, 0.0, -1.0])
for T in [0.25, 0.5, 1.0, 2.0, 10.0]:
    e = np.exp(logits / T - (logits / T).max())  # subtract the max so exp never overflows (Lesson 8.5)
    probs = e / e.sum()
    print(f"T = {T:<5} probs = {np.round(probs, 3)}  entropy = {entropy_bits(probs):.3f} bits  (max = log₂ 4 = 2)")
print("Low T → low entropy (predictable text). High T → close to 2 bits (random text).")

ps = np.linspace(0.001, 0.999, 200)
plt.figure(figsize=(6, 3.5))
plt.plot(ps, [entropy_bits([p, 1 - p]) for p in ps])
plt.xlabel("P(heads)"); plt.ylabel("entropy H (bits)"); plt.title("A coin is most uncertain at 50/50")
plt.grid(alpha=0.3); plt.tight_layout(); show()


# %% TODO 3: cross-entropy loss and perplexity   (Lesson 6.3)
def cross_entropy_nats(p_true):
    """p_true[i] = the probability the model gave to the token that really came next.
    Returns the average surprise in nats: mean of −ln p. This is the LLM training loss."""
    p_true = np.asarray(p_true, dtype=float)
    # TODO 3a: return the average of −ln(p_true).
    # Hint: np.log is the natural log (ln); .mean() averages.
    raise NotImplementedError("TODO 3a: cross_entropy_nats")


def perplexity(loss_nats):
    # TODO 3b: perplexity = e^(average loss). Hint: np.exp
    raise NotImplementedError("TODO 3b: perplexity")


loss = cross_entropy_nats([0.6, 0.3, 0.05, 0.9])
check(close(loss, 1.204), "the 4-token sentence from Lesson 6.3: loss = 1.204 nats per token")
check(close(perplexity(loss), 3.333), "perplexity e^1.204 ≈ 3.33 (as torn as choosing among ~3 tokens)")
check(close(perplexity(cross_entropy_nats([0.25] * 8)), 4.0), "always giving 1/4 → perplexity exactly 4")

# %% A tiny language model: guess the next character from the current one (a "bigram" model)
# This is the simplest possible LLM. Real LLMs look at thousands of previous tokens; ours looks at one.
train_text = (
    "the cat sat on the mat. the dog sat on the log. the cat saw the dog. "
    "the dog saw the cat and the cat ran to the mat. a rat sat on a hat. "
    "the rat saw the cat. the cat sat on the rat. no, the cat sat on the hat. "
    "the dog ran to the log and sat. the cat and the dog sat on the mat. "
)
test_text = "the rat sat on the log. the dog saw the hat and the cat ran to the dog."

chars = sorted(set(train_text))  # the vocabulary: every character that appears
V = len(chars)
idx = {c: i for i, c in enumerate(chars)}  # character → row/column number
assert set(test_text) <= set(chars), "the test text may only use characters seen in training"
print(f"vocabulary: {V} characters: {''.join(chars)!r}")


# %% TODO 4: count which character follows which
def count_bigrams(text):
    """counts[i, j] = how many times character chars[j] comes right after character chars[i]."""
    counts = np.zeros((V, V))
    # TODO 4: loop over every pair of neighbours (text[k], text[k + 1]) and add 1 to
    #         counts[row of the first, column of the second].
    # Hint: for k in range(len(text) - 1): ... use idx[...] to turn a character into a number.
    raise NotImplementedError("TODO 4: count_bigrams")
    return counts


counts = count_bigrams(train_text)
check(counts.sum() == len(train_text) - 1, f"a text of {len(train_text)} characters has {len(train_text) - 1} neighbour pairs")
check(counts[idx["t"], idx["h"]] == train_text.count("th"), f"'t' is followed by 'h' {train_text.count('th')} times")
check(counts[idx["."], idx[" "]] == train_text.count(". "), "every full stop here is followed by a space")


# %% TODO 5: turn counts into probabilities (with add-one smoothing)
def bigram_probs(counts):
    """P[i, j] = P(next = chars[j] | current = chars[i]). Every row must add up to 1."""
    # TODO 5: add 1 to every count (so no pair has probability 0: an unseen pair would give
    #         infinite surprise), then divide each row by its own total.
    # Hint: smoothed = counts + 1;  smoothed.sum(axis=1, keepdims=True) gives each row's total.
    raise NotImplementedError("TODO 5: bigram_probs")


P = bigram_probs(counts)
check(close(P.sum(axis=1), np.ones(V)), "every row of P adds up to 1")
check(P.min() > 0, "no probability is exactly 0 (smoothing worked)")
row_t = counts[idx["t"]]
check(close(P[idx["t"], idx["h"]], (row_t[idx["h"]] + 1) / (row_t.sum() + V)), "P(h | t) = (count + 1) / (row total + V)")

# %% See it: how surprised is our model by NEW text? (cross-entropy on the test text, Lesson 6.3)
p_true = [P[idx[test_text[k]], idx[test_text[k + 1]]] for k in range(len(test_text) - 1)]
loss_bigram = cross_entropy_nats(p_true)
loss_uniform = np.log(V)  # a model that knows nothing gives every character 1/V, so each surprise is ln V
print(f"uniform model: loss = {loss_uniform:.3f} nats/char, perplexity = {perplexity(loss_uniform):.1f} (= V)")
print(f"bigram model:  loss = {loss_bigram:.3f} nats/char, perplexity = {perplexity(loss_bigram):.1f}")
print(f"In bits: {loss_bigram / np.log(2):.2f} bits per character instead of {np.log2(V):.2f}.")
check(loss_bigram < loss_uniform - 1.0, "the bigram model is far less surprised than the uniform model")

# Generate text by sampling one character at a time, exactly how an LLM writes (just with a much worse model).
# For writing we use the raw counts (no smoothing), so it only produces pairs it has really seen.
P_raw = counts / counts.sum(axis=1, keepdims=True)
c = "t"
out = c
for _ in range(80):
    c = chars[rng.choice(V, p=P_raw[idx[c]])]  # draw the next character from row c
    out += c
print("sampled text:", out)


# %% TODO 6: KL divergence   D_KL(p‖q) = Σ p · ln(p / q)   (Lesson 6.4)
def kl_nats(p, q):
    """How many extra nats you pay, on average, for believing q when the truth is p."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    keep = p > 0  # terms with p = 0 contribute 0
    p, q = p[keep], q[keep]
    # TODO 6: return the sum of p * ln(p / q).
    raise NotImplementedError("TODO 6: kl_nats")


check(close(kl_nats([0.9, 0.1], [0.5, 0.5]), 0.368), "bent coin vs fair coin: D_KL(p‖q) = 0.368 nats")
check(close(kl_nats([0.5, 0.5], [0.9, 0.1]), 0.511), "the other way round: D_KL(q‖p) = 0.511 nats. Not symmetric!")
check(close(kl_nats([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0), "KL of a distribution with itself is 0")
check(close(kl_nats([0.9, 0.1], [0.5, 0.5]) / np.log(2), 0.531), "in bits (÷ ln 2): 0.531, as in the worked example")
# Cross-entropy = entropy + KL. So lowering the LLM loss = shrinking KL(data‖model).
pd, qm = np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.3, 0.2])
check(close(-(pd * np.log(qm)).sum(), entropy_bits(pd) * np.log(2) + kl_nats(pd, qm)), "H(p, q) = H(p) + D_KL(p‖q)")

# %% See it: forward KL covers, reverse KL picks one mode (Lesson 6.4)
# The truth p has two bumps. We may only use ONE bump q. Which one is "closest"? It depends on the direction.
x = np.linspace(-6, 6, 241)


def bump(center, width):
    b = np.exp(-0.5 * ((x - center) / width) ** 2)
    return b / b.sum()  # a bell shape on the grid, normalized to sum to 1


p_two = 0.5 * bump(-2, 0.7) + 0.5 * bump(2, 0.7)
q_wide = bump(0, 2.2)    # one wide bump that covers both
q_right = bump(2, 0.7)   # one narrow bump sitting on the right mode
for name, q in [("wide, covers both", q_wide), ("narrow, on one mode", q_right)]:
    print(f"q = {name:<20} forward D_KL(p‖q) = {kl_nats(p_two, q):6.2f}   reverse D_KL(q‖p) = {kl_nats(q, p_two):6.2f}")
print("Forward KL (cross-entropy training) prefers the wide q: it hates missing any of p's mass.")
print("Reverse KL (used in RLHF's penalty and in variational inference) prefers one mode: it hates putting mass where p has none.")
check(kl_nats(p_two, q_wide) < kl_nats(p_two, q_right), "forward KL prefers the covering q")
check(kl_nats(q_right, p_two) < kl_nats(q_wide, p_two), "reverse KL prefers the single-mode q")

plt.figure(figsize=(7, 3.5))
plt.plot(x, p_two, label="truth p (two modes)", lw=2)
plt.plot(x, q_wide, "--", label="q wide (forward KL's pick)")
plt.plot(x, q_right, ":", label="q on one mode (reverse KL's pick)")
plt.legend(); plt.title("Which single bump is closest to p? It depends on the direction of KL")
plt.grid(alpha=0.3); plt.tight_layout(); show()


# %% TODO 7: mutual information from a 2×2 table   I(X ; Y) = Σ P(x,y) · log₂[ P(x,y) / (P(x) P(y)) ]   (Lesson 6.5)
def mutual_info_bits(joint):
    """joint[i, j] = P(X = i and Y = j). How many bits knowing X tells you about Y."""
    joint = np.asarray(joint, dtype=float)
    px = joint.sum(axis=1)  # P(X = i): add across each row
    py = joint.sum(axis=0)  # P(Y = j): add down each column
    total = 0.0
    # TODO 7: for every cell (i, j) with joint[i, j] > 0, add
    #         joint[i, j] * log₂( joint[i, j] / (px[i] * py[j]) ) to total. Then return total.
    # Hint: two nested loops: for i in range(joint.shape[0]): for j in range(joint.shape[1]): ...
    raise NotImplementedError("TODO 7: mutual_info_bits")
    return total


# The spam example from Lesson 6.5: 10 emails. Rows: contains "FREE"? (yes, no). Columns: (spam, normal).
spam_table = np.array([[4, 1],
                       [1, 4]]) / 10
check(close(mutual_info_bits(spam_table), 0.278), "'contains FREE' tells you 0.278 bits about spam (= the tree's information gain)")
check(close(mutual_info_bits([[0.25, 0.25], [0.25, 0.25]]), 0.0), "independent → 0 bits")
check(close(mutual_info_bits([[0.5, 0.0], [0.0, 0.5]]), 1.0), "X decides Y completely → 1 bit (all of H(Y))")
# Same number, the decision-tree way: I = H(Y) − H(Y | X)
h_y = entropy_bits(spam_table.sum(axis=0))
h_y_given_x = 0.5 * entropy_bits([0.8, 0.2]) + 0.5 * entropy_bits([0.2, 0.8])
check(close(h_y - h_y_given_x, mutual_info_bits(spam_table)), "I(X ; Y) = H(Y) − H(Y | X) = 1 − 0.722")


# %% TODO 8: InfoNCE, CLIP's matching-game loss   (Lesson 6.5)
def infonce(S, tau):
    """S[i, j] = cosine similarity of image i and caption j. The right caption for image i is caption i.
    Loss = average over rows of −ln softmax(S[i] / tau)[i]."""
    logits = S / tau
    logits = logits - logits.max(axis=1, keepdims=True)  # subtract each row's max (safe exp, same softmax)
    # TODO 8: 1. probs = softmax of each row: np.exp(logits) divided by its row sum (keepdims=True)
    #         2. take the diagonal (probability of the correct caption for each image): np.diag(probs)
    #         3. return the average of −ln of those diagonal probabilities
    raise NotImplementedError("TODO 8: infonce")


S = np.array([[0.9, 0.1, 0.0, 0.2],
              [0.1, 0.8, 0.3, 0.0],
              [0.0, 0.2, 0.7, 0.1],
              [0.3, 0.0, 0.1, 0.9]])
check(close(infonce(np.zeros((4, 4)), 0.1), np.log(4)), "blind guessing among 4 captions: loss = ln 4 = 1.386 nats")
check(close(infonce(S, 1.0), 0.911), "with τ = 1 the loss is still high: 0.911 nats (close to blind guessing)")
check(close(infonce(S, 0.1), 0.0056), "with τ = 0.1 the same similarities give only 0.0056 nats")

# %% See it: what the loss proves about mutual information
for tau in [1.0, 0.5, 0.1]:
    L = infonce(S, tau)
    print(f"τ = {tau:<4} InfoNCE loss = {L:.3f} nats  →  I(image ; caption) ≥ ln 4 − loss = {np.log(4) - L:.3f} nats")
print("The bound can never exceed ln 4 ≈ 1.386 with 4 pairs. CLIP used batches of 32,768 to raise that ceiling,")
print("and learns τ, because cosines between −1 and 1 are too timid for softmax without a small temperature.")

# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Add more training text to `train_text` (write a few more sentences in the same style).
#    Does the bigram model's test perplexity go down? Can it ever get below the entropy of
#    the real next-character distribution? (Hint: cross-entropy = entropy + KL.)
# 2. Replace add-one smoothing with add-0.01. Is the test loss better or worse? Why might
#    a pair that never appeared in training be dangerous with too little smoothing?
# 3. In the InfoNCE cell, make the off-diagonal similarities bigger (harder negatives).
#    What happens to the loss, and to the bound on mutual information?

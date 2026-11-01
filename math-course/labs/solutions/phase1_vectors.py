# %% [markdown]
# # Lab 1: Vectors & embeddings from scratch ➡️
#
# You'll write the core vector tools with plain loops first (length, add, scale, dot product,
# cosine similarity), then use them on a small hand-made table of word vectors: a tiny semantic
# search engine, the famous king − man + woman, L1 vs L2 distance, a k-nearest-neighbours
# classifier, and normalizing so one matrix product searches everything at once.
#
# How to run: open this file in VS Code and click "Run Cell" above each `# %%`,
# or run the whole thing with `python3 phase1_vectors.py`. Setup (once):
# `pip install -r requirements.txt`.
# ✅ SOLUTION FILE: every TODO is filled in. Try labs/phase1_vectors.py yourself first!
# Lessons to reread if stuck: 1.1 (vectors, length), 1.2 (add & scale), 1.3 (dot product,
# cosine), 1.4 (norms, k-NN), 1.5 (embeddings).

# %%
import math
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


# %% TODO 1: the length of a vector  ‖v‖ = √(v₁² + v₂² + …)  (Pythagoras)
def length(v):
    # TODO: square every number, add the squares up, take the square root.
    # Hint:
    #   total = 0
    #   for x in v:
    #       total = total + x * x
    #   return math.sqrt(total)
    total = 0
    for x in v:
        total = total + x * x
    return math.sqrt(total)


check(close(length([3, 4]), 5), "‖[3, 4]‖ = √(9 + 16) = 5")
check(close(length([1, 2, 2]), 3), "‖[1, 2, 2]‖ = √(1 + 4 + 4) = 3")
v768 = rng.standard_normal(768)  # the size of one GPT-2 token embedding
check(close(length(v768), np.linalg.norm(v768)), "same as np.linalg.norm on a 768-number vector")


# %% TODO 2: add and scale, slot by slot
def add(a, b):
    # TODO: return a new list where slot i is a[i] + b[i].
    # Hint:
    #   out = []
    #   for i in range(len(a)):
    #       out.append(a[i] + b[i])
    out = []
    for i in range(len(a)):
        out.append(a[i] + b[i])
    return out


def scale(s, a):
    # TODO: return a new list where slot i is s * a[i]  (stretch the arrow s times).
    out = []
    for x in a:
        out.append(s * x)
    return out


check(close(add([3, 1], [1, 2]), [4, 3]), "[3, 1] + [1, 2] = [4, 3]")
check(close(scale(2, [3, 1]), [6, 2]), "2 · [3, 1] = [6, 2]")
check(close(add([3, 1], scale(-1, [1, 2])), [2, -1]), "a − b = a + (−1)·b = [2, −1]")
a, b = np.array([3.0, 1.0]), np.array([1.0, 2.0])
check(close(add(a, b), a + b) and close(scale(2, a), 2 * a), "NumPy's + and * do exactly this, slot by slot")


# %% See it: the arrows, plus two places add & scale appear in AI
plt.figure(figsize=(4.5, 4.5))
for vec, color, name in [(a, "tab:blue", "a"), (b, "tab:orange", "b"), (a + b, "tab:green", "a + b")]:
    plt.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1, color=color, label=name)
plt.plot([a[0], a[0] + b[0]], [a[1], a[1] + b[1]], "k:", lw=1)  # b drawn again from a's tip
plt.xlim(-1, 5); plt.ylim(-1, 5); plt.gca().set_aspect("equal"); plt.grid(alpha=0.3)
plt.title("tip to tail: a + b"); plt.legend(); plt.tight_layout(); plt.show()

# 1. Mean pooling: a sentence vector = the average of its word vectors
the, cat, sat = [0, 4], [2, 0], [4, 2]
sentence = scale(1 / 3, add(add(the, cat), sat))
print("mean of 'the', 'cat', 'sat' =", np.round(sentence, 3))
check(close(sentence, [2, 2]), "mean pooling: ([0,4] + [2,0] + [4,2]) / 3 = [2, 2]")
# 2. One training step: weights ← weights − 0.1 · gradient  (Phase 3 explains the gradient)
w, g = [2.0, -1.0], [0.5, -1.0]
w = add(w, scale(-0.1, g))
check(close(w, [1.95, -0.9]), "a training step is add + scale: [2, −1] − 0.1·[0.5, −1] = [1.95, −0.9]")


# %% TODO 3: the dot product and cosine similarity
def dot(a, b):
    # TODO: multiply slot by slot, then add everything up:  a·b = a₁b₁ + a₂b₂ + …
    # Hint: start total = 0, loop over i, add a[i] * b[i].
    total = 0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total


def cosine(a, b):
    # TODO: cos = (a·b) / (‖a‖ · ‖b‖). Use your dot() and length().
    #   1 = same direction, 0 = at right angles (unrelated), −1 = opposite.
    return dot(a, b) / (length(a) * length(b))


check(close(dot([3, 4], [4, 3]), 24), "[3, 4]·[4, 3] = 12 + 12 = 24")
check(close(cosine([3, 4], [4, 3]), 0.96), "cosine = 24 / (5 × 5) = 0.96: almost the same direction")
check(close(cosine([1, 0], [0, 1]), 0), "at right angles: cosine 0")
check(close(cosine([1, 2], [-2, -4]), -1), "opposite directions: cosine −1")
check(close(cosine([1, 2], [10, 20]), 1), "length doesn't matter, only direction: cosine 1")


# %% See it: in 768 dimensions, random arrows are almost exactly at right angles
cos_2d = [cosine(rng.standard_normal(2), rng.standard_normal(2)) for _ in range(500)]
cos_768 = [cosine(rng.standard_normal(768), rng.standard_normal(768)) for _ in range(500)]
plt.figure(figsize=(7, 3.5))
plt.hist(cos_2d, bins=40, alpha=0.6, label="2 numbers per vector")
plt.hist(cos_768, bins=40, alpha=0.6, label="768 numbers per vector")
plt.xlabel("cosine between two random vectors"); plt.ylabel("count"); plt.legend()
plt.title("Lots of room: random directions in 768-D barely overlap"); plt.tight_layout(); plt.show()
print(f"typical size of a random cosine: 2-D {np.mean(np.abs(cos_2d)):.2f}, 768-D {np.mean(np.abs(cos_768)):.3f}")
print("So a cosine of 0.5 between two embeddings is a strong signal, not a coincidence.")
check(np.mean(np.abs(cos_768)) < 0.05, "random 768-D vectors have cosines close to 0")


# %% A hand-made word-vector table (real embeddings are learned, with hundreds of numbers)
# Each word gets 6 scores from 0 to 5, one per feature:
FEATURES = ["royal", "male", "female", "young", "animal", "food"]
WORD_TABLE = {
    "king":     [5, 4, 0, 0, 0, 0],
    "queen":    [5, 0, 4, 0, 0, 0],
    "prince":   [4, 4, 0, 3, 0, 0],
    "princess": [4, 0, 4, 3, 0, 0],
    "man":      [0, 4, 0, 1, 0, 0],
    "woman":    [0, 0, 4, 1, 0, 0],
    "boy":      [0, 4, 0, 4, 0, 0],
    "girl":     [0, 0, 4, 4, 0, 0],
    "cat":      [0, 0, 0, 0, 5, 0],
    "kitten":   [0, 0, 0, 4, 5, 0],
    "lion":     [2, 1, 0, 0, 5, 0],
    "pizza":    [0, 0, 0, 0, 0, 5],
    "apple":    [0, 0, 0, 1, 0, 4],
}
WORDS = list(WORD_TABLE)                                   # the 13 words, in order
TABLE = np.array([WORD_TABLE[w] for w in WORDS], float)    # shape (13, 6): one row per word


def vec(word):
    """Look up a word's vector: just read its row (that's all an embedding layer does)."""
    return TABLE[WORDS.index(word)]


print("table shape:", TABLE.shape, "→", len(WORDS), "words ×", len(FEATURES), "numbers")
check(close(dot(vec("king"), vec("queen")), 25), "king·queen = 5×5 = 25 (they only share 'royal')")
check(close(cosine(vec("king"), vec("queen")), 25 / 41), "cos(king, queen) = 25 / (√41 · √41) = 0.610")


# %% TODO 4: a tiny semantic search: the top-k most similar words
def top_k(query, words, table, k=3, exclude=()):
    # TODO: make two things:
    #   scores: the cosine of query with every row of table (a list, one number per word)
    #   order:  the row numbers sorted from the HIGHEST score to the lowest
    # Hint:
    #   scores = [cosine(query, row) for row in table]
    #   order = np.argsort(scores)[::-1]     # argsort sorts low → high; [::-1] flips it
    scores = [cosine(query, row) for row in table]
    order = np.argsort(scores)[::-1]
    # (ready-made) keep the first k words that aren't in `exclude`
    result = []
    for i in order:
        if words[i] not in exclude:
            result.append((words[i], round(float(scores[i]), 3)))
    return result[:k]


print("most like 'cat':   ", top_k(vec("cat"), WORDS, TABLE))
print("most like 'king':  ", top_k(vec("king"), WORDS, TABLE))
check([w for w, s in top_k(vec("cat"), WORDS, TABLE)] == ["cat", "lion", "kitten"], "cat → cat, lion, kitten")
check(top_k(vec("king"), WORDS, TABLE, k=2) == [("king", 1.0), ("prince", 0.878)], "king → king (1.0), prince (0.878)")
young_animal = [0, 0, 0, 1, 1, 0]   # a "question" that is not a word: 'young' + 'animal'
print("search 'young animal':", top_k(young_animal, WORDS, TABLE))
check(top_k(young_animal, WORDS, TABLE, k=1)[0][0] == "kitten", "searching for 'young animal' finds kitten")
print("This is how RAG finds documents for a chatbot: embed the question, rank everything by cosine.")


# %% TODO 5: word arithmetic: king − man + woman = ?
def analogy(a, minus, plus):
    # TODO: 1. target = vec(a) − vec(minus) + vec(plus)   (NumPy arrays: plain - and + work)
    #       2. return the single closest word that isn't one of the three inputs
    # Hint: top_k(target, WORDS, TABLE, k=1, exclude=(a, minus, plus)) gives [(word, score)];
    #       [0][0] picks out the word.
    target = vec(a) - vec(minus) + vec(plus)
    return top_k(target, WORDS, TABLE, k=1, exclude=(a, minus, plus))[0][0]


target = vec("king") - vec("man") + vec("woman")
print("king − man + woman =", target, "→", analogy("king", "man", "woman"))
check(close(target, [5, 0, 4, 0, 0, 0]), "[5,4,0,0,0,0] − [0,4,0,1,0,0] + [0,0,4,1,0,0] = [5,0,4,0,0,0]")
check(analogy("king", "man", "woman") == "queen", "king − man + woman → queen")
check(analogy("prince", "boy", "girl") == "princess", "prince − boy + girl → princess")
check(analogy("kitten", "boy", "man") == "cat", "kitten − boy + man → cat (grow up!)")


# %% TODO 6: two ways to measure distance: L1 ("taxi") and L2 ("straight line")
def l1_dist(a, b):
    # TODO: ‖a − b‖₁ = |a₁ − b₁| + |a₂ − b₂| + …   (blocks a taxi drives)
    # Hint: abs(x) gives the size of x without its sign. Loop over i and add abs(a[i] − b[i]).
    total = 0
    for i in range(len(a)):
        total = total + abs(a[i] - b[i])
    return total


def l2_dist(a, b):
    # TODO: ‖a − b‖₂ = √((a₁ − b₁)² + (a₂ − b₂)² + …)   (as the crow flies)
    # Hint: it's the length of the difference: length(add(a, scale(-1, b)))
    return length(add(a, scale(-1, b)))


check(close(l1_dist([4, 6], [1, 2]), 7), "L1 from [4, 6] to [1, 2]: 3 + 4 = 7 blocks")
check(close(l2_dist([4, 6], [1, 2]), 5), "L2 from [4, 6] to [1, 2]: √(9 + 16) = 5")
check(close(l2_dist([4, 6], [1, 2]), np.linalg.norm(np.array([4, 6]) - np.array([1, 2]))), "matches np.linalg.norm(a − b)")
check(l2_dist([1, 1], [2, 2]) <= l1_dist([1, 1], [2, 2]), "the straight line is never longer than the taxi route")


# %% TODO 7: k-nearest neighbours: label a new point by a vote of its k closest neighbours
FRUIT = np.array([[1, 8], [2, 7], [1, 6], [3, 9],     # apples: [sourness, sweetness]
                  [8, 2], [7, 1], [9, 3], [6, 3]], float)  # lemons
FRUIT_LABELS = ["apple"] * 4 + ["lemon"] * 4


def knn_predict(query, X, labels, k=3, dist=l2_dist):
    # TODO: 1. d = the distance from query to every row of X (use the dist function you're given)
    #       2. nearest = the row numbers of the k smallest distances
    #       3. return the label that appears most often among those k rows
    # Hint:
    #   d = [dist(query, x) for x in X]
    #   nearest = np.argsort(d)[:k]
    #   votes = [labels[i] for i in nearest]
    #   return max(set(votes), key=votes.count)     # the most common vote
    d = [dist(query, x) for x in X]
    nearest = np.argsort(d)[:k]
    votes = [labels[i] for i in nearest]
    return max(set(votes), key=votes.count)


check(knn_predict([2, 8], FRUIT, FRUIT_LABELS) == "apple", "sweet, not sour [2, 8] → apple")
check(knn_predict([7, 2], FRUIT, FRUIT_LABELS) == "lemon", "sour, not sweet [7, 2] → lemon")
check(knn_predict([7, 2], FRUIT, FRUIT_LABELS, k=1, dist=l1_dist) == "lemon", "works with the L1 distance too")


# %% See it: the norm you choose can change the answer
two = np.array([[6, 6], [4, 7]], float)  # one apple, one lemon
two_labels = ["apple", "lemon"]
mystery = [4, 4]
print("mystery [4, 4] → apple [6, 6]: L2 =", round(l2_dist(mystery, two[0]), 2), " L1 =", l1_dist(mystery, two[0]))
print("mystery [4, 4] → lemon [4, 7]: L2 =", round(l2_dist(mystery, two[1]), 2), " L1 =", l1_dist(mystery, two[1]))
check(knn_predict(mystery, two, two_labels, k=1, dist=l2_dist) == "apple", "by L2 (straight line) the nearest is the apple (2.83 < 3)")
check(knn_predict(mystery, two, two_labels, k=1, dist=l1_dist) == "lemon", "by L1 (taxi) the nearest is the lemon (3 < 4)")

plt.figure(figsize=(4.5, 4.5))
plt.scatter(FRUIT[:4, 0], FRUIT[:4, 1], c="tab:red", s=80, label="apple")
plt.scatter(FRUIT[4:, 0], FRUIT[4:, 1], c="gold", s=80, edgecolors="k", label="lemon")
grid = np.array([[sx, sw] for sx in np.linspace(0, 10, 41) for sw in np.linspace(0, 10, 41)])
guess = [knn_predict(p, FRUIT, FRUIT_LABELS) for p in grid]
plt.scatter(grid[:, 0], grid[:, 1], c=["tab:red" if gl == "apple" else "gold" for gl in guess], s=4, alpha=0.4)
plt.xlabel("sourness"); plt.ylabel("sweetness"); plt.title("3-NN: what every spot would be called")
plt.legend(); plt.tight_layout(); plt.show()


# %% TODO 8: normalize: shrink or stretch a vector to length 1 (same direction)
def normalize(v):
    # TODO: divide every number of v by v's length.
    # Hint: np.array(v, dtype=float) / length(v)
    return np.array(v, dtype=float) / length(v)


check(close(normalize([3, -4]), [0.6, -0.8]), "[3, −4] / 5 = [0.6, −0.8]")
check(close(length(normalize(v768)), 1), "any vector, normalized, has length 1")
check(close(dot(normalize([3, 4]), normalize([4, 3])), cosine([3, 4], [4, 3])), "for length-1 vectors, dot product = cosine")


# %% See it: normalize every row once, then ONE matrix × vector scores every word at once
UNIT = np.array([normalize(row) for row in TABLE])  # shape (13, 6), every row has length 1
scores_all = UNIT @ normalize(vec("cat"))            # 13 dot products in one line (Phase 2)
best3 = np.argsort(scores_all)[::-1][:3]
print("one matrix product, top 3 for 'cat':", [(WORDS[i], round(float(scores_all[i]), 3)) for i in best3])
check([WORDS[i] for i in best3] == [w for w, s in top_k(vec("cat"), WORDS, TABLE)], "same answer as your top_k loop")
print("Vector databases store normalized embeddings for exactly this reason: search = one big matrix product.")


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Add three words of your own to WORD_TABLE (say "puppy", "dog", "burger"). Pick their 6 scores
#    so that kitten − cat + dog → puppy works. Does anything else break?
# 2. Find another analogy that works in this table, and one that fails. Why does it fail? Which
#    feature is missing from the table?
# 3. In the k-NN plot, try k = 1 and k = 7, and dist=l1_dist. Where does the border between apples
#    and lemons move? Which k would you trust more if one apple were mislabelled as a lemon?

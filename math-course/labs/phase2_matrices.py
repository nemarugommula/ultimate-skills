# %% [markdown]
# # Lab 2: Matrices & transformations from scratch 🧮
#
# You'll multiply a matrix by a vector with your own loop (then see why everyone uses @), build a
# neural-network layer that runs on a whole batch, center data with broadcasting, solve a 2×2
# system with the determinant, compute Google's PageRank by power iteration, compress a picture
# with the SVD, and count how few numbers LoRA needs to fine-tune a big model.
#
# How to run: open this file in VS Code and click "Run Cell" above each `# %%`,
# or run the whole thing with `python3 phase2_matrices.py`. Setup (once):
# `pip install -r requirements.txt`.
# Fill in each TODO until its ✅ checks pass. Solutions are in labs/solutions/.
# Lessons to reread if stuck: 2.1 (matrices), 2.2 (matmul = a layer), 2.3 (shapes, broadcasting),
# 2.4 (determinant, solving, LoRA), 2.5 (eigenvectors, PageRank), 2.6 (SVD).

# %%
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


# %% TODO 1: matrix × vector = one dot product per row
def matvec_loop(A, x):
    # TODO: output number i is (row i of A) · x. Collect them in a list and return it as an array.
    # Hint:
    #   out = []
    #   for i in range(A.shape[0]):          # A.shape[0] = number of rows
    #       out.append(np.dot(A[i], x))      # A[i] is row i
    #   return np.array(out)
    raise NotImplementedError("TODO 1: matvec_loop")


A = np.array([[2, -1], [1, 1]])
x = np.array([3, 2])
check(close(matvec_loop(A, x), [4, 5]), "[[2, −1], [1, 1]] · [3, 2] = [2·3 − 1·2, 1·3 + 1·2] = [4, 5]")
check(close(matvec_loop(A, x), 3 * A[:, 0] + 2 * A[:, 1]), "same as 3·(column 1) + 2·(column 2): columns = where the axes land")
M, v = rng.standard_normal((5, 3)), rng.standard_normal(3)
check(close(matvec_loop(M, v), M @ v), "agrees with NumPy's @ on a random 5×3 matrix")
check(matvec_loop(M, v).shape == (5,), "(5×3) · (3) → 5 numbers: one per row")


# %% See it: a matrix moves the whole plane (here, a 30° turn), and @ is much faster than a loop
th = np.radians(30)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])  # rotate 30° counterclockwise
house = np.array([[0, 0], [2, 0], [2, 1.5], [1, 2.5], [0, 1.5], [0, 0]])  # one corner per row
turned = np.array([matvec_loop(R, corner) for corner in house])
plt.figure(figsize=(4.5, 4.5))
plt.plot(house[:, 0], house[:, 1], "o-", label="house")
plt.plot(turned[:, 0], turned[:, 1], "o-", label="R · house (30° turn)")
plt.gca().set_aspect("equal"); plt.grid(alpha=0.3); plt.legend(); plt.title("A matrix is a machine that moves space")
plt.tight_layout(); plt.show()
check(close(np.linalg.norm(turned[2]), np.linalg.norm(house[2])), "a rotation keeps every length (distance to the corner [2, 1.5] is still 2.5)")

big_A, big_x = rng.standard_normal((2000, 2000)), rng.standard_normal(2000)
y_np = big_A @ big_x                     # a first call "warms up" NumPy's fast math library
t_loop, t_np = [], []
for repeat in range(5):                  # time each way 5 times and keep the fastest
    t0 = time.perf_counter(); y_loop = matvec_loop(big_A, big_x); t_loop.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); y_np = big_A @ big_x; t_np.append(time.perf_counter() - t0)
print(f"2000×2000 matrix × vector: your loop {min(t_loop) * 1000:.2f} ms, @ {min(t_np) * 1000:.2f} ms")
check(close(y_loop, y_np), "same 2,000 numbers either way (@ is usually several times faster)")


# %% TODO 2: a dense neural-network layer on a whole batch:  y = W x + b  for every input
def dense(X, W, b):
    # X holds a batch of inputs, ONE INPUT PER ROW (PyTorch's layout): shape (batch, n_in).
    # W has shape (n_out, n_in), b has n_out numbers.
    # TODO: return one output per row: shape (batch, n_out).
    # Hint: for one input x it's W @ x + b. With inputs in rows the same numbers are
    #       X @ W.T + b   (W.T, "W transpose", flips W's rows and columns: Lesson 2.2).
    raise NotImplementedError("TODO 2: dense")


W = np.array([[1, 2, 0], [0, 1, -1]])
b = np.array([0.5, 0])
x = np.array([3, 1, 2])
check(close(dense(x[None, :], W, b), [[5.5, -1]]), "one input [3, 1, 2]: [1·3 + 2·1 + 0·2 + 0.5, 0·3 + 1·1 − 1·2 + 0] = [5.5, −1]")
X_batch = rng.standard_normal((4, 3))
Y = dense(X_batch, W, b)
check(Y.shape == (4, 2), "a batch of 4 inputs (4×3) → 4 outputs (4×2)")
check(close(Y[2], W @ X_batch[2] + b), "row 2 of the batch result = W @ (input 2) + b")


# %% See it: stacking layers = a neural network, and the most famous shape error
X_img = rng.standard_normal((64, 784))              # 64 flattened 28×28 images
W1, b1 = rng.standard_normal((512, 784)) * 0.05, np.zeros(512)
W2, b2 = rng.standard_normal((10, 512)) * 0.05, np.zeros(10)
h = np.maximum(0, dense(X_img, W1, b1))             # layer 1 + ReLU (Lesson 0.2)
scores = dense(h, W2, b2)                            # layer 2: 10 scores per image
print("shapes:", X_img.shape, "→", h.shape, "→", scores.shape)
print("knobs (parameters):", W1.size + b1.size + W2.size + b2.size)
check(scores.shape == (64, 10), "64 images in, 10 class scores out for each")
check(W1.size + b1.size + W2.size + b2.size == 407050, "784·512 + 512 + 512·10 + 10 = 407,050 parameters")
try:
    X_img @ W2.T                                     # oops: skipped the first layer
except ValueError as err:
    print("the classic mistake →", err)
print("Read shapes like dominoes: (64×784)·(784×512) works because the inside numbers match.")


# %% TODO 3: broadcasting: subtract each column's average from every row
def center(X):
    # TODO: return X minus its column averages, so every column averages 0.
    # Hint: X.mean(axis=0) has one number per column, shape (n_cols,).
    #       X - X.mean(axis=0) "stretches" that row of averages down every row (broadcasting).
    raise NotImplementedError("TODO 3: center")


X3 = np.array([[1, 2], [3, 4], [5, 9]], float)
check(close(center(X3), [[-2, -3], [0, -1], [2, 4]]), "column averages are [3, 5]; subtract them from every row")
check(close(center(X3).mean(axis=0), [0, 0]), "after centering, every column averages 0")
check(center(rng.standard_normal((100, 7))).shape == (100, 7), "the shape doesn't change")


# %% See it: the broadcasting rule (line shapes up from the RIGHT; each pair must match or be 1)
pairs = [((4, 3), (3,)), ((32, 3, 8, 8), (3, 1, 1)), ((5, 1), (1, 6)), ((4, 3), (4,))]
for s1, s2 in pairs:
    try:
        print(f"{str(s1):>15} with {str(s2):<10} → {np.broadcast_shapes(s1, s2)}")
    except ValueError:
        print(f"{str(s1):>15} with {str(s2):<10} → ValueError: 3 vs 4 don't match and neither is 1")
imgs = rng.random((32, 3, 8, 8))                      # (batch, color channels, height, width)
channel_mean = imgs.mean(axis=(0, 2, 3))              # one average per color: shape (3,)
normed = imgs - channel_mean.reshape(3, 1, 1)         # (32,3,8,8) − (3,1,1): per-color centering
check(close(normed.mean(axis=(0, 2, 3)), [0, 0, 0]), "image batches are centered per color channel exactly this way")


# %% TODO 4: the determinant (area change) and solving A x = t
def det2(A):
    # TODO: for A = [[a, b], [c, d]] return a·d − b·c.
    # Hint: a = A[0, 0], b = A[0, 1], c = A[1, 0], d = A[1, 1]
    raise NotImplementedError("TODO 4a: det2")


def solve2(A, t):
    # TODO: return x = A⁻¹ t using the 2×2 inverse recipe:
    #   A⁻¹ = (1 / det) · [[d, −b], [−c, a]]
    # Hint: build inv = np.array([[d, -b], [-c, a]]) / det2(A), then return inv @ t
    raise NotImplementedError("TODO 4b: solve2")


A = np.array([[2, 1], [1, 3]], float)
t = np.array([5, 10], float)
check(close(det2(A), 5), "det [[2, 1], [1, 3]] = 2·3 − 1·1 = 5")
check(close(solve2(A, t), [1, 3]), "2 apples + 1 banana = $5, 1 apple + 3 bananas = $10 → apples $1, bananas $3")
check(close(A @ solve2(A, t), t), "plug the answer back in: A x = t")
check(close(solve2(A, t), np.linalg.solve(A, t)), "matches np.linalg.solve")
check(close(det2(np.array([[1, 2], [2, 4]])), 0), "det [[1, 2], [2, 4]] = 0: the columns point the same way, the plane gets squashed")


# %% See it: |det| is how much the unit square's area gets scaled
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
moved = square @ A.T                                  # every corner times A (corners are rows)
area = 0.5 * abs(sum(moved[i, 0] * moved[i + 1, 1] - moved[i + 1, 0] * moved[i, 1] for i in range(4)))
plt.figure(figsize=(4.5, 4.5))
plt.fill(square[:, 0], square[:, 1], alpha=0.4, label="unit square (area 1)")
plt.fill(moved[:, 0], moved[:, 1], alpha=0.4, label=f"A · square (area {area:.0f})")
plt.gca().set_aspect("equal"); plt.grid(alpha=0.3); plt.legend(); plt.title("det A = 5: areas get 5× bigger")
plt.tight_layout(); plt.show()
check(close(area, det2(A)), "the parallelogram's area (shoelace formula) equals det A = 5")


# %% TODO 5: PageRank, part 1: the link matrix of a 4-page web
PAGES = ["A", "B", "C", "D"]
LINKS = {"A": ["B", "C"], "B": ["C"], "C": ["A"], "D": ["C"]}  # page → the pages it links to


def link_matrix(pages, links):
    # Column j describes page j: a surfer on page j clicks each of its links with equal chance.
    # TODO: start with L = np.zeros((n, n)). For every page j and every page t it links to,
    #       set L[row of t, column j] = 1 / (number of links on page j).
    # Hint:
    #   for j, page in enumerate(pages):
    #       for target in links[page]:
    #           L[pages.index(target), j] = 1 / len(links[page])
    raise NotImplementedError("TODO 5: link_matrix")


L = link_matrix(PAGES, LINKS)
print(L)
check(close(L[:, 0], [0, 0.5, 0.5, 0]), "column A: A links to B and C, half each")
check(close(L[:, 3], [0, 0, 1, 0]), "column D: D's only link goes to C")
check(close(L.sum(axis=0), [1, 1, 1, 1]), "every column adds to 1 (the surfer goes somewhere)")


# %% TODO 6: PageRank, part 2: power iteration  r ← 0.85 · L r + 0.15 / n
def pagerank(L, d=0.85, steps=50):
    # TODO: start with equal ranks r = np.ones(n) / n, then repeat `steps` times:
    #   r = d * (L @ r) + (1 - d) / n
    # (85% of the time the surfer clicks a link, 15% of the time they jump to a random page.)
    raise NotImplementedError("TODO 6: pagerank")


r1 = pagerank(L, steps=1)
check(close(r1[2], 0.569), "after 1 step: r_C = 0.85 × (0.25/2 + 0.25 + 0.25) + 0.15/4 = 0.569")
r = pagerank(L)
print("PageRank:", ", ".join(f"{p} {rank:.3f}" for p, rank in zip(PAGES, r)))
check(close(r, [0.373, 0.196, 0.394, 0.0375]), "final ranks: A 0.373, B 0.196, C 0.394, D 0.0375")
check(close(r.sum(), 1), "the ranks still add to 1 (they're probabilities)")
check(close(pagerank(L, steps=200), r), "more steps change nothing: it has settled")


# %% See it: the settled ranks are the top eigenvector of the "Google matrix"
G = 0.85 * L + 0.15 / 4                                # the same step written as one matrix
vals, vecs = np.linalg.eig(G)
top = vecs[:, np.argmax(vals.real)].real
top = top / top.sum()                                   # scale it to add up to 1
print("eigenvalues of G:", np.round(vals, 3))
check(close(top, r), "power iteration found G's eigenvector with eigenvalue 1: G r = 1 · r")
history = [pagerank(L, steps=s) for s in range(15)]
plt.figure(figsize=(6, 3.5))
plt.plot(history, "o-"); plt.legend(PAGES); plt.xlabel("step"); plt.ylabel("rank")
plt.title("Power iteration settles in a few steps"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
print("C wins: three pages link to it. A is 2nd: its one link comes from the top page, C.")


# %% TODO 7: compress a picture with the SVD: keep only the k biggest "layers"
def rank_k(A, k):
    # A = U Σ Vᵀ, a sum of layers  σ₁·(u₁ v₁ᵀ) + σ₂·(u₂ v₂ᵀ) + …  with σ₁ ≥ σ₂ ≥ …
    # TODO: keep the first k layers: U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    # Hint: U, S, Vt = np.linalg.svd(A, full_matrices=False) gives the three pieces
    #       (S is a plain list of the σ's, biggest first).
    raise NotImplementedError("TODO 7: rank_k")


A = np.array([[3, 0], [4, 5]], float)
check(close(np.linalg.svd(A)[1], [6.708, 2.236]), "σ's of [[3, 0], [4, 5]] are 6.708 and 2.236 (Lesson 2.6)")
check(close(rank_k(A, 2), A), "keeping every layer gives A back exactly")
check(close(rank_k(A, 1), [[1.5, 1.5], [4.5, 4.5]]), "rank 1 keeps the big stretch: [[1.5, 1.5], [4.5, 4.5]]")
check(close(np.linalg.norm(A - rank_k(A, 1)), 2.236), "the error left over is exactly the dropped σ₂ = 2.236")
check(np.linalg.matrix_rank(rank_k(rng.standard_normal((6, 5)), 3)) == 3, "rank_k really gives a rank-k matrix")


# %% See it: a 60×80 synthetic "photo" (shading + rectangle + disk + a little noise)
hgt, wid = 60, 80
yy, xx = np.mgrid[0:hgt, 0:wid]                         # row and column number of every pixel
photo = 0.3 * xx / wid                                   # left-to-right shading
photo = photo + 0.6 * ((yy > 10) & (yy < 30) & (xx > 10) & (xx < 40))   # a rectangle
photo = photo + 0.8 * ((yy - 38) ** 2 + (xx - 58) ** 2 < 14 ** 2)       # a disk
photo = photo + 0.03 * rng.standard_normal((hgt, wid))                  # camera noise

S = np.linalg.svd(photo, compute_uv=False)
ks = [1, 2, 5, 10, 20]
plt.figure(figsize=(12, 2.6))
plt.subplot(1, len(ks) + 1, 1)
plt.imshow(photo, cmap="gray"); plt.axis("off"); plt.title(f"original: {hgt * wid} numbers", fontsize=8)
for i, k in enumerate(ks):
    plt.subplot(1, len(ks) + 1, i + 2)
    plt.imshow(rank_k(photo, k), cmap="gray"); plt.axis("off")
    plt.title(f"rank {k}: {k * (hgt + wid + 1)} numbers", fontsize=8)
plt.tight_layout(); plt.show()

print(f"original: {hgt * wid} numbers")
errors = []
for k in range(1, 41):
    err = np.linalg.norm(photo - rank_k(photo, k)) / np.linalg.norm(photo)   # relative error
    errors.append(err)
    if k in ks:
        print(f"rank {k:>2}: {k * (hgt + wid + 1):>5} numbers, error {100 * err:5.1f}%")
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, 41), errors, "o-", ms=3)
plt.xlabel("rank k (layers kept)"); plt.ylabel("relative error"); plt.yscale("log")
plt.title("Error vs rank: the first few layers do most of the work"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
check(close(errors[4], np.sqrt((S[5:] ** 2).sum()) / np.sqrt((S ** 2).sum())), "error at rank 5 = √(σ₆² + σ₇² + …) / √(all σ²)")
check(errors[9] < 0.1 and 10 * (hgt + wid + 1) < hgt * wid / 3, "rank 10: under 10% error with under a third of the numbers")
print("After about rank 10 the layers left are mostly the noise. PCA (Lesson 2.6) keeps the top layers for the same reason.")


# %% TODO 8: LoRA: how many numbers does a rank-r update need?
def lora_params(d_out, d_in, r):
    # LoRA trains W + B·A, where B is (d_out × r) and A is (r × d_in).
    # TODO: return how many numbers B and A hold together.
    # Hint: a (rows × cols) matrix holds rows · cols numbers.
    raise NotImplementedError("TODO 8: lora_params")


check(lora_params(4096, 4096, 8) == 65536, "d = 4,096, r = 8: 4,096·8 + 8·4,096 = 65,536 (vs 16,777,216 for all of W)")
check(lora_params(10, 6, 2) == 32, "a 10×6 weight with r = 2: 10·2 + 2·6 = 32 (vs 60)")
budget = 32 * 4 * lora_params(4096, 4096, 16)             # the recap's LoRA budget
full = 32 * 4 * 4096 * 4096
print(f"32 layers × 4 matrices: LoRA trains {budget:,} of {full:,} numbers ({100 * budget / full:.2f}%)")
check(budget == 16777216 and full // budget == 128, "the Phase 2 recap: 16,777,216 numbers, 128× fewer")


# %% See it: B·A is a full-size matrix, but its rank is at most r
d, r = 64, 4
B = rng.standard_normal((d, r))
A_ = rng.standard_normal((r, d))
delta_W = B @ A_                                          # (64×4)·(4×64) → 64×64
print("B·A shape:", delta_W.shape, " rank:", np.linalg.matrix_rank(delta_W), " numbers trained:", lora_params(d, d, r), "of", d * d)
check(delta_W.shape == (d, d) and np.linalg.matrix_rank(delta_W) == r, "BA fits on top of W (same shape) and has rank 4")
x = rng.standard_normal(d)
check(close(B @ (A_ @ x), delta_W @ x), "B·(A·x) = (B·A)·x, but the left side is far cheaper to compute")


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Add the link B → A to LINKS (so B links to C and A). Predict first: who gains rank, who loses?
#    Then rerun TODO 5–6. Also try d = 0.5 and d = 0.99: what does the "random jump" protect against?
# 2. Make the photo harder to compress: add a diagonal stripe (pixels where |xx − yy| < 3). How many
#    layers do you need now for 10% error? Why are diagonal lines "expensive" but boxes "cheap"?
# 3. A 7B-parameter model has about 7 billion numbers. If LoRA adds rank-16 adapters only to matrices
#    of size 4,096 × 4,096, and there are 128 of them, what fraction of the model are you training?

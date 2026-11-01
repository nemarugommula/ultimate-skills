# %% [markdown]
# # Lab 2: Make it train 🛠️
#
# You'll build, by hand and in PyTorch, the tools of Phase 2 that make deep networks train:
# per-layer gradient norms in a 30-layer plain vs residual MLP (watch the gradient vanish, then
# survive), gradient clipping by global norm, BatchNorm / LayerNorm / RMSNorm from scratch
# (checked against nn.BatchNorm1d, nn.LayerNorm and nn.RMSNorm), inverted dropout in train vs
# eval mode, the "initial loss ≈ ln K" sanity check, and the overfit-one-batch test that
# catches a forgotten zero_grad().
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase2_make_it_train.py`.
# CPU only, tiny synthetic data: the whole file runs in a few seconds.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 2.1 (vanishing/exploding, clipping), 2.2 (residuals),
# 2.3 (normalization), 2.4 (dropout), 2.6 (debugging)

# %%
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% Setup: a deep MLP, with or without skip connections (Lessons 2.1, 2.2)
class Block(nn.Module):
    """One block: Linear → tanh. With residual=True it returns x + block(x)."""

    def __init__(self, d, residual):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.residual = residual

    def forward(self, x):
        h = torch.tanh(self.lin(x))
        return x + h if self.residual else h


def deep_mlp(depth=30, d=64, residual=False, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(*[Block(d, residual) for _ in range(depth)], nn.Linear(d, 1))


X = torch.randn(256, 64)
Y = torch.randn(256, 1)


# %% TODO 1: per-layer gradient norms (Lessons 2.1 and 2.2)
def grad_norms(model):
    """Run one forward + backward pass of MSE loss on (X, Y) and return a list with the
    gradient norm of every Block's weight matrix, first block first."""
    # TODO: model.zero_grad(); loss = MSE(model(X), Y); loss.backward()
    # then collect m.lin.weight.grad.norm().item() for every m in model that is a Block
    raise NotImplementedError("TODO 1: grad_norms")


plain = grad_norms(deep_mlp(residual=False))
res = grad_norms(deep_mlp(residual=True))
print(f"plain:    block 1 {plain[0]:.1e}   block 30 {plain[-1]:.1e}   ratio {plain[0] / plain[-1]:.1e}")
print(f"residual: block 1 {res[0]:.1e}   block 30 {res[-1]:.1e}   ratio {res[0] / res[-1]:.1e}")
check(len(plain) == 30, "one gradient norm per block (30 blocks)")
check(plain[0] / plain[-1] < 1e-3, "plain stack: block 1 gets < 1/1000 of block 30's gradient (vanishing)")
check(res[0] / res[-1] > 0.1, "residual stack: block 1 keeps > 10% of block 30's gradient (the highway)")

plt.semilogy(range(1, 31), plain, label="plain")
plt.semilogy(range(1, 31), res, label="residual")
plt.xlabel("block"); plt.ylabel("gradient norm (log scale)"); plt.legend(); plt.title("Gradient through 30 blocks")
show()


# %% TODO 2: clip gradients by global norm (Lesson 2.1)
def clip_by_global_norm(grads, max_norm):
    """grads: a list of tensors. Return (clipped list, total norm before clipping).
    Total norm = sqrt(sum of every squared entry of every tensor).
    If total > max_norm, multiply every tensor by max_norm / total; otherwise return them unchanged."""
    # TODO: total = sqrt(sum over tensors of (g ** 2).sum()); use math.sqrt and .item()
    # Hint: if total > max_norm: return [g * (max_norm / total) for g in grads], total
    raise NotImplementedError("TODO 2: clip_by_global_norm")


g_small = [torch.tensor([0.3, 0.4])]
out, n = clip_by_global_norm(g_small, 1.0)
check(abs(n - 0.5) < 1e-6 and torch.allclose(out[0], g_small[0]), "a norm-0.5 gradient is left alone at max_norm 1.0")

g_big = [torch.tensor([30.0, 40.0]), torch.tensor([0.0])]
out, n = clip_by_global_norm(g_big, 1.0)
check(abs(n - 50.0) < 1e-4 and torch.allclose(out[0], torch.tensor([0.6, 0.8])), "(30, 40) has norm 50 and clips to (0.6, 0.8): same direction")

model = deep_mlp(depth=4, residual=True)
nn.functional.mse_loss(model(X), 100 * Y).backward()        # big targets → big gradients
grads = [p.grad.clone() for p in model.parameters()]
mine, my_total = clip_by_global_norm(grads, 1.0)
torch_total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
check(abs(my_total - torch_total.item()) < 1e-3 * my_total, f"total norm matches clip_grad_norm_ ({my_total:.2f})")
check(all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(mine, model.parameters())), "clipped gradients match PyTorch's")


# %% TODO 3: BatchNorm by hand, training mode (Lesson 2.3)
def batch_norm(x, eps=1e-5):
    """x: (batch, features). Normalize each FEATURE (column) with the mean and the biased
    variance of that column over the batch. γ = 1, β = 0."""
    # TODO: mean and biased variance over dim 0 (keepdim=True), then (x − mean) / sqrt(var + eps)
    # Hint: x.var(dim=0, unbiased=False, keepdim=True)
    raise NotImplementedError("TODO 3: batch_norm")


x = torch.randn(16, 6) * 3 + 5
bn = nn.BatchNorm1d(6)                    # fresh: γ = 1, β = 0, training mode
check(torch.allclose(batch_norm(x), bn(x), atol=1e-5), "your BatchNorm matches nn.BatchNorm1d in train mode")
check(torch.allclose(batch_norm(x).mean(0), torch.zeros(6), atol=1e-5), "every feature now has mean 0 over the batch")
check(torch.allclose(batch_norm(torch.tensor([[2.0], [4.0], [6.0], [8.0]])).squeeze(),
                     torch.tensor([-1.3416, -0.4472, 0.4472, 1.3416]), atol=1e-3), "2, 4, 6, 8 → −1.34, −0.45, 0.45, 1.34 (Lesson 2.3)")
pair = batch_norm(torch.tensor([[1.0], [7.0]])).squeeze()
check(torch.allclose(pair, torch.tensor([-1.0, 1.0]), atol=1e-4), "a batch of 2 always normalizes to −1 and +1")


# %% TODO 4: LayerNorm and RMSNorm by hand (Lesson 2.3)
def layer_norm(x, eps=1e-5):
    """Normalize each EXAMPLE (row) over its features: subtract the row mean, divide by sqrt(biased var + eps)."""
    # TODO: like batch_norm, but over the LAST dimension (dim=-1)
    raise NotImplementedError("TODO 4a: layer_norm")


def rms_norm(x, eps=1e-6):
    """No mean subtraction: divide each row by sqrt(mean of its squares + eps)."""
    # TODO: x / sqrt(mean of x² over the last dimension + eps)
    raise NotImplementedError("TODO 4b: rms_norm")


check(torch.allclose(layer_norm(x), nn.LayerNorm(6)(x), atol=1e-5), "your LayerNorm matches nn.LayerNorm")
check(torch.allclose(rms_norm(x), nn.RMSNorm(6, eps=1e-6)(x), atol=1e-5), "your RMSNorm matches nn.RMSNorm")
check(torch.allclose(rms_norm(torch.tensor([[2.0, 4.0, 6.0, 8.0]])), torch.tensor([[0.3651, 0.7303, 1.0954, 1.4606]]), atol=1e-3),
      "RMSNorm of 2, 4, 6, 8: RMS = √30 ≈ 5.48 → 0.37, 0.73, 1.10, 1.46")
one = x[:1]
check(torch.allclose(layer_norm(one), layer_norm(x)[:1], atol=1e-6), "LayerNorm of one example doesn't depend on its batch-mates")


# %% TODO 5: inverted dropout (Lesson 2.4)
def dropout(x, p, train, generator=None):
    """Training: zero each entry with probability p and scale the survivors by 1/(1 − p).
    Eval (train=False): return x unchanged. Use torch.rand(x.shape, generator=generator)."""
    # TODO: if not training (or p == 0) return x
    # keep = (torch.rand(x.shape, generator=generator) >= p).float(); return x * keep / (1 − p)
    raise NotImplementedError("TODO 5: dropout")


gen = torch.Generator().manual_seed(0)
ones = torch.ones(100_000)
d = dropout(ones, 0.3, train=True, generator=gen)
check(abs((d == 0).float().mean().item() - 0.3) < 0.01, "about 30% of entries are dropped at p = 0.3")
check(torch.allclose(d[d > 0], torch.full_like(d[d > 0], 1 / 0.7)), "survivors are scaled by 1/(1 − p) ≈ 1.4286")
check(abs(d.mean().item() - 1.0) < 0.01, "the expected value is preserved (mean ≈ 1.0)")
check(torch.equal(dropout(ones, 0.3, train=False), ones), "eval mode is the identity")


# %% TODO 6: the initial-loss check (Lesson 2.6)
def expected_initial_loss(k):
    """Cross-entropy of a model that gives every one of k classes probability 1/k."""
    # TODO: −ln(1/k) = ln(k)
    raise NotImplementedError("TODO 6: expected_initial_loss")


def fresh_classifier(k, seed=0):
    """A small classifier whose last layer starts tiny, so its first predictions are nearly uniform."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, k))
    nn.init.normal_(model[2].weight, std=1e-3)
    nn.init.zeros_(model[2].bias)
    return model


check(abs(expected_initial_loss(10) - 2.3026) < 1e-3, "ln(10) ≈ 2.303 for 10 classes")
check(abs(expected_initial_loss(1000) - 6.9078) < 1e-3, "ln(1000) ≈ 6.908 for ImageNet's 1,000 classes")
xb, yb = torch.randn(64, 20), torch.randint(0, 10, (64,))
first = nn.functional.cross_entropy(fresh_classifier(10)(xb), yb).item()
print(f"fresh 10-class model: first loss {first:.4f} vs ln 10 = {math.log(10):.4f}")
check(abs(first - expected_initial_loss(10)) < 0.05, "a well-initialized model starts within 0.05 of ln K")


# %% TODO 7: overfit one batch, and catch a bug with it (Lesson 2.6)
def overfit_one_batch(model, x, y, steps=300, lr=0.1, forget_zero_grad=False):
    """Train on the same (x, y) for `steps` plain SGD steps with cross-entropy.
    After every backward(), record the global norm of the gradients stored in .grad.
    If forget_zero_grad is True, skip opt.zero_grad() (the classic bug: gradients pile up).
    Return (final loss, list of recorded gradient norms)."""
    # TODO: opt = torch.optim.SGD(model.parameters(), lr=lr); norms = []
    # repeat `steps` times: (zero_grad unless forget_zero_grad) → loss = cross_entropy(model(x), y)
    #   → loss.backward() → norms.append(sqrt(sum of p.grad² over all parameters).item()) → opt.step()
    # return (cross_entropy(model(x), y).item(), norms)
    raise NotImplementedError("TODO 7: overfit_one_batch")


xs, ys = torch.randn(16, 20), torch.randint(0, 10, (16,))
good, g_norms = overfit_one_batch(fresh_classifier(10, seed=1), xs, ys)
buggy, b_norms = overfit_one_batch(fresh_classifier(10, seed=1), xs, ys, forget_zero_grad=True)
print(f"correct loop:     final loss {good:.4f}   gradient norm first {g_norms[0]:.3f} → last {g_norms[-1]:.3f}")
print(f"forgot zero_grad: final loss {buggy:.4f}   gradient norm first {b_norms[0]:.3f} → last {b_norms[-1]:.3f}")
check(good < 0.05, "a correct loop memorizes 16 examples (loss < 0.05)")
check(g_norms[-1] < 0.1 * g_norms[0], "in a correct loop the gradient shrinks as the batch is memorized")
check(b_norms[-1] > 10 * g_norms[-1], "forgot zero_grad(): .grad keeps piling up (10×+ larger), even though the loss looks fine")
print("Lesson: the loss alone can look perfect. Logging gradient norms (Lesson 2.6) is what exposes this bug.")

print("\n🎉 Lab 2 complete: you can measure, clip, normalize, drop out and debug.")

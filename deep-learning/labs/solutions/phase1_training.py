# %% [markdown]
# # Lab 1: Training a network, piece by piece 🏋️
#
# You'll build (and check) every piece of Phase 1 in PyTorch, on tiny synthetic data, CPU only:
# the gradient of cross-entropy vs MSE on a logit, a hand-written backward pass checked against
# autograd, a complete DataLoader training loop, the effect of the init scale on a 10-layer
# network, SGD vs Adam vs AdamW (and why the "W" matters), a learning-rate sweep, and a
# warmup + cosine schedule with torch.optim.lr_scheduler.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase1_training.py`.
# Fill each TODO until its ✅ checks pass (solutions: labs/solutions/). The whole file runs in well under a minute.
# Lessons to reread if stuck: 1.1 (losses), 1.2 (backprop), 1.3 (the loop), 1.4 (init),
# 1.5 (optimizers), 1.6 (learning rate), 1.7 (batch size)

# %%
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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


def xor_data(n=1000, seed=0):
    """2D points; class 1 when x1 and x2 have the same sign (an XOR pattern)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    y = (x[:, 0] * x[:, 1] > 0).long()
    return x, y


# %% TODO 1: the push on the logit, cross-entropy vs MSE (Lesson 1.1)
def grad_ce_logit(p, y):
    """d(binary cross-entropy)/d(logit) for a sigmoid output p and a 0/1 target y."""
    # TODO: one line. The σ′ cancels out.
    return p - y


def grad_mse_logit(p, y):
    """d((p - y)^2)/d(logit) for a sigmoid output p = σ(z)."""
    # TODO: chain rule: d/dp of the square, times the sigmoid's slope p(1 - p).
    return 2 * (p - y) * p * (1 - p)


z = torch.tensor([-3.89, 0.0, 2.0], requires_grad=True)
y = torch.tensor([1.0, 1.0, 0.0])
p = torch.sigmoid(z)
nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum").backward()
check(torch.allclose(grad_ce_logit(p.detach(), y), z.grad, atol=1e-6), "cross-entropy push = p − y (matches autograd)")
z.grad = None
((torch.sigmoid(z) - y) ** 2).sum().backward()
check(torch.allclose(grad_mse_logit(p.detach(), y), z.grad, atol=1e-6), "MSE push = 2(p − y)·p(1 − p) (matches autograd)")
ratio = (grad_ce_logit(p[0], y[0]) / grad_mse_logit(p[0], y[0])).item()
print(f"confidently wrong (p = {p[0].item():.3f}): cross-entropy pushes {ratio:.1f}× harder than MSE")
check(ratio > 20, "at p ≈ 0.02 cross-entropy's push is over 20× MSE's")


# %% TODO 2: backprop by hand, checked against autograd (Lesson 1.2)
def manual_grads(w1, w2, b, x1=2.0, x2=-1.0, y=1.0):
    """Forward and backward for L = (relu(w1*x1 + w2*x2 + b) - y)^2. Returns (L, dL/dw1, dL/dw2, dL/db)."""
    z = w1 * x1 + w2 * x2 + b
    a = max(0.0, z)
    L = (a - y) ** 2
    # TODO: walk backwards: dL/da = 2(a − y); ReLU gates (1 if z > 0 else 0); add copies; multiply swaps.
    dL_da = 2 * (a - y)
    dL_dz = dL_da * (1.0 if z > 0 else 0.0)
    return L, dL_dz * x1, dL_dz * x2, dL_dz


for w1v, w2v, bv in [(1.5, 0.5, 0.5), (-0.3, -2.0, 1.0), (0.2, 0.1, -3.0)]:
    params = [torch.tensor(v, requires_grad=True) for v in (w1v, w2v, bv)]
    w1t, w2t, bt = params
    Lt = (torch.relu(w1t * 2.0 + w2t * -1.0 + bt) - 1.0) ** 2
    Lt.backward()
    L, g1, g2, gb = manual_grads(w1v, w2v, bv)
    check(abs(L - Lt.item()) < 1e-6 and all(abs(m - p.grad.item()) < 1e-6 for m, p in zip((g1, g2, gb), params)),
          f"hand gradients match autograd at w1={w1v}, w2={w2v}, b={bv}: {g1:.2f}, {g2:.2f}, {gb:.2f}")
check(manual_grads(0.2, 0.1, -3.0)[1:] == (0.0, 0.0, 0.0), "a ReLU that is off passes zero gradient to everything behind it")


# %% TODO 3: a complete training loop with a DataLoader (Lesson 1.3)
def train_epoch(model, loader, opt, loss_fn):
    """One pass over the loader. Returns the average loss over batches."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        # TODO: the five lines: forward, loss, zero_grad, backward, step
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, n = total + loss.item(), n + 1
    return total / n


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


x, y = xor_data(1200)
xtr, ytr, xva, yva = x[:1000], y[:1000], x[1000:], y[1000:]
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
loader = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    first = loss_fn(model(xtr), ytr).item()
print(f"loss before training: {first:.3f} (ln 2 = {math.log(2):.3f})")
check(abs(first - math.log(2)) < 0.1, "the loss at step 0 is about ln K for K = 2 classes")
losses = [train_epoch(model, loader, opt, loss_fn) for _ in range(20)]
acc = accuracy(model, xva, yva)
print(f"epoch 1 loss {losses[0]:.3f} → epoch 20 loss {losses[-1]:.3f}; validation accuracy {acc:.1%}")
check(losses[-1] < losses[0] / 3, "the training loss fell by more than 3×")
check(acc > 0.9, "validation accuracy above 90% on XOR")
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("TODO 3: the loop"); show()


# %% TODO 4: the init scale decides whether a deep net starts alive (Lesson 1.4)
def activation_stds(init, depth=10, width=256, n=1024):
    """Std of the activations after each of `depth` Linear+ReLU layers. init(weight) changes the weight in place,
    or init=None keeps PyTorch's default. Biases are zeroed when init is given."""
    g = torch.Generator().manual_seed(0)
    h = torch.randn(n, width, generator=g)
    stds = []
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(depth):
            lin = nn.Linear(width, width)
            if init is not None:
                init(lin.weight)
                nn.init.zeros_(lin.bias)
            # TODO: apply the layer and ReLU, record the std
            h = torch.relu(lin(h))
            stds.append(h.std().item())
    return stds


he = activation_stds(lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))
big = activation_stds(lambda w: nn.init.normal_(w, std=1.0))
small = activation_stds(lambda w: nn.init.normal_(w, std=0.01))
default = activation_stds(None)
for name, s in [("He", he), ("N(0, 1)", big), ("N(0, 0.01²)", small), ("PyTorch default", default)]:
    print(f"{name:16s} layer 1 std {s[0]:10.3g}   layer 10 std {s[-1]:10.3g}")
check(0.5 < he[-1] / he[0] < 2, "He init keeps layer 10's std within 2× of layer 1's")
check(big[-1] > 1e6, "N(0, 1) weights explode the signal (> 10⁶ by layer 10)")
check(small[-1] < 1e-6, "std 0.01 weights make the signal vanish (< 10⁻⁶ by layer 10)")
check(default[-1] < default[0] / 5, "PyTorch's default shrinks a deep plain ReLU stack (it isn't He init)")
plt.semilogy(range(1, 11), he, label="He"); plt.semilogy(range(1, 11), default, label="PyTorch default")
plt.semilogy(range(1, 11), big, label="N(0, 1)"); plt.semilogy(range(1, 11), small, label="std 0.01")
plt.xlabel("layer"); plt.ylabel("activation std"); plt.legend(); plt.title("TODO 4: init scale"); show()


# %% TODO 5: SGD vs Adam vs AdamW, and why the "W" matters (Lesson 1.5)
def make_optimizer(name, params, lr):
    """Return the optimizer called name: "sgd", "momentum" (0.9), "adam", or "adamw" (weight_decay 0.1)."""
    # TODO: four torch.optim constructors
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
    raise ValueError(name)


def final_loss(name, lr, steps=300):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    o = make_optimizer(name, net.parameters(), lr)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        idx = torch.randint(0, 1000, (64,), generator=g)
        loss = loss_fn(net(xtr[idx]), ytr[idx])
        o.zero_grad(); loss.backward(); o.step()
    with torch.no_grad():
        return loss_fn(net(xtr), ytr).item()


results = {name: final_loss(name, 1e-2) for name in ["sgd", "momentum", "adam", "adamw"]}
print("loss after 300 steps at lr 0.01:", {k: round(v, 3) for k, v in results.items()})
check(results["adam"] < results["sgd"] / 2, "at the same small lr, Adam beats plain SGD by more than 2×")
check(results["momentum"] < results["sgd"], "momentum beats plain SGD")

# The W: a weight whose gradient is always 0 (think: an unused embedding row), decay 0.1, lr 0.01, 100 steps.
def decayed(opt_cls):
    w = nn.Parameter(torch.ones(1))
    o = opt_cls([w], lr=0.01, weight_decay=0.1)
    for _ in range(100):
        o.zero_grad(); w.grad = torch.zeros(1); o.step()
    return w.item()

w_l2, w_decoupled = decayed(torch.optim.Adam), decayed(torch.optim.AdamW)
print(f"after 100 steps: Adam with L2 → {w_l2:.3f}, AdamW → {w_decoupled:.3f} (expected (1 − 0.01·0.1)^100 = {(1 - 0.001) ** 100:.3f})")
check(abs(w_decoupled - (1 - 0.001) ** 100) < 1e-4, "AdamW shrinks the weight by exactly lr·λ per step")
check(w_l2 < 0.3, "Adam + L2 divides the decay by √v: an unused weight moves ≈ lr per step, far more than lr·λ")


# %% TODO 6: a learning-rate sweep finds the band that works (Lesson 1.6)
def lr_sweep(lrs, steps=150):
    """Final full-batch loss after `steps` plain-SGD steps for each lr (same init every time)."""
    out = []
    for lr in lrs:
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
        o = torch.optim.SGD(net.parameters(), lr=lr)
        for _ in range(steps):
            # TODO: one full-batch step on (xtr, ytr)
            loss = loss_fn(net(xtr), ytr)
            o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            L = loss_fn(net(xtr), ytr).item()
        out.append(L if math.isfinite(L) else float("inf"))
    return out


lrs = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
sweep = lr_sweep(lrs)
for lr, L in zip(lrs, sweep):
    print(f"lr {lr:>7}: loss {L:.3f}")
best = min(range(len(lrs)), key=lambda i: sweep[i])
check(0 < best < len(lrs) - 1, f"the best lr ({lrs[best]}) is in the middle, not at either end of the sweep")
check(sweep[0] > 2 * sweep[best] and sweep[-1] > 2 * sweep[best], "both too small and too big are much worse: the U-shape")
plt.semilogx(lrs, [min(L, 2) for L in sweep], "o-"); plt.xlabel("learning rate"); plt.ylabel("loss after 150 steps")
plt.title("TODO 6: the U-curve"); show()


# %% TODO 7: warmup + cosine with a scheduler (Lesson 1.6)
TOTAL, WARMUP, FINAL = 1000, 100, 0.1


def warmup_cosine(step):
    """Multiplier of the peak lr: linear warmup (step+1)/WARMUP, then cosine from 1 down to FINAL."""
    # TODO: two cases
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / (TOTAL - WARMUP)
    return FINAL + (1 - FINAL) * 0.5 * (1 + math.cos(math.pi * progress))


opt = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine)
seen = []
for step in range(TOTAL):
    opt.step()
    seen.append(opt.param_groups[0]["lr"])
    sched.step()
print(f"lr at steps 0, 49, 99, 550, 999: {[f'{seen[s]:.2e}' for s in (0, 49, 99, 550, 999)]}")
check(abs(seen[99] - 3e-4) < 1e-12, "the warmup reaches the peak (3e-4) at step 99")
check(abs(seen[49] - 1.5e-4) < 1e-12, "halfway through warmup the lr is half the peak")
check(abs(seen[550] - 3e-4 * (0.1 + 0.9 * 0.5)) < 1e-9, "halfway through the cosine the lr is halfway between peak and final")
check(abs(seen[-1] - 3e-5) < 1e-6, "the run ends at 10% of the peak")
plt.plot(seen); plt.xlabel("step"); plt.ylabel("lr"); plt.title("TODO 7: warmup + cosine"); show()

print("\n🎉 Lab 1 complete: you've built and checked every piece of a training run.")

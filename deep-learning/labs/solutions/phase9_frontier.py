# %% [markdown]
# # Lab 9: Frontier & robustness 🔭
#
# Six small builds, one per lesson of Phase 9, all on tiny synthetic data on your CPU:
# an InfoNCE (contrastive) loss, a GCN layer by hand, a linear recurrence computed two ways
# (a loop and a convolution), an FGSM attack on a trained MLP, temperature scaling with ECE,
# and an ensemble that disagrees on out-of-distribution points.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase9_frontier.py`.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 9.1 (InfoNCE), 9.2 (GCN), 9.3 (SSMs), 9.5 (FGSM), 9.6 (calibration, ensembles)

# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# %% TODO 1: the InfoNCE loss (Lesson 9.1)
def info_nce(z1, z2, tau=0.5):
    """z1[i] and z2[i] are two augmented views of image i (shape N × d).
    Every other view in the 2N batch is a negative. Return the mean "pick your twin" cross-entropy."""
    # TODO: normalize the 2N embeddings to length 1, build the 2N × 2N cosine-similarity matrix / tau,
    # mask the diagonal with -inf (a view is not its own twin), and use cross-entropy where the
    # target for row i is the index of its twin (i + N for the first half, i − N for the second).
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    n = z1.shape[0]
    twin = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, twin)


a = torch.randn(64, 16)
rand_loss = info_nce(a, torch.randn(64, 16)).item()
copy_loss = info_nce(a, a + 0.01 * torch.randn(64, 16), tau=0.1).item()
check(abs(rand_loss - math.log(2 * 64 - 1)) < 0.5, f"random pairs score about log(2N − 1) = {math.log(127):.2f} (got {rand_loss:.2f})")
check(copy_loss < 0.2, f"near-identical twins at τ = 0.1 give a loss near 0 (got {copy_loss:.3f})")
# the worked example from the lesson: one anchor, twin similarity 0.9, two negatives at 0.2, τ = 0.5
logits = torch.tensor([[0.9, 0.2, 0.2]]) / 0.5
check(abs(F.cross_entropy(logits, torch.tensor([0])).item() - 0.401) < 1e-3, "the lesson's worked example: loss 0.401")


# %% TODO 2: a GCN layer by hand (Lesson 9.2)
def gcn_layer(A, H, W):
    """One graph convolution: D^-1/2 (A + I) D^-1/2 · H · W  (no activation).
    A: n × n adjacency (0/1, symmetric), H: n × d node features, W: d × d_out weights."""
    # TODO: add self-loops, compute degrees of A + I, scale rows and columns by 1/sqrt(degree), multiply.
    A_hat = A + torch.eye(A.shape[0])
    d = A_hat.sum(1).rsqrt()
    return (d[:, None] * A_hat * d[None, :]) @ H @ W


path = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)   # A — B — C
out = gcn_layer(path, torch.tensor([[3.0], [0.0], [1.0]]), torch.ones(1, 1))
check(torch.allclose(out.squeeze(), torch.tensor([1.5, 4 / math.sqrt(6), 0.5]), atol=1e-4), "the lesson's path graph: [1.5, 1.633, 0.5]")
A = (torch.rand(8, 8) < 0.35).float(); A = ((A + A.T) > 0).float(); A.fill_diagonal_(0)
H, W = torch.randn(8, 4), torch.randn(4, 3)
perm = torch.randperm(8)
check(torch.allclose(gcn_layer(A[perm][:, perm], H[perm], W), gcn_layer(A, H, W)[perm], atol=1e-5),
      "relabelling the nodes just relabels the outputs (permutation equivariance)")


# %% TODO 3: a linear recurrence, as a loop and as a convolution (Lesson 9.3)
def recurrence_loop(x, a, b):
    """h_t = a · h_{t−1} + b · x_t with h_0 = 0. Return all h_t (same length as x)."""
    # TODO: a plain Python loop over time.
    h, out = torch.tensor(0.0), []
    for t in range(len(x)):
        h = a * h + b * x[t]
        out.append(h)
    return torch.stack(out)


def recurrence_conv(x, a, b):
    """The same h_t computed as ONE causal convolution with kernel K = [b, a·b, a²·b, ...]."""
    # TODO: build K, flip it (conv1d is cross-correlation), left-pad x by len(x) − 1, call F.conv1d.
    T = len(x)
    K = b * a ** torch.arange(T, dtype=x.dtype)
    return F.conv1d(F.pad(x, (T - 1, 0))[None, None], K.flip(0)[None, None])[0, 0]


check(torch.allclose(recurrence_loop(torch.tensor([1.0, 2.0, 3.0]), 0.5, 1.0), torch.tensor([1.0, 2.5, 4.25])),
      "the lesson's worked example: [1, 2.5, 4.25]")
xs = torch.randn(500)
check(torch.allclose(recurrence_loop(xs, 0.9, 0.3), recurrence_conv(xs, 0.9, 0.3), atol=1e-4),
      "loop and convolution give the same states (that's what lets SSMs train in parallel)")


# %% TODO 4: FGSM on a trained MLP (Lesson 9.5)
def make_moons(n, noise=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * math.pi
    y = torch.arange(n) % 2
    x = torch.stack([torch.where(y == 0, torch.cos(t), 1 - torch.cos(t)), torch.where(y == 0, torch.sin(t), 0.5 - torch.sin(t))], 1)
    return x + noise * torch.randn(n, 2, generator=g), y


Xtr, ytr = make_moons(600, seed=1)
Xte, yte = make_moons(400, seed=2)
torch.manual_seed(0)
clf = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
for _ in range(400):
    opt.zero_grad(); F.cross_entropy(clf(Xtr), ytr).backward(); opt.step()
accuracy = lambda m, x, y: (m(x).argmax(1) == y).float().mean().item()


def fgsm(model, x, y, eps):
    """Return x moved by eps in the SIGN of the gradient of the loss with respect to x."""
    # TODO: x.clone().requires_grad_(True), cross-entropy, backward, then x + eps · sign(grad); detach.
    x = x.clone().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return (x + eps * x.grad.sign()).detach()


x_adv = fgsm(clf, Xte, yte, 0.3)
clean, attacked = accuracy(clf, Xte, yte), accuracy(clf, x_adv, yte)
print(f"clean accuracy {clean:.3f}, under FGSM ε = 0.3: {attacked:.3f}")
check((x_adv - Xte).abs().max().item() <= 0.3 + 1e-6, "every coordinate moved by at most ε")
check(clean > 0.9 and attacked < clean - 0.2, "FGSM costs the model more than 20 points of accuracy")
eps_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
plt.plot(eps_grid, [accuracy(clf, fgsm(clf, Xte, yte, e), yte) for e in eps_grid], marker="o")
plt.xlabel("ε"); plt.ylabel("accuracy under FGSM"); plt.title("Accuracy vs attack size"); show()


# %% TODO 5: temperature scaling and ECE (Lesson 9.6)
def ece(probs, labels, n_bins=15):
    """Expected calibration error: Σ over confidence bins of (bin share) × |accuracy − confidence|."""
    # TODO: conf, pred = probs.max(1); loop over equal-width bins (lo, hi]; add the weighted gaps.
    conf, pred = probs.max(1)
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inb = (conf > lo) & (conf <= hi)
        if inb.any():
            total += inb.float().mean().item() * abs((pred[inb] == labels[inb]).float().mean().item() - conf[inb].mean().item())
    return total


def fit_temperature(logits, labels):
    """Find T > 0 minimizing the NLL of softmax(logits / T). Return T as a float."""
    # TODO: optimize log_T (so T = exp(log_T) stays positive) with LBFGS or Adam on F.cross_entropy(logits / T, labels).
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), labels)
        loss.backward()
        return loss.detach()

    opt.step(closure)
    return log_T.exp().item()


z = 1.2 * torch.randn(6000, 10); cls = torch.randint(0, 10, (6000,)); z[torch.arange(6000), cls] += 4.0
labels = torch.multinomial(F.softmax(z, 1), 1).squeeze(1)       # labels drawn from calibrated probabilities
over = 2.5 * z                                                   # an overconfident model's logits
T = fit_temperature(over[:3000], labels[:3000])                  # fit on the validation half
before, after = ece(F.softmax(over[3000:], 1), labels[3000:]), ece(F.softmax(over[3000:] / T, 1), labels[3000:])
print(f"fitted T = {T:.2f}; test ECE {100 * before:.1f}% → {100 * after:.1f}%")
check(2.0 < T < 3.0, "the fitted temperature undoes the 2.5× overconfidence")
check(after < 0.03 < before, "ECE drops below 3% after temperature scaling")
check(torch.equal(over[3000:].argmax(1), (over[3000:] / T).argmax(1)), "accuracy is untouched: the argmax never changes")


# %% TODO 6: an ensemble's disagreement on out-of-distribution points (Lesson 9.6)
def train_member(seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
    o = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        o.zero_grad(); F.cross_entropy(m(Xtr), ytr).backward(); o.step()
    return m


def disagreement(models, x):
    """Std (over the ensemble) of each model's probability for class 1, one number per input."""
    # TODO: stack softmax(m(x))[:, 1] for every model into M × N, return the std over dim 0.
    with torch.no_grad():
        p = torch.stack([F.softmax(m(x), 1)[:, 1] for m in models])
    return p.std(0)


ensemble = [train_member(s) for s in range(5)]
angle = torch.rand(400) * 2 * math.pi
ood = torch.stack([torch.cos(angle), torch.sin(angle)], 1) * (4 + 2 * torch.rand(400, 1)) + torch.tensor([0.5, 0.25])
d_in, d_ood = disagreement(ensemble, Xte).mean().item(), disagreement(ensemble, ood).mean().item()
single_conf = F.softmax(ensemble[0](ood), 1).max(1).values.mean().item()
print(f"mean disagreement: in-distribution {d_in:.3f}, far away {d_ood:.3f}; one model's mean confidence far away: {single_conf:.3f}")
check(d_ood > 3 * d_in, "the ensemble disagrees much more on OOD points than on normal ones")
check(single_conf > 0.9, "yet a single model is still confident far from the data")

print("\n🎉 Lab 9 complete: you built the core of SSL, GNNs, SSMs, attacks and uncertainty from scratch.")

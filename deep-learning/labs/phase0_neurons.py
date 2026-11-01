# %% [markdown]
# # Lab 0: Neurons & networks, from one neuron to SwiGLU 🧠
#
# You'll build Phase 0 with PyTorch on a laptop CPU, using tiny made-up data:
# a neuron by hand with tensors, the perceptron rule (it learns AND and loops forever on XOR),
# a 2-layer `nn.Sequential` network that solves XOR, parameter counting by hand, the vanishing
# gradient through sigmoid vs ReLU stacks, the SwiGLU block of modern LLMs, and a softmax that
# never overflows.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase0_neurons.py`
# inside labs/. Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 0.1 (neuron), 0.2 (perceptron, XOR), 0.3 (layers, parameters),
# 0.4 (sigmoid vs ReLU), 0.5 (GELU, SiLU, SwiGLU), 0.6 (softmax, temperature, stability)

# %%
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # seeded → you get the same numbers every run


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def close(a, b, tol=1e-4):
    """True when every number in a is within tol of the matching number in b."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return a.shape == b.shape and bool(torch.all((a - b).abs() < tol))


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: one neuron, by hand (Lesson 0.1)
def neuron(x, w, b):
    """A sigmoid neuron: σ(x · w + b). x: (n, d) inputs, w: (d,) weights, b: a number. Returns (n,)."""
    # TODO: z = x @ w + b (the weighted sum, one per row), then return torch.sigmoid(z)
    # Hint: @ is matrix multiplication; (n, d) @ (d,) gives (n,).
    raise NotImplementedError("TODO 1: neuron")


emails = torch.tensor([[4.0, 2.0], [1.0, 0.0]])  # [links, ALL-CAPS words]
w_spam, b_spam = torch.tensor([0.8, 0.5]), -3.0
p = neuron(emails, w_spam, b_spam)
print("P(spam):", p)
check(close(p[0], 0.7685, 1e-3), "4 links, 2 caps: z = 1.2 → σ(1.2) = 0.769 (Lesson 0.1's worked example)")
check(close(p[1], 0.0998, 1e-3), "1 link, 0 caps: z = −2.2 → σ(−2.2) = 0.100")
check(close(neuron(emails, 2 * w_spam, 2 * b_spam) > 0.5, p > 0.5), "doubling weights and bias changes no yes/no decision")


# %% TODO 2: the perceptron rule (Lesson 0.2)
X4 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
TARGETS = {"AND": torch.tensor([0.0, 0.0, 0.0, 1.0]), "OR": torch.tensor([0.0, 1.0, 1.0, 1.0]),
           "XOR": torch.tensor([0.0, 1.0, 1.0, 0.0])}


def perceptron_epoch(X, y, w, b, lr=1.0):
    """One pass over the data with the perceptron rule. Returns (w, b, number of mistakes)."""
    mistakes = 0
    for x, t in zip(X, y):
        pred = float(x @ w + b > 0)  # step activation
        # TODO: if pred != t: move w by lr * (t − pred) * x and b by lr * (t − pred); count the mistake
        # Hint: (t − pred) is +1 for a missed "yes", −1 for a false alarm.
        raise NotImplementedError("TODO 2: perceptron_epoch")
    return w, b, mistakes


def train_perceptron(X, y, epochs=20):
    w, b, history = torch.zeros(2), 0.0, []
    for _ in range(epochs):
        w, b, m = perceptron_epoch(X, y, w, b)
        history.append(m)
        if m == 0:
            break
    return w, b, history


for name, y in TARGETS.items():
    w, b, hist = train_perceptron(X4, y)
    print(f"{name}: mistakes per epoch {hist}, w = {w.tolist()}, b = {b}")
w, b, hist = train_perceptron(X4, TARGETS["AND"])
check(hist[-1] == 0 and len(hist) == 6, "AND is learned in 5 epochs (the 6th has 0 mistakes)")
check(train_perceptron(X4, TARGETS["OR"])[2][-1] == 0, "OR is learned too")
check(all(m > 0 for m in train_perceptron(X4, TARGETS["XOR"])[2]), "XOR still makes mistakes after 20 epochs: no line exists")


# %% See it: the best any single line can do on XOR is 75%
gen = torch.Generator().manual_seed(1)
W = torch.randn(10000, 2, generator=gen)
B = torch.randn(10000, generator=gen)
preds = (X4 @ W.T + B > 0).float()  # (4 points, 10,000 random lines)
acc = (preds == TARGETS["XOR"][:, None]).float().mean(0)
best = torch.maximum(acc, 1 - acc).max().item()  # either side of a line may be "class 1"
print(f"best of 10,000 random lines on XOR: {best:.0%}")
check(best == 0.75, "no line beats 75% on XOR (Lesson 0.2's four inequalities)")


# %% TODO 3: a hidden layer solves XOR, trained with gradients (Lessons 0.2 and 0.3)
def make_xor_mlp():
    """2 inputs → 4 hidden units with tanh → 1 output logit."""
    # TODO: return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    # (No sigmoid at the end: the loss below takes raw logits, Lesson 0.6.)
    raise NotImplementedError("TODO 3: make_xor_mlp")


torch.manual_seed(0)
mlp = make_xor_mlp()
opt = torch.optim.Adam(mlp.parameters(), lr=0.05)
y_xor = TARGETS["XOR"][:, None]
for step in range(300):
    loss = F.binary_cross_entropy_with_logits(mlp(X4), y_xor)
    opt.zero_grad()
    loss.backward()
    opt.step()
probs = torch.sigmoid(mlp(X4)).squeeze()
print("XOR probabilities after training:", probs.detach().numpy().round(3), " loss", round(loss.item(), 4))
check(((probs > 0.5).float() == TARGETS["XOR"]).all().item(), "the 2-layer network gets all 4 XOR cases right")
check(sum(isinstance(m, nn.Linear) for m in mlp) == 2, "it has exactly two Linear layers (one hidden layer)")


# %% TODO 4: count parameters by hand (Lesson 0.3)
def count_params(sizes):
    """Weights + biases of an MLP with these layer sizes, e.g. [2, 8, 8, 1]."""
    # TODO: for each pair (n_in, n_out) of neighbouring sizes add n_in * n_out + n_out
    # Hint: zip(sizes, sizes[1:]) gives the pairs.
    raise NotImplementedError("TODO 4: count_params")


def build_mlp(sizes):
    layers = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


for sizes in ([2, 8, 8, 1], [2, 44, 1], [2, 8, 8, 8, 1], [784, 128, 10]):
    print(f"{str(sizes):18s} by hand {count_params(sizes):>7}   PyTorch {sum(p.numel() for p in build_mlp(sizes).parameters()):>7}")
check(count_params([2, 8, 8, 1]) == 105, "2 → 8 → 8 → 1 has 105 parameters")
check(count_params([784, 128, 10]) == 101_770, "784 → 128 → 10 has 101,770 parameters")
check(count_params([2, 44, 1]) == count_params([2, 8, 8, 8, 1]) == 177, "the wide and the deep racers of Lesson 0.3 both have 177")
check(all(count_params(s) == sum(p.numel() for p in build_mlp(s).parameters()) for s in ([3, 16, 16, 2], [5, 1])), "your formula matches PyTorch")


# %% See it: depth folds the line (Lesson 0.3)
tent = lambda x: 2 * torch.relu(x) - 4 * torch.relu(x - 0.5)
xs = torch.linspace(0, 1, 2**14 + 1, dtype=torch.float64)
ys = xs
for L in range(1, 7):
    ys = tent(ys)
    s = torch.sign(torch.diff(ys))
    print(f"{L} layers of 2 ReLUs → {int((s[1:] != s[:-1]).sum()) + 1} straight pieces (one wide layer: {2**L - 1} units)")
plt.figure(figsize=(7, 2.6))
plt.plot(xs, ys, color="purple")
plt.title("6 layers × 2 ReLUs = 64 straight pieces")
plt.tight_layout()
show()


# %% TODO 5: vanishing gradients, measured (Lesson 0.4)
def first_to_last_grad_ratio(act, depth=10, width=64, seed=0):
    """Gradient norm of the first Linear layer ÷ that of the last, for a deep stack using `act`."""
    torch.manual_seed(seed)
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        if act is nn.ReLU:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")  # He init for ReLU (Lesson 1.4)
        else:
            nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
    net = nn.Sequential(*layers, nn.Linear(width, 1))
    x = torch.randn(256, width)
    # TODO: compute any scalar loss from net(x) (e.g. net(x).pow(2).mean()), call .backward(),
    #       then return net[0].weight.grad.norm() / net[-1].weight.grad.norm() as a float
    raise NotImplementedError("TODO 5: first_to_last_grad_ratio")


ratios = {a.__name__: first_to_last_grad_ratio(a) for a in (nn.Sigmoid, nn.Tanh, nn.ReLU)}
for k, v in ratios.items():
    print(f"{k:8s} layer-1 / last-layer gradient = {v:.1e}")
check(ratios["Sigmoid"] < 1e-4, "10 sigmoid layers: the first layer gets less than 1/10,000 of the last layer's gradient")
check(ratios["ReLU"] > 1000 * ratios["Sigmoid"], "ReLU (with He init) passes over 1,000× more gradient to layer 1 than sigmoid")


# %% TODO 6: the SwiGLU block of modern LLMs (Lesson 0.5)
class SwiGLU(nn.Module):
    """Llama-style feed-forward block: down( silu(gate(x)) * up(x) )."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        # TODO: return self.down(F.silu(self.gate(x)) * self.up(x))
        # The gate branch goes through SiLU (Swish); the up branch stays linear; multiply, then project down.
        raise NotImplementedError("TODO 6: SwiGLU.forward")


d = 768
classic = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))
swiglu = SwiGLU(d, hidden=int(2 / 3 * 4 * d))
n_classic, n_swiglu = sum(p.numel() for p in classic.parameters()), sum(p.numel() for p in swiglu.parameters())
print(f"classic FFN {n_classic:,} weights, SwiGLU FFN {n_swiglu:,} weights")
x_tok = torch.randn(2, 5, d)  # (batch, tokens, features)
check(tuple(swiglu(x_tok).shape) == (2, 5, d), "SwiGLU maps (batch, tokens, 768) back to (batch, tokens, 768)")
check(n_classic == n_swiglu == 4_718_592, "hidden size 2/3 · 4d = 2,048 keeps the budget equal to the classic 4d MLP")
with torch.no_grad():
    v = swiglu.up(x_tok)
    g = swiglu.gate(x_tok)
    check(close(swiglu(x_tok), swiglu.down(g * torch.sigmoid(g) * v)), "silu(g) is g · σ(g): the gate is a self-gated sigmoid")

zz = torch.linspace(-4, 3, 300)
plt.figure(figsize=(7, 3.2))
for f, name in [(torch.relu, "ReLU"), (F.elu, "ELU"), (F.gelu, "GELU"), (F.silu, "SiLU")]:
    plt.plot(zz, f(zz), label=name)
plt.axhline(0, color="gray", lw=0.6)
plt.legend()
plt.title("SiLU dips to −0.28, GELU to −0.17 (Lesson 0.5)")
plt.tight_layout()
show()


# %% TODO 7: a softmax that never overflows (Lesson 0.6)
def stable_softmax(z, T=1.0):
    """Softmax over the last dimension of z, at temperature T, without overflow."""
    # TODO: divide z by T, subtract the max of each row (z.max(dim=-1, keepdim=True).values),
    #       exponentiate, and divide by the row sums (.sum(dim=-1, keepdim=True))
    raise NotImplementedError("TODO 7: stable_softmax")


logits = torch.tensor([2.0, 1.0, 0.1])
print("softmax:", stable_softmax(logits), " T = 0.5:", stable_softmax(logits, 0.5))
check(close(stable_softmax(logits), [0.6590, 0.2424, 0.0986]), "softmax([2, 1, 0.1]) = [0.659, 0.242, 0.099]")
check(close(stable_softmax(logits + 100), stable_softmax(logits)), "adding 100 to every logit changes nothing")
big = torch.tensor([1000.0, 1001.0])
naive = torch.exp(big) / torch.exp(big).sum()
print("naive:", naive, " stable:", stable_softmax(big))
check(torch.isnan(naive).all().item() and close(stable_softmax(big), [0.2689, 0.7311]), "naive gives NaN on [1000, 1001]; yours gives [0.269, 0.731]")
check(stable_softmax(logits, 0.1)[0] > 0.9999 and (stable_softmax(logits, 100.0) - 1 / 3).abs().max() < 0.01, "T → 0 approaches argmax, T → ∞ approaches uniform")
batch = torch.randn(4, 10) * 30
check(close(stable_softmax(batch), torch.softmax(batch, dim=-1)), "matches torch.softmax on a batch of rows")


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Give the XOR network of TODO 3 only 2 hidden units, and try seeds 0–9. How often does it get stuck?
#    Then try 8 hidden units. (Wider nets are easier to train, even when 2 units are enough in theory.)
# 2. In TODO 5, swap nn.ReLU's He init for Xavier. What happens to the ReLU ratio? (Lesson 1.4 explains.)
# 3. Use `stable_softmax(logits, T)` to sample 1,000 classes with `torch.multinomial` at T = 0.5, 1 and 2.
#    How often does each class come up? That's what an LLM's temperature setting does.

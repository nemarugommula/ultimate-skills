# %% [markdown]
# # Lab 3: Convolutional networks from scratch 🖼️
#
# You'll build the pieces of a CNN by hand and check each one against PyTorch:
# a convolution written with loops (and proof that PyTorch's "convolution" is cross-correlation),
# the output-size formula, parameter counting (and the depthwise-separable saving),
# a receptive-field calculator checked against a real gradient, and then the payoff:
# a 1,316-parameter CNN that recognises shapes in places it has never seen them,
# while a 16,708-parameter MLP fails. Last, you'll look at the CNN's feature maps.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase3_cnns.py`.
# Everything runs on CPU in well under a minute, with synthetic images made in code (no downloads).
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 3.1 (why CNNs), 3.2 (convolution, output size, parameters),
# 3.3 (receptive fields, pooling), 3.4 (depthwise-separable convs), 3.6 (feature maps)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)       # reproducible results
torch.set_num_threads(1)   # tiny tensors run faster on one CPU thread


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: a convolution with plain loops (Lesson 3.2)
def conv2d_naive(x, k):
    """Slide kernel k (kh × kw) over image x (H × W), no padding, stride 1.
    At each position: multiply the patch under the kernel by the kernel, number by number, and add up.
    Returns a tensor of shape (H − kh + 1) × (W − kw + 1)."""
    # TODO: make out = torch.zeros(H − kh + 1, W − kw + 1); for every (r, c):
    #       patch = x[r:r + kh, c:c + kw]; out[r, c] = (patch * k).sum()
    # Hint: H, W = x.shape and kh, kw = k.shape. Two nested for-loops are fine.
    raise NotImplementedError("TODO 1: conv2d_naive")


x = torch.tensor([[0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., 1.]])
vedge = torch.tensor([[-1., 0., 1.]] * 3)            # vertical-edge kernel: right minus left
check(torch.equal(conv2d_naive(x, vedge), torch.tensor([[0., 3., 3.]])), "the edge kernel finds the dark→bright edge: [0, 3, 3]")

img, ker = torch.randn(9, 7), torch.randn(3, 3)
ours = conv2d_naive(img, ker)
torch_out = F.conv2d(img[None, None], ker[None, None])[0, 0]   # PyTorch wants (batch, channels, H, W)
check(torch.allclose(ours, torch_out, atol=1e-5), "matches F.conv2d exactly")
flipped = torch.flip(ker, dims=(0, 1))
check(not torch.allclose(conv2d_naive(img, flipped), torch_out, atol=1e-3),
      "a flipped kernel gives a different answer: PyTorch does NOT flip (it's cross-correlation)")


# %% TODO 2: the output-size formula (Lesson 3.2)
def conv_out_size(n, k, s=1, p=0, d=1):
    """Output side for input side n, kernel k, stride s, padding p, dilation d:
    ⌊(n + 2p − d·(k − 1) − 1) / s⌋ + 1   (with d = 1 this is ⌊(n + 2p − k) / s⌋ + 1)."""
    # TODO: return the formula above. Hint: Python's // is floor division.
    raise NotImplementedError("TODO 2: conv_out_size")


for n, k, s, p, d in [(32, 3, 1, 1, 1), (32, 3, 2, 1, 1), (7, 3, 2, 0, 1), (224, 7, 2, 3, 1), (224, 11, 4, 2, 1), (28, 5, 1, 0, 1), (32, 3, 1, 2, 2)]:
    real = nn.Conv2d(1, 1, k, stride=s, padding=p, dilation=d)(torch.zeros(1, 1, n, n)).shape[-1]
    check(conv_out_size(n, k, s, p, d) == real, f"n={n}, k={k}, s={s}, p={p}, d={d} → {real}")

sizes = [32]
for _ in range(3):
    sizes.append(conv_out_size(sizes[-1], 3, 2, 0))
check(sizes == [32, 15, 7, 3], "three valid stride-2 convs: 32 → 15 → 7 → 3 (Lesson 3.2 quiz)")


# %% TODO 3: counting parameters (Lessons 3.1, 3.2, 3.4)
def conv_params(c_in, c_out, k, bias=True):
    """A conv layer has one k × k × c_in kernel per output channel, plus one bias per output channel."""
    # TODO: weights = k·k·c_in·c_out, plus c_out biases when bias=True
    raise NotImplementedError("TODO 3a: conv_params")


def separable_params(c_in, c_out, k):
    """Depthwise k × k (one filter per input channel) + pointwise 1 × 1 (c_in → c_out); no biases."""
    # TODO: depthwise has k·k·c_in weights, pointwise has c_in·c_out
    raise NotImplementedError("TODO 3b: separable_params")


count = lambda m: sum(p.numel() for p in m.parameters())
check(conv_params(3, 64, 3) == count(nn.Conv2d(3, 64, 3)) == 1792, "3×3 conv, 3 → 64 channels: 1,792 parameters")
check(conv_params(256, 64, 1) == count(nn.Conv2d(256, 64, 1)) == 16448, "1×1 conv, 256 → 64: 16,448 parameters")
dense = 224 * 224 * 3 * 1000 + 1000
print(f"dense layer on a 224×224×3 image → 1,000 units: {dense:,} parameters ({dense / conv_params(3, 64, 3):,.0f}× the conv)")
sep = nn.Sequential(nn.Conv2d(128, 128, 3, groups=128, bias=False), nn.Conv2d(128, 128, 1, bias=False))
check(separable_params(128, 128, 3) == count(sep) == 17536, "depthwise-separable 3×3, 128 → 128: 17,536 weights")
ratio = conv_params(128, 128, 3, bias=False) / separable_params(128, 128, 3)
check(abs(ratio - 1 / (1 / 128 + 1 / 9)) < 1e-9, f"saving = 1 / (1/N + 1/k²) = {ratio:.1f}× (MobileNet's 8–9×)")


# %% TODO 4: receptive fields (Lesson 3.3)
def receptive_field(layers):
    """layers: list of (k, s) or (k, s, d). Start with rf = 1 and jump = 1; for each layer:
    rf += d·(k − 1)·jump, then jump *= s. Return the final rf (pixels per side)."""
    # TODO: loop over the layers; read k, s (and d, default 1); update rf, then jump.
    # Hint: d = layer[2] if len(layer) > 2 else 1. Update rf BEFORE multiplying jump by s.
    raise NotImplementedError("TODO 4: receptive_field")


c3, pool = (3, 1), (2, 2)
check(receptive_field([c3, c3, c3]) == 7, "three 3×3 convs see 7×7 (like one 7×7, the VGG argument)")
check(receptive_field([c3, c3, pool, c3, c3]) == 14, "conv, conv, pool, conv, conv → 14 (Lesson 3.3 worked example)")
vgg16 = [c3, c3, pool, c3, c3, pool, c3, c3, c3, pool, c3, c3, c3, pool, c3, c3, c3]
check(receptive_field(vgg16) == 196, "VGG-16's last conv layer sees 196×196 pixels")
check(receptive_field([(3, 1, 1), (3, 1, 2), (3, 1, 4), (3, 1, 8)]) == 31, "dilations 1, 2, 4, 8 → 31×31 with four 3×3 layers")

# Measure it for real: pixels with a non-zero gradient are exactly the ones the output can see.
stack = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
probe = torch.randn(4, 1, 64, 64, requires_grad=True)
stack(probe)[:, :, 16, 16].sum().backward()
rows = (probe.grad.abs().sum(dim=(0, 1, 3)) > 0).nonzero()
measured = (rows.max() - rows.min() + 1).item()
check(measured == receptive_field([c3, pool, c3, c3]), f"gradient-measured RF ({measured}) = formula ({receptive_field([c3, pool, c3, c3])})")


# %% The data: shapes in one corner of the image (training) vs another corner (test)
S = 16
NAMES = ["horizontal bar", "vertical bar", "plus", "square"]


def shape(kind):
    m = torch.zeros(5, 5)
    if kind == 0: m[2, :] = 1
    elif kind == 1: m[:, 2] = 1
    elif kind == 2: m[2, :] = 1; m[:, 2] = 1
    else: m[0, :] = 1; m[-1, :] = 1; m[:, 0] = 1; m[:, -1] = 1
    return m


def make_shapes(n, lo, hi, gen):
    """n noisy 16×16 images, each with one 5×5 shape whose top-left corner is in rows/cols lo..hi."""
    X, y = torch.zeros(n, 1, S, S), torch.randint(0, 4, (n,), generator=gen)
    pos = torch.randint(lo, hi + 1, (n, 2), generator=gen)
    for i in range(n):
        r, c = pos[i].tolist()
        X[i, 0, r:r + 5, c:c + 5] = shape(int(y[i]))
    return X + 0.1 * torch.randn(X.shape, generator=gen), y


gen = torch.Generator().manual_seed(1)
X_train, y_train = make_shapes(400, 0, 5, gen)      # shapes only in the top-left area
X_test, y_test = make_shapes(400, 8, 11, gen)       # shapes only in the bottom-right area
X_same, y_same = make_shapes(400, 0, 5, gen)        # new images, same area as training

fig, axes = plt.subplots(1, 6, figsize=(10, 2))
for ax, (X, y, where) in zip(axes, [(X_train, y_train, "train")] * 3 + [(X_test, y_test, "test")] * 3):
    i = int(torch.randint(0, 400, (1,)))
    ax.imshow(X[i, 0], cmap="gray"); ax.set_title(f"{where}: {NAMES[y[i]]}", fontsize=7); ax.axis("off")
show()


# %% TODO 5: a tiny CNN vs an MLP on shifted shapes (Lessons 3.1 and 3.3)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        """conv → ReLU → conv → ReLU → global MAX pool over height and width → linear.
        Shapes: (N, 1, 16, 16) → (N, 8, 16, 16) → (N, 16, 16, 16) → (N, 16) → (N, 4)."""
        # TODO: ReLU after c1 and after c2, then x.amax(dim=(2, 3)) (global max pool), then self.fc
        raise NotImplementedError("TODO 5: TinyCNN.forward")


def fit(model, steps=150):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(X_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = lambda X, y: (model(X).argmax(1) == y).float().mean().item()
        return loss.item(), acc(X_same, y_same), acc(X_test, y_test)


torch.manual_seed(0)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(S * S, 64), nn.ReLU(), nn.Linear(64, 4))
cnn = TinyCNN()
check(cnn(X_train[:2]).shape == (2, 4), "TinyCNN outputs one score per class: shape (2, 4)")
results = {}
for name, model in [("MLP", mlp), ("CNN", cnn)]:
    loss, same, moved = fit(model)
    results[name] = (same, moved)
    print(f"{name}: {count(model):6,} parameters | train loss {loss:.4f} | same area {same:.0%} | new area {moved:.0%}")
check(count(cnn) < count(mlp) / 10, "the CNN has over 10× fewer parameters than the MLP")
check(results["CNN"][1] >= 0.9, "the CNN recognises shapes in an area it never saw in training (≥ 90%)")
check(results["MLP"][1] < 0.5, "the MLP fails there (< 50%): its weights are tied to pixel positions")


# %% TODO 6: look at the feature maps (Lesson 3.6)
def first_layer_maps(model, x):
    """The CNN's first-layer feature maps for a batch x: ReLU(conv1(x)), shape (N, 8, 16, 16)."""
    # TODO: apply the first conv layer, then ReLU
    raise NotImplementedError("TODO 6: first_layer_maps")


with torch.no_grad():
    x0 = X_test[:1]
    maps = first_layer_maps(cnn, x0)
    check(maps.shape == (1, 8, S, S), "8 feature maps, each 16×16 (padding 1 keeps the size)")
    moved = torch.roll(x0, shifts=(-3, -3), dims=(2, 3))              # move the shape 3 pixels up-left
    m_moved = first_layer_maps(cnn, moved)
    inner = (slice(None), slice(None), slice(2, 10), slice(2, 10))      # compare away from the borders
    check(torch.allclose(torch.roll(maps, (-3, -3), dims=(2, 3))[inner], m_moved[inner], atol=1e-5),
          "shift the input, and the feature maps shift the same way (equivariance)")

fig, axes = plt.subplots(2, 9, figsize=(12, 3))
axes[0, 0].imshow(x0[0, 0], cmap="gray"); axes[0, 0].set_title(NAMES[y_test[0]], fontsize=7)
axes[1, 0].axis("off")
for i in range(8):
    axes[0, i + 1].imshow(cnn.c1.weight[i, 0].detach(), cmap="bwr"); axes[0, i + 1].set_title(f"filter {i}", fontsize=7)
    axes[1, i + 1].imshow(maps[0, i], cmap="magma"); axes[1, i + 1].set_title(f"map {i}", fontsize=7)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
show()

print("\n🎉 Lab 3 complete: you built convolution, shapes, parameter counts and receptive fields by hand,")
print("   and watched weight sharing + global pooling beat an MLP 10× its size on shifted shapes.")

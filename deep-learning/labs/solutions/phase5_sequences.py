# %% [markdown]
# # Lab 5: Sequences from scratch 📜
#
# (SOLUTION file: every TODO is filled in. Try labs/phase5_sequences.py first!)
#
# You'll build the classic sequence toolkit in PyTorch, on tiny synthetic data, on CPU:
# an embedding lookup (and why it equals one-hot × matrix), a skip-gram with negative
# sampling that puts "cat" next to "dog" on a 17-sentence corpus, an RNN cell by hand that
# matches nn.RNN exactly, a measurement of how fast gradients fade with time lag (RNN vs
# LSTM), an LSTM that remembers the first token of a sequence through 20 distractors, and
# Bahdanau's additive attention by hand.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase5_sequences.py`
# inside labs/. Fill each TODO until its ✅ checks pass. The whole file runs in well under a minute.
# Lessons to reread if stuck: 5.1 (embeddings, SGNS), 5.2 (RNN, BPTT), 5.3 (LSTM), 5.4 (attention)

# %%
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
T0 = time.time()


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: an embedding is a lookup (Lesson 5.1)
def embed(ids, table):
    """Return the rows of `table` (V × d) for the integer ids (any shape) → shape ids.shape + (d,)."""
    # TODO: index the table with the ids (one line). No loops, no one-hot.
    return table[ids]


table = torch.randn(10, 4)
ids = torch.tensor([3, 7, 3])
out = embed(ids, table)
check(out.shape == (3, 4), "3 ids → a 3 × 4 matrix")
check(torch.equal(out[0], out[2]), "the same word always gets the same vector")
one_hot = F.one_hot(ids, num_classes=10).float()
check(torch.allclose(one_hot @ table, out), "lookup == one-hot × table (same answer, far less work)")
emb = nn.Embedding(10, 4)
check(torch.equal(embed(ids, emb.weight), emb(ids)), "matches nn.Embedding")


# %% TODO 2: skip-gram with negative sampling (Lesson 5.1)
def sgns_loss(w, c_pos, c_neg):
    """Skip-gram negative-sampling loss, averaged over the batch.
    w: (B, d) centre vectors, c_pos: (B, d) real neighbours, c_neg: (B, k, d) random 'fake' neighbours.
    loss = −log σ(w·c_pos) − Σ_k log σ(−w·c_neg_k)
    """
    # TODO: score the real pair and the k fake pairs with dot products, then use F.logsigmoid.
    pos = (w * c_pos).sum(-1)                               # (B,)
    neg = (w.unsqueeze(1) * c_neg).sum(-1)                  # (B, k)
    return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()


zero = torch.zeros(1, 3)
check(abs(sgns_loss(zero, zero, torch.zeros(1, 5, 3)).item() - 6 * math.log(2)) < 1e-6,
      "all-zero vectors: every guess is 50/50, loss = 6 × ln 2 = 4.159 (k = 5)")
w, c = torch.tensor([[0.5, 0.2]]), torch.tensor([[0.4, 0.1]])
check(abs(sgns_loss(w, c, torch.zeros(1, 0, 2)).item() - 0.5892) < 1e-3, "Lesson 5.1's worked example: −ln σ(0.22) = 0.589")

corpus = ["cat chases mouse", "dog chases ball", "cat drinks milk", "dog drinks water", "kitten drinks milk",
          "puppy chases ball", "cat sleeps on sofa", "dog sleeps on sofa", "kitten sleeps on sofa", "puppy drinks water",
          "car drives on road", "truck drives on road", "bus drives on road", "car needs fuel", "truck needs fuel",
          "bus carries people", "car carries people"]
sents = [s.split() for s in corpus]
vocab = sorted({w for s in sents for w in s})
idx = {w: i for i, w in enumerate(vocab)}
pairs = torch.tensor([(idx[s[i]], idx[s[j]]) for s in sents for i in range(len(s))
                      for j in range(max(0, i - 2), min(len(s), i + 3)) if j != i])
counts = torch.bincount(pairs[:, 0], minlength=len(vocab)).float()
noise = counts ** 0.75 / (counts ** 0.75).sum()             # word2vec's unigram^(3/4) noise distribution

torch.manual_seed(0)
W = nn.Embedding(len(vocab), 4)                          # tiny corpus → tiny vectors
C = nn.Embedding(len(vocab), 4)
opt = torch.optim.Adam(list(W.parameters()) + list(C.parameters()), lr=0.05)
for step in range(300):
    batch = pairs[torch.randint(0, len(pairs), (64,))]
    negs = torch.multinomial(noise, 64 * 5, replacement=True).view(64, 5)
    loss = sgns_loss(W(batch[:, 0]), C(batch[:, 1]), C(negs))
    opt.zero_grad(); loss.backward(); opt.step()
cos = lambda a, b: F.cosine_similarity(W.weight[idx[a]], W.weight[idx[b]], dim=0).item()
print(f"loss {loss.item():.3f} · cos(cat, dog) = {cos('cat', 'dog'):.2f} · cos(cat, car) = {cos('cat', 'car'):.2f}")
check(cos("cat", "dog") > cos("cat", "car") + 0.3, "cat ends up much closer to dog than to car (they never share a sentence)")
check(cos("car", "truck") > cos("car", "kitten") + 0.3, "and car is closer to truck than to kitten")


# %% TODO 3: an RNN cell by hand (Lesson 5.2)
def rnn_forward(x, W_ih, W_hh, b_ih, b_hh):
    """x: (T, input) one sequence. Return all hidden states (T, hidden), starting from h0 = 0.
    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t−1} + b_hh)"""
    # TODO: loop over time and stack the states.
    h = torch.zeros(W_hh.shape[0])
    hs = []
    for t in range(x.shape[0]):
        h = torch.tanh(W_ih @ x[t] + b_ih + W_hh @ h + b_hh)
        hs.append(h)
    return torch.stack(hs)


torch.manual_seed(0)
rnn = nn.RNN(input_size=5, hidden_size=8)
x = torch.randn(12, 5)
mine = rnn_forward(x, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0)
theirs, _ = rnn(x.unsqueeze(1))
check(mine.shape == (12, 8), "12 steps → 12 hidden states of size 8")
check(torch.allclose(mine, theirs[:, 0], atol=1e-6), "your loop matches nn.RNN exactly")
check(sum(p.numel() for p in rnn.parameters()) == 8 * (5 + 8) + 2 * 8, "parameters don't depend on length: 8 × (5 + 8) + 2 × 8 = 120")


# %% TODO 4: how far back can it remember? (Lessons 5.2 and 5.3)
def grad_vs_lag(model, T=100, inp=8, seed=0):
    """Feed one random sequence of length T through `model` (batch_first). Backprop from the LAST output
    and return a length-T tensor: ‖∂ output_T / ∂ x_t‖ for every t (index −1 = the last step)."""
    torch.manual_seed(seed)
    x = torch.randn(1, T, inp, requires_grad=True)
    # TODO: run the model, sum the last output vector, backward, return the per-step gradient norms.
    out, _ = model(x)
    out[0, -1].sum().backward()
    return x.grad[0].norm(dim=1)


torch.manual_seed(1)
rnn = nn.RNN(8, 32, batch_first=True)
lstm = nn.LSTM(8, 32, batch_first=True)
with torch.no_grad():                                        # forget-gate bias = 1 (gates are [i, f, g, o])
    lstm.bias_ih_l0[32:64].fill_(1.0); lstm.bias_hh_l0[32:64].zero_()
g_rnn, g_lstm = grad_vs_lag(rnn), grad_vs_lag(lstm)
for k in (1, 10, 30, 60):
    print(f"{k:2d} steps back:  RNN {g_rnn[-1 - k].item():.1e}   LSTM {g_lstm[-1 - k].item():.1e}")
check(g_rnn.shape == (100,), "one gradient size per time step")
check(g_rnn[-1 - 30] < 1e-3 * g_rnn[-2], "RNN: the signal from 30 steps back is < 0.1% of the one from 1 step back")
check(g_lstm[-1 - 30] > 100 * g_rnn[-1 - 30], "LSTM: 30 steps back it keeps over 100× more gradient than the RNN")
plt.semilogy(g_rnn.flip(0).detach(), label="RNN"); plt.semilogy(g_lstm.flip(0).detach(), label="LSTM (forget bias 1)")
plt.xlabel("steps back"); plt.ylabel("‖∂out/∂x‖"); plt.legend(); plt.title("Gradient vs time lag"); show()


# %% TODO 5: an LSTM that remembers (Lesson 5.3)
# Task: a sequence starts with a "key" token (0–3), followed by L = 50 random distractor tokens (4–9).
# Predict the key at the end. Chance = 25%.
def make_batch(n, L=50):
    key = torch.randint(0, 4, (n,))
    seq = torch.cat([key[:, None], torch.randint(4, 10, (n, L))], dim=1)
    return seq, key


class Rememberer(nn.Module):
    def __init__(self, cell=nn.LSTM, hidden=32, forget_bias=None):
        super().__init__()
        self.emb = nn.Embedding(10, 16)
        self.rnn = cell(16, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 4)
        if forget_bias is not None:                          # LSTM gates are packed [i, f, g, o]
            with torch.no_grad():
                self.rnn.bias_ih_l0[hidden:2 * hidden].fill_(forget_bias)
                self.rnn.bias_hh_l0[hidden:2 * hidden].zero_()

    def forward(self, seq):
        # TODO: embed → run the recurrent layer → take the LAST time step's output → linear to 4 classes.
        h, _ = self.rnn(self.emb(seq))
        return self.out(h[:, -1])


def train(model, steps=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        seq, key = make_batch(64)
        loss = F.cross_entropy(model(seq), key)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # Lesson 5.2: cheap insurance against explosions
        opt.step()
    seq, key = make_batch(1000)
    with torch.no_grad():
        return (model(seq).argmax(1) == key).float().mean().item()


results = {}
for name, kw in [("vanilla RNN", dict(cell=nn.RNN)), ("LSTM, default init", dict(cell=nn.LSTM)),
                 ("LSTM, forget bias 3", dict(cell=nn.LSTM, forget_bias=3.0))]:
    torch.manual_seed(0)
    results[name] = train(Rememberer(**kw))
    print(f"remember the key through 50 distractors: {name:20s} {results[name]:.0%}  (chance 25%)")
check(results["LSTM, forget bias 3"] > 0.9, "an LSTM whose forget gates start open carries the key across 50 distractors (> 90%)")
check(results["vanilla RNN"] < 0.6, "the vanilla RNN can't (< 60%)")
check(results["LSTM, default init"] < 0.5, "and an LSTM with forget gates starting at ≈ 0.5 can't either: the bias matters (Lesson 5.3)")


# %% TODO 6: additive (Bahdanau) attention by hand (Lesson 5.4)
def additive_attention(s, H, W, U, v):
    """s: (d_s,) decoder state, H: (T, d_h) encoder states, W: (a, d_s), U: (a, d_h), v: (a,).
    scores e_j = v · tanh(W s + U h_j); weights = softmax(e); context = Σ_j weight_j h_j.
    Return (weights (T,), context (d_h,))."""
    # TODO: three lines: scores, softmax, weighted sum.
    e = torch.tanh(W @ s + H @ U.T) @ v
    a = F.softmax(e, dim=0)
    return a, a @ H


s = torch.tensor([0.5, -0.5])
H = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
a, ctx = additive_attention(s, H, torch.eye(2), torch.eye(2), torch.ones(2))
print("weights", a.numpy().round(3), "context", ctx.numpy().round(3))
check(torch.allclose(a, torch.tensor([0.195, 0.315, 0.490]), atol=1e-3), "Lesson 5.4's worked example: weights (0.195, 0.315, 0.490)")
check(torch.allclose(ctx, torch.tensor([0.685, 0.805]), atol=1e-3), "context (0.685, 0.805)")
check(abs(a.sum().item() - 1) < 1e-6, "attention weights sum to 1")
torch.manual_seed(0)
T, dh, ds, A = 7, 6, 5, 4
H = torch.randn(T, dh); s = torch.randn(ds)
a, ctx = additive_attention(s, H, torch.randn(A, ds), torch.randn(A, dh), torch.randn(A))
check(a.shape == (T,) and ctx.shape == (dh,), "works for any sizes: 7 source states → 7 weights, one context vector")

print(f"\n🎉 Lab 5 complete in {time.time() - T0:.1f} s")


# %% 🔬 Try it yourself
# 1. TODO 2: train without negatives (k = 0: pass torch.zeros(64, 0, 4)). What happens to cos(cat, car)?
# 2. TODO 4: set the LSTM forget-gate bias to −2 instead of 1. Does the LSTM still beat the RNN?
# 3. TODO 5: try forget_bias=1.0, then nn.GRU, then L = 20 for all of them. Which ones learn, and why?
# 4. TODO 6: make one score much bigger than the others (scale v by 10). How sharp do the weights get?

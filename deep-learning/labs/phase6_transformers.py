# %% [markdown]
# # Lab 6: Transformers from scratch, from one attention head to a tiny GPT ⚡
#
# You'll build Phase 6 with PyTorch on a laptop CPU, no downloads:
# scaled dot-product attention by hand (checked against PyTorch's fused kernel), the causal mask
# (and a test that the future can't leak), multi-head split/merge shapes, RoPE on queries and keys
# (and its "only the distance matters" property), a pre-norm transformer block, a tiny
# character-level GPT trained for a few hundred steps on a built-in text, and the KV-cache and
# Chinchilla arithmetic every LLM engineer does on a napkin.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase6_transformers.py`
# inside labs/. Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 6.1 (attention, √d), 6.2 (multi-head), 6.3 (RoPE), 6.4 (the block,
# pre-norm), 6.7 (next-token loss, causal mask), 6.8 (C ≈ 6ND, Chinchilla), 6.9 (KV cache, SDPA)

# %%
import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # seeded → you get the same numbers every run
t_start = time.time()


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def close(a, b, tol=1e-5):
    """True when every number in a is within tol of the matching number in b."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return a.shape == b.shape and bool(torch.all((a - b).abs() < tol))


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: scaled dot-product attention, by hand (Lesson 6.1)
def attention(q, k, v, mask=None):
    """softmax(q kᵀ / √d) v. q: (..., Tq, d), k: (..., Tk, d), v: (..., Tk, dv).
    mask: optional bool tensor broadcastable to (..., Tq, Tk); True = "may attend"."""
    # TODO: scores = q @ kᵀ / √d  (k.transpose(-2, -1) swaps the last two dims; d = q.shape[-1])
    #       if mask is given: scores = scores.masked_fill(~mask, float("-inf"))
    #       weights = softmax over the LAST dim; return weights @ v
    raise NotImplementedError("TODO 1: attention")


q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
out = attention(q, k, v)
check(out.shape == (2, 5, 8), "output has one vector per query: shape (2, 5, 8)")
check(close(out, F.scaled_dot_product_attention(q, k, v)), "matches F.scaled_dot_product_attention")
# Tiny case you can do by hand: one query that matches key 0 much better than key 1
q1, k1, v1 = torch.tensor([[1.0, 0.0]]), torch.tensor([[4.0, 0.0], [0.0, 0.0]]), torch.tensor([[10.0], [0.0]])
w0 = torch.softmax(torch.tensor([4 / math.sqrt(2), 0.0]), 0)[0]
check(close(attention(q1, k1, v1), [[10 * w0.item()]]), f"by hand: weight on key 0 = softmax(4/√2, 0)[0] = {w0:.3f}")


# %% TODO 2: the causal mask (Lessons 6.4, 6.7)
def causal_mask(T):
    """(T, T) bool matrix, True where query i may look at key j, i.e. j ≤ i (the past and itself)."""
    # TODO: a lower-triangular matrix of True. Hint: torch.tril(torch.ones(T, T, dtype=torch.bool))
    raise NotImplementedError("TODO 2: causal_mask")


m = causal_mask(4)
print(m.int())
check(m.sum().item() == 10, "4 tokens: 1 + 2 + 3 + 4 = 10 allowed (query, key) pairs")
x = torch.randn(1, 6, 8)
y1 = attention(x, x, x, causal_mask(6))
x2 = x.clone()
x2[0, 4:] = torch.randn(2, 8)  # change the FUTURE (positions 4 and 5)
y2 = attention(x2, x2, x2, causal_mask(6))
check(close(y1[0, :4], y2[0, :4]), "changing tokens 4–5 leaves outputs 0–3 unchanged: no peeking at the future")
check(close(y1, F.scaled_dot_product_attention(x, x, x, is_causal=True)), "matches SDPA with is_causal=True")


# %% TODO 3: multi-head split and merge (Lesson 6.2)
def split_heads(x, n_heads):
    """(B, T, C) → (B, n_heads, T, C // n_heads): each head gets its own slice of the channels."""
    # TODO: x.view(B, T, n_heads, C // n_heads) then .transpose(1, 2)
    raise NotImplementedError("TODO 3: split_heads")


def merge_heads(x):
    """(B, H, T, hd) → (B, T, H * hd): undo split_heads."""
    # TODO: .transpose(1, 2) back, then .contiguous().view(B, T, H * hd)
    raise NotImplementedError("TODO 3: merge_heads")


x = torch.randn(2, 7, 32)
h = split_heads(x, 4)
check(h.shape == (2, 4, 7, 8), "32 channels, 4 heads → (B, 4, T, 8)")
check(torch.equal(merge_heads(h), x), "merge_heads(split_heads(x)) gives back exactly x")
check(torch.equal(h[0, 1, 3], x[0, 3, 8:16]), "head 1 sees channels 8–15 of each token")


# %% TODO 4: RoPE on queries and keys (Lesson 6.3)
def rope(x, base=10000.0):
    """Rotate each pair (x[..., 2i], x[..., 2i+1]) of a (..., T, d) tensor by angle pos · base^(−2i/d)."""
    # TODO: pos = arange(T) as a column (T, 1); freqs = base ** (-arange(0, d, 2) / d)  → (d/2,)
    #       ang = pos * freqs  (T, d/2); cos, sin = ang.cos(), ang.sin()
    #       x1, x2 = x[..., 0::2], x[..., 1::2]   (the two halves of every pair)
    #       rotated pair = (x1·cos − x2·sin, x1·sin + x2·cos); interleave back with torch.stack(..., -1).flatten(-2)
    raise NotImplementedError("TODO 4: rope")


T, d = 16, 8
x = torch.randn(T, d)
check(close(rope(x).norm(dim=-1), x.norm(dim=-1)), "rotation keeps every vector's length")
check(close(rope(x)[0], x[0]), "position 0 is not rotated")
# The magic property: the score between a query at position m and a key at position n depends only on m − n.
qv, kv = torch.randn(d), torch.randn(d)
Q, K = rope(qv.expand(T, d)), rope(kv.expand(T, d))  # the same q and k placed at every position
s_a, s_b = Q[5] @ K[2], Q[12] @ K[9]  # both have distance 3
check(close(s_a, s_b, 1e-4), f"score(q@5, k@2) = score(q@12, k@9) = {s_a:.4f}: only the distance matters")


# %% TODO 5: a pre-norm transformer block (Lesson 6.4)
class Block(nn.Module):
    """x + Attn(Norm(x)), then x + MLP(Norm(x)). Causal multi-head self-attention."""

    def __init__(self, C, n_heads):
        super().__init__()
        self.h = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.qkv = nn.Linear(C, 3 * C)  # one matrix makes Q, K and V
        self.proj = nn.Linear(C, C)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

    def forward(self, x):
        # TODO: q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1); split each into heads (TODO 3)
        #       a = attention(q, k, v, causal_mask(T))  (TODO 1, 2); x = x + self.proj(merge_heads(a))
        #       x = x + self.mlp(self.norm2(x)); return x
        raise NotImplementedError("TODO 5: Block.forward")


blk = Block(32, 4)
x = torch.randn(2, 10, 32)
y = blk(x)
check(y.shape == x.shape, "a block keeps the shape (B, T, C): blocks stack")
n_params = sum(p.numel() for p in blk.parameters())
check(n_params == 12 * 32 * 32 + 13 * 32, f"≈ 12·C² parameters per block (+ biases/norms): {n_params:,}")
x2 = x.clone()
x2[:, 7:] = 0
check(close(blk(x2)[:, :7], y[:, :7]), "still causal: changing tokens 7+ doesn't change outputs 0–6")


# %% TODO 6: a tiny character-level GPT, trained on a built-in text (Lessons 6.5, 6.7)
TEXT = (
    "the cat sat on the mat. the dog sat on the log. the cat saw the dog. "
    "the dog saw the cat. a model reads the text and guesses the next letter. "
    "attention lets every letter look back at the letters before it. "
) * 12
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in TEXT])
V, CTX = len(chars), 32
print(f"{len(TEXT)} characters, vocabulary of {V}")


class TinyGPT(nn.Module):
    def __init__(self, V, C=64, n_heads=4, n_layers=2, ctx=CTX):
        super().__init__()
        self.tok = nn.Embedding(V, C)
        self.pos = nn.Embedding(ctx, C)  # learned positions keep it short (RoPE from TODO 4 also works)
        self.blocks = nn.Sequential(*[Block(C, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(C)
        self.head = nn.Linear(C, V)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        return self.head(self.norm(self.blocks(x)))  # (B, T, V) logits


def next_token_loss(model, idx):
    """Teacher forcing: the logits at position t are scored against the token at t + 1."""
    # TODO: logits = model(idx[:, :-1]); targets = idx[:, 1:]
    #       return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    raise NotImplementedError("TODO 6: next_token_loss")


def batch(n=32):
    starts = torch.randint(0, len(data) - CTX - 1, (n,))
    return torch.stack([data[s : s + CTX + 1] for s in starts])


gpt = TinyGPT(V)
opt = torch.optim.AdamW(gpt.parameters(), lr=3e-3)
first = next_token_loss(gpt, batch()).item()
losses = []
for step in range(300):
    loss = next_token_loss(gpt, batch())
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
print(f"loss: start {first:.2f} (≈ ln {V} = {math.log(V):.2f}), end {losses[-1]:.2f}")
check(abs(first - math.log(V)) < 0.6, "an untrained model guesses about uniformly: loss ≈ ln(vocab)")
check(losses[-1] < 0.6, f"after 300 steps the loss is below 0.6 (perplexity {math.exp(losses[-1]):.2f})")


@torch.no_grad()
def generate(model, prompt, n=60):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):
        logits = model(idx[:, -CTX:])[:, -1]
        idx = torch.cat([idx, logits.argmax(-1, keepdim=True)], dim=1)  # greedy: take the most likely
    return "".join(chars[i] for i in idx[0])


sample = generate(gpt, "the cat ")
print("sample:", repr(sample))
check(sample.startswith("the cat s"), "it learned the text: 'the cat ' continues with 's…'")

plt.figure(figsize=(6, 3))
plt.plot(losses, color="teal")
plt.axhline(math.log(V), ls="--", color="gray", label="uniform guess ln V")
plt.xlabel("step"); plt.ylabel("next-token loss"); plt.legend(); plt.tight_layout(); show()


# %% TODO 7: KV-cache and Chinchilla arithmetic (Lessons 6.8, 6.9)
def kv_cache_bytes(layers, kv_heads, head_dim, tokens, batch=1, bytes_per_value=2):
    """Memory for the keys AND values of every layer, head and token."""
    # TODO: 2 (K and V) × layers × kv_heads × head_dim × tokens × batch × bytes_per_value
    raise NotImplementedError("TODO 7: kv_cache_bytes")


def chinchilla(compute_flops, tokens_per_param=20):
    """Compute-optimal (params, tokens) for a FLOP budget, using C ≈ 6·N·D and D = 20·N."""
    # TODO: C = 6·N·(20·N) = 120·N²  →  N = sqrt(C / (6 · tokens_per_param)); D = tokens_per_param · N
    raise NotImplementedError("TODO 7: chinchilla")


GiB = 2**30
check(kv_cache_bytes(32, 8, 128, 1) == 131072, "Llama 3 8B: 128 KiB of KV cache per token in bf16")
check(kv_cache_bytes(32, 8, 128, 131072) / GiB == 16, "… so 16 GiB at a 128k context")
check(kv_cache_bytes(32, 32, 128, 8192) / kv_cache_bytes(32, 8, 128, 8192) == 4, "GQA with 8 K/V heads instead of 32: a 4× smaller cache")
n_opt, d_opt = chinchilla(6 * 70e9 * 1.4e12)
check(abs(n_opt / 1e9 - 70) < 1 and abs(d_opt / 1e12 - 1.4) < 0.02, f"Chinchilla's budget → {n_opt / 1e9:.0f}B params, {d_opt / 1e12:.2f}T tokens")
n24, d24 = chinchilla(1e24)
print(f"1e24 FLOPs → about {n24 / 1e9:.0f}B params on {d24 / 1e12:.1f}T tokens")
check(85e9 < n24 < 95e9, "a 1e24 budget is compute-optimal at roughly 90B parameters")

print(f"\n🎉 Lab 6 done in {time.time() - t_start:.1f} s")


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Remove `causal_mask(T)` from `Block.forward` and retrain TinyGPT. The training loss drops much
#    faster. Why is that a bug, and what does `generate` print now? (Lesson 6.7)
# 2. Replace the learned position embedding with `rope` (TODO 4) applied to q and k inside the block.
#    Does it still learn? What happens if you generate past CTX characters?
# 3. Add a KV cache to `generate`: keep each layer's k and v, and feed only the newest token.
#    Check that it produces exactly the same text, then time both. (Lesson 6.9)
# 4. Use `kv_cache_bytes` to find the largest batch of 32k-token chats Llama 3 70B (80 layers,
#    8 K/V heads, head dim 128) can serve with 40 GiB left for the cache, in bf16 and in fp8.

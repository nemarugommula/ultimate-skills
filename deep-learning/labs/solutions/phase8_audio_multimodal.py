# %% [markdown]
# # Lab 8: Audio, speech & multimodal from scratch 🎧
#
# You'll build, with NumPy and a little PyTorch, the pieces behind Whisper, TTS, audio LLMs and VLMs:
# synthetic tones and chirps, an STFT by hand (checked against `torch.stft`), a mel filterbank and
# log-mel features, CTC greedy decoding plus `nn.CTCLoss` on a toy, the CTC alignment count,
# residual vector quantization (the trick inside neural audio codecs), and image-to-token counting
# with a LLaVA-style projector shape check.
#
# Setup (once, from the deep-learning folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase8_audio_multimodal.py`.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Everything is synthetic and runs on CPU in a few seconds. No downloads.
# Lessons to reread if stuck: 8.1 (spectrograms, mel), 8.2 (CTC), 8.4 (RVQ, tokens per second),
# 8.5 (patches, projectors)

# %%
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

rng = np.random.default_rng(0)
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


SR = 16000  # samples per second, like Whisper


# %% TODO 1: synthesize a tone and a chirp (Lesson 8.1)
def tone(freq, sr=SR, dur=1.0, amp=0.5):
    """A sine wave at `freq` Hz lasting `dur` seconds: amp * sin(2π f t) for t = 0, 1/sr, 2/sr, ..."""
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def chirp(f0, f1, sr=SR, dur=1.0, amp=0.5):
    """A tone whose pitch glides linearly from f0 to f1 Hz. The phase is the running sum of 2π f(t) / sr."""
    n = int(sr * dur)
    f = np.linspace(f0, f1, n)
    phase = np.cumsum(2 * np.pi * f / sr)
    return amp * np.sin(phase)


x440 = tone(440)
check(x440.shape == (16000,), "one second at 16 kHz is 16,000 samples")
check(abs(x440[0]) < 1e-12 and abs(np.max(x440) - 0.5) < 1e-3, "the tone starts at 0 and peaks at amp = 0.5")
xc = chirp(500, 3000)
zc_start = np.sum(np.diff(np.sign(xc[:1600])) != 0)   # zero crossings ≈ 2 × frequency × time
zc_end = np.sum(np.diff(np.sign(xc[-1600:])) != 0)
check(zc_end > 3 * zc_start, "the chirp wiggles much faster at the end than at the start")


# %% TODO 2: the STFT by hand, checked against torch.stft (Lesson 8.1)
def stft_mag(x, n_fft=400, hop=160):
    """Magnitude spectrogram, shape (n_fft // 2 + 1, frames).
    For each frame start s = 0, hop, 2·hop, ... while s + n_fft <= len(x):
    take x[s : s + n_fft], multiply by a Hann window, take np.fft.rfft, keep np.abs."""
    win = np.hanning(n_fft + 1)[:-1]   # periodic Hann window, same as torch.hann_window(n_fft)
    frames = [np.abs(np.fft.rfft(x[s:s + n_fft] * win)) for s in range(0, len(x) - n_fft + 1, hop)]
    return np.stack(frames, axis=1)


S = stft_mag(x440)
check(S.shape == (201, 98), "25 ms window, 10 ms hop: 201 frequency bins × 98 full frames")
peak_bin = int(S.mean(axis=1).argmax())
check(peak_bin == 11, "440 Hz lands in bin 440 / (16000 / 400) = 11")
T = torch.stft(torch.tensor(x440), n_fft=400, hop_length=160, window=torch.hann_window(400, dtype=torch.float64),
               center=False, return_complex=True).abs().numpy()
check(T.shape == S.shape and np.allclose(T, S, atol=1e-6), "your STFT matches torch.stft(center=False)")

Sc = stft_mag(xc)
peaks = Sc.argmax(axis=0) * SR / 400
check(peaks[-1] - peaks[0] > 2000, "the chirp's loudest frequency rises by more than 2,000 Hz")
plt.imshow(np.log(Sc + 1e-6), origin="lower", aspect="auto"); plt.title("chirp: a rising line"); show()


# %% TODO 3: a mel filterbank (Lesson 8.1)
def hz_to_mel(f):
    """The HTK mel formula: 2595 · log10(1 + f / 700)."""
    return 2595 * np.log10(1 + np.asarray(f, dtype=float) / 700)


def mel_to_hz(m):
    return 700 * (10 ** (np.asarray(m, dtype=float) / 2595) - 1)


def mel_filterbank(n_mels=80, n_fft=400, sr=SR, fmin=0.0, fmax=None):
    """Matrix of shape (n_mels, n_fft // 2 + 1). Row m is a triangle that rises from edge m to a peak at
    edge m + 1 and falls to zero at edge m + 2, where the n_mels + 2 edges are evenly spaced in mels."""
    fmax = sr / 2 if fmax is None else fmax
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rise = (freqs - lo) / (mid - lo)
        fall = (hi - freqs) / (hi - mid)
        fb[m] = np.clip(np.minimum(rise, fall), 0, None)
    return fb


check(abs(hz_to_mel(1000) - 1000) < 0.1, "1,000 Hz ≈ 1,000 mels (that's how the scale was anchored)")
check(abs((hz_to_mel(200) - hz_to_mel(100)) / (hz_to_mel(7100) - hz_to_mel(7000)) - 9.13) < 0.1,
      "100 → 200 Hz is ~9× more mels than 7,000 → 7,100 Hz")
FB = mel_filterbank()
check(FB.shape == (80, 201), "80 mel filters over 201 FFT bins")
widths = (FB > 0).sum(axis=1)
check(widths[-1] > 5 * max(1, widths[5]), "high-frequency triangles are much wider than low ones")
try:
    from transformers.audio_utils import mel_filter_bank as hf_mel   # optional cross-check (no download)
    ref = hf_mel(num_frequency_bins=201, num_mel_filters=80, min_frequency=0.0, max_frequency=8000.0,
                 sampling_rate=SR, norm=None, mel_scale="htk").T
    check(np.allclose(ref, FB, atol=1e-6), "matches transformers' HTK mel filterbank")
except ImportError:
    print("(transformers not installed: skipped the cross-check)")


# %% TODO 4: log-mel features, Whisper style (Lesson 8.1)
def log_mel(x, n_mels=80):
    """log10 of (mel filterbank @ power spectrogram), clamped to at most 8 orders of magnitude below its max
    (Whisper's recipe, before its final rescaling). Power = magnitude squared."""
    P = stft_mag(x) ** 2
    M = mel_filterbank(n_mels) @ P
    L = np.log10(np.maximum(M, 1e-10))
    return np.maximum(L, L.max() - 8.0)


LM = log_mel(xc + tone(300, amp=0.005))   # chirp plus a very quiet hum
check(LM.shape == (80, 98), "log-mel picture: 80 rows × 98 frames")
check(LM.max() - LM.min() <= 8.0 + 1e-9, "dynamic range clamped to 8 orders of magnitude")
hum_row = int(np.argmax(mel_filterbank()[:, int(300 / 40)]))
lin = mel_filterbank() @ stft_mag(xc + tone(300, amp=0.005)) ** 2
bright = (LM[hum_row].max() - LM.min()) / (LM.max() - LM.min())
check(lin[hum_row].max() / lin.max() < 1e-3 and bright > 0.4,
      f"the quiet hum is < 0.1% of the linear max, but {100 * bright:.0f}% of the way up the log display")
plt.imshow(LM, origin="lower", aspect="auto"); plt.title("log-mel: chirp + quiet hum"); show()


# %% TODO 5: CTC greedy decoding and nn.CTCLoss (Lesson 8.2)
def ctc_collapse(path, blank=0):
    """Merge consecutive repeats, then drop blanks. [0, 1, 1, 0, 2, 2, 0, 2] → [1, 2, 2]."""
    out, prev = [], None
    for s in path:
        if s != prev and s != blank:
            out.append(s)
        prev = s
    return out


check(ctc_collapse([0, 1, 1, 0, 2, 2, 0, 2]) == [1, 2, 2], "merge repeats, then drop blanks")
check(ctc_collapse([3, 3, 3]) == [3], "a long run is one symbol")
check(ctc_collapse([0, 0, 0]) == [], "all blanks decode to nothing")

vocab = ["_", "h", "e", "l", "o"]
Tn = 12
logits = torch.randn(Tn, 1, len(vocab), requires_grad=True)   # stands in for a network's outputs
target = torch.tensor([[1, 2, 3, 3, 4]])                       # h e l l o
ctc = nn.CTCLoss(blank=0)
opt = torch.optim.Adam([logits], lr=0.1)
first = None
for step in range(200):
    opt.zero_grad()
    loss = ctc(logits.log_softmax(-1), target, torch.tensor([Tn]), torch.tensor([5]))
    first = loss.item() if first is None else first
    loss.backward()
    opt.step()
best = logits.argmax(-1)[:, 0].tolist()
text = "".join(vocab[i] for i in ctc_collapse(best))
print("best path:", "".join(vocab[i] for i in best), "→", text, f"(loss {first:.2f} → {loss.item():.3f})")
check(text == "hello", "greedy decoding of the trained toy spells 'hello' (a blank separates the l's)")
check(loss.item() < 0.1 * first, "the CTC loss fell by more than 10×")


# %% TODO 6: count CTC alignments with dynamic programming (Lesson 8.2)
def count_alignments(label, T):
    """How many length-T frame strings over (blank + letters) collapse to `label`?
    Extended label: _ l1 _ l2 _ ... _. From state s you may stay, move to s+1, or skip to s+2 when
    the state s+2 is a letter different from the letter at s (you can't skip the blank between doubles)."""
    ext = ["_"]
    for c in label:
        ext += [c, "_"]
    S_ = len(ext)
    a = np.zeros(S_, dtype=object)
    a[0] = 1
    if S_ > 1:
        a[1] = 1
    for _ in range(1, T):
        b = np.zeros(S_, dtype=object)
        for s in range(S_):
            v = a[s] + (a[s - 1] if s > 0 else 0)
            if s > 1 and ext[s] != "_" and ext[s] != ext[s - 2]:
                v += a[s - 2]
            b[s] = v
        a = b
    return int(a[-1] + (a[-2] if S_ > 1 else 0))


check(count_alignments("cat", 4) == 7, "'cat' in 4 frames: 7 paths (_cat c_at ca_t cat_ ccat caat catt)")
check(count_alignments("hello", 5) == 0 and count_alignments("hello", 6) == 1, "'hello' needs ≥ 6 frames")
check(count_alignments("hello", 12) == 8008, "'hello' in 12 frames: 8,008 alignments")


# %% TODO 7: residual vector quantization (Lesson 8.4)
def kmeans(P, K, iters=25, seed=0):
    r = np.random.default_rng(seed)
    C = P[r.choice(len(P), K, replace=False)].copy()
    for _ in range(iters):
        idx = ((P[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if np.any(idx == k):
                C[k] = P[idx == k].mean(0)
    return C


def train_rvq(P, stages=4, K=8):
    """Train one codebook per stage with k-means on the residuals left by the previous stages."""
    books, R = [], P.copy()
    for s in range(stages):
        C = kmeans(R, K, seed=s)
        books.append(C)
        R = R - C[((R[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)]
    return books


def rvq_encode(x, books):
    """Return one code index per stage: at each stage pick the codeword nearest to the current residual,
    then subtract it from the residual."""
    codes, r = [], np.array(x, dtype=float)
    for C in books:
        k = int(((C - r) ** 2).sum(1).argmin())
        codes.append(k)
        r = r - C[k]
    return codes


def rvq_decode(codes, books):
    return sum(books[s][k] for s, k in enumerate(codes))


P = rng.uniform(-1, 1, size=(1500, 2))
books = train_rvq(P)
Q = rng.uniform(-1, 1, size=(500, 2))
errs = [np.mean([np.linalg.norm(q - rvq_decode(rvq_encode(q, books[:s]), books[:s])) for q in Q]) for s in range(1, 5)]
print("mean error with 1..4 stages:", np.round(errs, 3))
check(all(errs[i + 1] < errs[i] for i in range(3)), "every extra stage lowers the error")
check(errs[3] < errs[0] / 5, "4 stages are at least 5× better than 1")
check(len(rvq_encode(Q[0], books)) == 4, "one code per stage")
# tokens per second for real codecs: frames per second × codebooks
check(75 * 8 == 600 and 75 * 8 * 10 == 6000, "EnCodec 6 kbps: 75 Hz × 8 codebooks × 10 bits = 600 tokens/s")
check(12.5 * 8 * 11 == 1100 and 12.5 * 8 * 60 * 5 == 30000, "Mimi: 1.1 kbps, and a 5-minute call = 30,000 tokens")


# %% TODO 8: image tokens and a projector shape check (Lesson 8.5)
def patch_tokens(side, patch=14, merge=1):
    """Tokens for a square image of `side` pixels: (side // patch)² patches, divided by `merge`
    (merge = 4 for a 2×2 merge)."""
    return (side // patch) ** 2 // merge


def qwen2vl_tokens(h, w):
    """Qwen2-VL-style native resolution: round each side to a multiple of 28, then one token per 28×28 block
    (14-px patches merged 2×2). Ignores the min/max pixel caps."""
    return round(h / 28) * round(w / 28)


check(patch_tokens(336) == 576, "LLaVA-1.5: 336 px, patch 14 → 576 tokens")
check(patch_tokens(448, patch=16, merge=4) == 196, "448 px, patch 16, 2×2 merge → 196 tokens")
check(qwen2vl_tokens(224, 224) + 2 == 66, "Qwen2-VL's own example: 224×224 → 64 tokens + 2 markers = 66")
check(qwen2vl_tokens(1080, 1920) == 2691, "a Full-HD screenshot → 2,691 tokens")

img = torch.randn(2, 3, 336, 336)
patchify = nn.Conv2d(3, 1024, kernel_size=14, stride=14)
projector = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 4096))
with torch.no_grad():
    tok = projector(patchify(img).flatten(2).transpose(1, 2))
check(tuple(tok.shape) == (2, 576, 4096), "two images → (2, 576, 4096): 576 tokens each, in the LLM's width")
n_params = sum(p.numel() for p in projector.parameters())
check(n_params == 1024 * 4096 + 4096 + 4096 * 4096 + 4096, f"projector parameters = {n_params:,}")

print("\n🎉 Lab 8 complete: you built spectrograms, mel features, CTC, RVQ and image tokenization by hand.")

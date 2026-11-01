# %% [markdown]
# # Lab 7 — Reward model + DPO on a real (small) LLM
# Fill in the 3 TODOs. Each is followed by check() calls that tell you if you got it right.
# Setup: pip install "transformers>=4.45,<5" (the model, ~270 MB, downloads on first run).
# Solution: solutions/phase7_dpo.py
#
# Part A: a Bradley–Terry reward model learns (and inherits) labeler biases (seconds, torch).
# Part B: DPO from scratch on SmolLM2-135M-Instruct to make it concise (~5–30 min on a laptop CPU depending on the machine; ~1–2 min on a free Colab GPU).
#
# Colab (recommended): Runtime → Change runtime type → GPU. Upload this file (Files panel),
# then run:  !pip install "transformers>=4.45,<5"  and  !python phase7_dpo.py

# %%
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# %% Part A — TODO 1: the Bradley–Terry loss (Lesson 7.3)
def bt_loss(r_chosen, r_rejected):
    # P(chosen beats rejected) = sigmoid(r_chosen - r_rejected). Loss = mean negative log of that.
    # Hint: F.logsigmoid is the numerically stable log(sigmoid(x)).
    raise NotImplementedError("TODO 1: Bradley–Terry loss")


check(abs(bt_loss(torch.tensor([1.0]), torch.tensor([1.0])).item() - np.log(2)) < 1e-6, "equal scores → loss log 2 (50/50)")
check(bt_loss(torch.tensor([3.0]), torch.tensor([0.0])).item() < 0.05, "chosen far above rejected → loss ≈ 0")

# Synthetic answers: features [correct, polite, long]. True quality = 2*correct + 0.5*polite.
# Labelers are charmed by length (bias 0.8). Train a linear reward model on their choices.
def answers(n):
    return torch.stack([torch.bernoulli(torch.full((n,), 0.5)), torch.rand(n), torch.rand(n)], dim=1)

A, B = answers(2000), answers(2000)
true_gap = 2 * (A[:, 0] - B[:, 0]) + 0.5 * (A[:, 1] - B[:, 1]) + 0.8 * (A[:, 2] - B[:, 2])
a_wins = torch.bernoulli(torch.sigmoid(true_gap)).bool()
chosen, rejected = torch.where(a_wins[:, None], A, B), torch.where(a_wins[:, None], B, A)
rm = torch.nn.Linear(3, 1, bias=False)
opt = torch.optim.Adam(rm.parameters(), lr=0.05)
for _ in range(300):
    loss = bt_loss(rm(chosen).squeeze(-1), rm(rejected).squeeze(-1))
    opt.zero_grad(); loss.backward(); opt.step()
w = rm.weight.detach().squeeze()
print(f"learned reward weights: correct {w[0]:.2f}, polite {w[1]:.2f}, long {w[2]:.2f}")
check(w[2] > 0.4, "the reward model inherited the labelers' length bias (RL would now pad answers!)")


# %% Part B — DPO on a real LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
policy = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
ref = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
for p in ref.parameters():
    p.requires_grad_(False)

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
            ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"),
            ("Greece", "Athens"), ("Russia", "Moscow"), ("China", "Beijing"), ("India", "New Delhi"), ("Chile", "Santiago"),
            ("Cuba", "Havana"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Ireland", "Dublin"), ("Austria", "Vienna"),
            ("Portugal", "Lisbon"), ("Thailand", "Bangkok"), ("Argentina", "Buenos Aires"), ("Finland", "Helsinki")]
FILLER = ["What a wonderful question! I'm so glad you asked. ", "Great question! Let me explain this carefully for you. ",
          "Thank you for asking! This is something many people wonder about. "]


def pair(country, capital):
    prompt = tok.apply_chat_template([{"role": "user", "content": f"What is the capital of {country}?"}],
                                     add_generation_prompt=True, tokenize=False)
    chosen = f"{capital}."
    rejected = random.choice(FILLER) + f"The capital of {country} is {capital}, a beautiful and historic city that many people love to visit."
    return prompt, chosen, rejected


random.shuffle(CAPITALS)
train_pairs = [pair(*c) for c in CAPITALS[:18]]
test_pairs = [pair(*c) for c in CAPITALS[18:]]


# %% TODO 2 — sequence log-probability of a response (sum over response tokens only)
def seq_logprob(model, prompt, response):
    # 1. Tokenize prompt and (response + tok.eos_token) separately (add_special_tokens=False), concatenate.
    # 2. logits = model(ids).logits. The logits at position t predict token t+1, so drop the last position.
    # 3. log_softmax over the vocab, then .gather the log-prob of each actual next token.
    # 4. Keep only the positions that predict RESPONSE tokens and sum them. (Careful with the off-by-one!)
    raise NotImplementedError("TODO 2: sequence log-prob of the response")


@torch.no_grad()
def brute_force_logprob(model, prompt, response):
    # The slow, obviously-correct way: one forward pass per response token, reading only the LAST position.
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    r_ids = tok(response + tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p_ids, r_ids], dim=1).to(device)
    total = 0.0
    for k in range(p_ids.shape[1], ids.shape[1]):
        last = model(ids[:, :k]).logits[0, -1].float()          # scores for the token at position k
        total += torch.log_softmax(last, dim=-1)[ids[0, k]].item()
    return total


france = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}],
                                 add_generation_prompt=True, tokenize=False)
with torch.no_grad():
    fast = seq_logprob(ref, france, "Paris.").item()
    banana = seq_logprob(ref, france, "Banana.").item()
slow = brute_force_logprob(ref, france, "Paris.")
print(f"log π('Paris.') = {fast:.3f} (one pass) vs {slow:.3f} (token by token); log π('Banana.') = {banana:.3f}")
check(abs(fast - slow) < 1e-2, "seq_logprob matches the slow token-by-token sum (no off-by-one)")
check(fast > banana, "the model finds 'Paris.' far likelier than 'Banana.' for France's capital")


# %% TODO 3 — the DPO loss (Lesson 7.5)
def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    # margin = beta * [(log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected))]
    # loss = -log sigmoid(margin). Return (loss, margin).
    raise NotImplementedError("TODO 3: DPO loss")


l, m = dpo_loss(torch.tensor(-5.0), torch.tensor(-9.0), torch.tensor(-5.0), torch.tensor(-9.0))
check(abs(l.item() - np.log(2)) < 1e-6 and m.item() == 0, "policy == reference → margin 0, loss log 2")
l2, _ = dpo_loss(torch.tensor(-4.0), torch.tensor(-12.0), torch.tensor(-5.0), torch.tensor(-9.0))
check(l2.item() < np.log(2), "raising chosen and lowering rejected (vs reference) lowers the loss")


@torch.no_grad()
def eval_pairs(pairs, beta=0.1):
    acc, avg_len = 0, 0
    for prompt, c, r in pairs:
        _, m = dpo_loss(seq_logprob(policy, prompt, c), seq_logprob(policy, prompt, r), seq_logprob(ref, prompt, c), seq_logprob(ref, prompt, r), beta)
        acc += m.item() > 0
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = policy.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        avg_len += out.shape[1] - enc.input_ids.shape[1]
    return acc / len(pairs), avg_len / len(pairs)


def sample_answer(prompt):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = policy.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


acc0, len0 = eval_pairs(test_pairs)
print(f"BEFORE DPO: held-out implicit-reward accuracy {acc0:.2f}, avg answer length {len0:.1f} tokens")
print("  sample:", repr(sample_answer(test_pairs[0][0])))

# %% DPO training loop (given)
opt = torch.optim.AdamW(policy.parameters(), lr=5e-6)
policy.train()
t0 = time.time()
for epoch in range(3):
    random.shuffle(train_pairs)
    for i in range(0, len(train_pairs), 3):
        losses = []
        for prompt, c, r in train_pairs[i:i + 3]:
            with torch.no_grad():
                ref_c, ref_r = seq_logprob(ref, prompt, c), seq_logprob(ref, prompt, r)
            loss, _ = dpo_loss(seq_logprob(policy, prompt, c), seq_logprob(policy, prompt, r), ref_c, ref_r)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch + 1}: last batch loss {loss.item():.3f}  [{time.time() - t0:.0f}s]")
policy.eval()

acc1, len1 = eval_pairs(test_pairs)
print(f"AFTER DPO:  held-out implicit-reward accuracy {acc1:.2f}, avg answer length {len1:.1f} tokens")
print("  sample:", repr(sample_answer(test_pairs[0][0])))
# Honest caveat: accuracy near 1.00 is near-trivial here, because every rejected answer shares the same
# filler template ("Great question!…"), so separating the pairs is easy. Shorter answers are the real test.
check(acc1 >= 0.8, "DPO's implicit reward ranks held-out pairs correctly (≥ 80%)")
check(len1 < len0, "answers got shorter: the model learned the 'be concise' preference")

# %% [markdown]
# # Lab 8 — SOLUTION: GRPO from scratch on a real (small) LLM
# Try the exercise file first!
#
# Best on a free Colab GPU (a few minutes). On a laptop CPU it takes ~15–50 min depending on load.
# Model: HuggingFaceTB/SmolLM2-135M-Instruct (~270 MB download the first time).

# %%
import random
import re
import time
import warnings

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
policy = AutoModelForCausalLM.from_pretrained(MODEL).to(device)


def make_prompt(a, b):
    msgs = [{"role": "user", "content": f"What is {a}+{b}? Answer with just the number."}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def make_task():
    a, b = random.randint(1, 30), random.randint(1, 30)
    return make_prompt(a, b), a + b


# %% TODO 1 — the verifier (RLVR reward)
def reward_fn(text, answer):
    """1.0 if the LAST number in the response equals the answer; +0.2 bonus if the response
    is exactly the number (clean format). Otherwise 0."""
    nums = re.findall(r"-?\d+", text)
    if not nums or int(nums[-1]) != answer:
        return 0.0
    return 1.0 + (0.2 if text.strip() == str(answer) else 0.0)


check(reward_fn("12", 12) == 1.2, "clean correct answer → 1.2")
check(reward_fn("7 + 5 = 12", 12) == 1.0, "correct final number with extra text → 1.0")
check(reward_fn("7 + 5 = 13", 12) == 0.0, "wrong → 0")
check(reward_fn("I don't know", 12) == 0.0, "no number → 0")
check(reward_fn("12 or maybe 13", 12) == 0.0, "hedging with a wrong last guess → 0 (but '13 or 12' still fools this rule: see 8.2)")


# %% TODO 2 — group-relative advantages (GRPO, Lesson 8.3)
def group_advantages(rewards, eps=1e-4):
    """rewards: tensor (n_prompts, G). Return (r - mean) / (std + eps) within each group."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
check(torch.allclose(adv[0], torch.tensor([0.866, -0.866, -0.866, 0.866]), atol=1e-3), "correct answers get positive advantage, wrong get negative")
check(torch.allclose(adv[1], torch.zeros(4)), "all-equal group → zero advantage (no learning signal)")


# %% TODO 3 — per-token log-probs of the response tokens (Lesson 7.1)
def response_logprobs(model, seqs, prompt_len, attn):
    """seqs: (B, T) prompt+response token ids (left-padded prompts of equal length prompt_len).
    Return (B, T - prompt_len) log π(token) for each response token."""
    logits = model(input_ids=seqs, attention_mask=attn).logits[:, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_logp[:, prompt_len - 1:]


# %% TODO 4 — the GRPO loss (PPO-clip with group advantages, token-level averaging like DAPO)
def grpo_loss(new_logp, old_logp, adv, mask, clip_eps=0.2):
    """new_logp/old_logp: (B, L) per-token log-probs; adv: (B,) one advantage per response;
    mask: (B, L) 1 for real response tokens. Average over all real tokens in the batch.
    NOTE: with inner_epochs=1 (the default in train_grpo) new_logp == old_logp, so ratio == 1 and the clip
    never fires: training is REINFORCE with a group-mean baseline (Lesson 8.3). The clip only matters if you
    set inner_epochs >= 2 (try it: the loop then prints the fraction of clipped tokens)."""
    ratio = torch.exp(new_logp - old_logp)
    a = adv.unsqueeze(1)
    per_tok = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    return (per_tok * mask).sum() / mask.sum()


nl = torch.log(torch.tensor([[0.3, 0.5]])).requires_grad_(True)
ol = torch.log(torch.tensor([[0.2, 0.5]]))
l = grpo_loss(nl, ol, torch.tensor([1.0]), torch.ones(1, 2))
l.backward()
check(abs(l.item() - (-(1.2 + 1.0) / 2)) < 1e-5, "token 1 clipped at 1.2, token 2 ratio 1.0 → loss = −1.1")
check(abs(nl.grad[0, 0].item()) < 1e-9 and nl.grad[0, 1].item() < 0, "clipped token gets zero gradient, unclipped token still learns")


# %% Evaluation helper (given)
@torch.no_grad()
def evaluate(model, n=64, seed=123, sample=False):
    """Accuracy and clean-format rate on n held-out problems. sample=False: greedy decoding (the model's
    single most likely answer). sample=True: temperature-1 sampling, the distribution GRPO trains."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tasks = [(rng.randint(1, 30), rng.randint(1, 30)) for _ in range(n)]
    correct, clean = 0, 0
    for i in range(0, n, 16):
        batch = tasks[i:i + 16]
        enc = tok([make_prompt(a, b) for a, b in batch], return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=12, do_sample=sample, temperature=1.0 if sample else None,
                             top_p=1.0 if sample else None, pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (a, b), t in zip(batch, texts):
            r = reward_fn(t, a + b)
            correct += r >= 1.0
            clean += r > 1.0
    return correct / n, clean / n


# %% GRPO training loop (given, but read it!)
def train_grpo(steps=50, prompts_per_step=4, G=8, lr=2e-5, max_new_tokens=12, inner_epochs=1):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        tasks = [make_task() for _ in range(prompts_per_step)]
        prompts = [p for p, _ in tasks for _ in range(G)]
        answers = [a for _, a in tasks for _ in range(G)]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        P = enc["input_ids"].shape[1]
        policy.eval()
        with torch.no_grad():  # 1. sample a group of G answers per prompt from the current policy
            seqs = policy.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                                   pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(seqs[:, P:], skip_special_tokens=True)
        rewards = torch.tensor([reward_fn(t, a) for t, a in zip(texts, answers)]).view(prompts_per_step, G)
        adv = group_advantages(rewards).view(-1).to(device)          # 2. score with the verifier, group-normalize
        resp = seqs[:, P:]
        # mask: response tokens up to and including the first EOS
        is_eos = resp == tok.eos_token_id
        after_eos = torch.cumsum(is_eos.int(), dim=1) - is_eos.int() > 0
        mask = (~after_eos).float()
        attn = torch.cat([enc["attention_mask"], mask.long()], dim=1)
        # old_logp is computed in eval() mode, new_logp below in train() mode. They only differ if the model
        # uses dropout; if it did, old and new would disagree before any update: a mini train/inference
        # mismatch (Lesson 8.11).
        with torch.no_grad():
            old_logp = response_logprobs(policy, seqs, P, attn)
        policy.train()
        # Skips the update only when the WHOLE batch has identical rewards. Individual all-same groups still
        # get zero advantage and add nothing; DAPO's dynamic sampling instead resamples per group (Lesson 8.4).
        keep = rewards.view(-1).std() > 0
        clip_frac = None
        if keep:
            for _ in range(inner_epochs):                              # 3. PPO-style clipped updates (reuse the batch)
                new_logp = response_logprobs(policy, seqs, P, attn)
                loss = grpo_loss(new_logp, old_logp, adv, mask)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                if inner_epochs > 1:                                   # share of real tokens outside [1-eps, 1+eps]
                    with torch.no_grad():
                        outside = ((torch.exp(new_logp - old_logp) - 1).abs() > 0.2).float()
                        clip_frac = ((outside * mask).sum() / mask.sum()).item()
        if step % 5 == 0 or step == 1:
            acc = (rewards >= 1.0).float().mean().item()
            extra = f"  clip frac {clip_frac:.2f}" if clip_frac is not None else ""
            print(f"step {step:3d}  train reward {rewards.mean():.2f}  train accuracy {acc:.2f}  "
                  f"mean length {mask.sum(1).mean():.1f} tok{extra}  [{time.time() - t0:.0f}s]  e.g. {texts[0]!r}")


acc0, clean0 = evaluate(policy)
sacc0, _ = evaluate(policy, sample=True)
print(f"BEFORE: greedy accuracy {acc0:.2f} (clean-format {clean0:.2f}) | sampled (T=1) accuracy {sacc0:.2f}")
train_grpo()
acc1, clean1 = evaluate(policy)
sacc1, _ = evaluate(policy, sample=True)
print(f"AFTER:  greedy accuracy {acc1:.2f} (clean-format {clean1:.2f}) | sampled (T=1) accuracy {sacc1:.2f}")
check(sacc1 > sacc0 + 0.2, "GRPO made sampled answers far more reliable (> +20 points at T=1)")
check(abs(acc1 - sacc1) < 0.5 * abs(acc0 - sacc0), "sampled accuracy converged toward greedy: the distribution SHARPENED")
print(f"Greedy accuracy changed by {acc1 - acc0:+.2f}; with 64 problems the standard error is about ±0.06, so small changes are noise.")
# The model's single most likely answer was often already right. RL mostly SHARPENED the distribution so
# that samples agree with it (Lessons 8.7–8.8). Notice the clean-format bonus never kicks in (0.00): the model
# never samples a bare number, so that reward is never seen. That's an exploration problem! Research ideas:
# more steps / a bigger model / harder problems (3-digit numbers) to move greedy accuracy; a format hint in
# the prompt so the bonus is sometimes sampled; pass@k before vs after (Lab 9's pass_at_k).

# %% [markdown]
# # Lab 5: REINFORCE and PPO from scratch 🎯
#
# Build the pieces of every modern policy-gradient method (returns, the REINFORCE loss
# with a baseline, GAE, and PPO's clipped loss), then train both on CartPole.
#
# Needs torch + gymnasium (installed in Lab 4). Runs on a laptop CPU in a few minutes.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons: 5.2 (REINFORCE), 5.3 (baselines), 5.5 (GAE), 5.6 (PPO)

# %%
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


# %% TODO 1: discounted returns (Lesson 0.3)
def discounted_returns(rewards, gamma):
    """Return [G_0, G_1, ...] using the backward loop G = r + gamma * G."""
    raise NotImplementedError("TODO 1: discounted_returns")


check(np.allclose(discounted_returns([0, 0, 10], 0.9), [8.1, 9.0, 10.0]), "discounted_returns([0,0,10], 0.9) = [8.1, 9, 10]")


# %% TODO 2: REINFORCE loss with a baseline (Lessons 5.2–5.3)
def reinforce_loss(log_probs, returns, baseline):
    """log_probs: tensor (T,) of log π(a_t|s_t); returns: tensor (T,); baseline: tensor or float.
    advantage = returns - baseline;  loss = -(log_probs * advantage).sum()"""
    raise NotImplementedError("TODO 2: reinforce_loss")


lp = torch.log(torch.tensor([0.5, 0.25]))
check(abs(reinforce_loss(lp, torch.tensor([2.0, 2.0]), 2.0).item()) < 1e-9, "returns equal to the baseline → zero loss (no learning signal)")
check(reinforce_loss(lp, torch.tensor([3.0, 1.0]), 0.0).item() > 0, "positive returns → loss = −Σ G·log π (positive, since log π < 0)")


def train_reinforce(episodes=600, lr=1e-3, gamma=0.99, seed=SEED, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    policy = mlp(4, 2)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        log_probs, rewards, done = [], [], False
        while not done:
            dist = Categorical(logits=policy(torch.as_tensor(s, dtype=torch.float32)))
            a = dist.sample()
            s, r, term, trunc, _ = env.step(a.item())
            log_probs.append(dist.log_prob(a)); rewards.append(r)
            done = term or trunc
        G = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
        # baseline = this episode's mean return; dividing by the std keeps the gradient scale steady.
        # (This bends Lesson 5.3's rule a little: G.mean() includes each step's OWN return, so the baseline
        #  depends slightly on the action. Leaving that step out is exactly RLOO, Lesson 8.3.)
        loss = reinforce_loss(torch.stack(log_probs), G / (G.std() + 1e-8), G.mean() / (G.std() + 1e-8))
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(sum(rewards))
        if verbose and ep % 100 == 0:
            print(f"episode {ep:4d}  avg return (last 20) {np.mean(history[-20:]):6.1f}")
    return policy, history


t0 = time.time()
_, hist_rf = train_reinforce()
print(f"REINFORCE: final avg return (last 50) {np.mean(hist_rf[-50:]):.1f}  [{time.time() - t0:.0f}s]")
check(np.mean(hist_rf[-50:]) >= 150, "REINFORCE learns to balance (last-50 avg ≥ 150)")
# 🔬 Try lr=1e-2: returns often climb to ~200 and then COLLAPSE to ~10 and never recover.
#    One oversized step made the policy nearly deterministic and wrong. That's the problem PPO solves.


# %% TODO 3: GAE (Lesson 5.5)
def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """rewards, dones: length T. values: length T+1 (includes the value of the state after the last step).
    Backward loop:  nonterm = 1 - dones[t]
                    delta = rewards[t] + gamma * values[t+1] * nonterm - values[t]
                    last  = delta + gamma * lam * nonterm * last
    Return (advantages, advantages + values[:T]) as float32 numpy arrays."""
    raise NotImplementedError("TODO 3: gae")


A, _ = gae([0, 0, 0, 0, 0, 1], [0] * 7, [0, 0, 0, 0, 0, 1], gamma=1.0, lam=0.0)
check(np.allclose(A, [0, 0, 0, 0, 0, 1]), "λ=0 with an untrained critic: only the last step gets credit (Lesson 5.5)")
A, _ = gae([0, 0, 0, 0, 0, 1], [0] * 7, [0, 0, 0, 0, 0, 1], gamma=1.0, lam=1.0)
check(np.allclose(A, [1, 1, 1, 1, 1, 1]), "λ=1: every step gets the Monte Carlo advantage")
A, _ = gae([1.0], [0.0, 5.0], [0], gamma=0.5, lam=0.9)
check(abs(A[0] - 3.5) < 1e-6, "bootstraps from values[T] when the rollout is cut mid-episode")


# %% TODO 4: PPO clipped loss (Lesson 5.6)
def ppo_policy_loss(new_logp, old_logp, adv, clip_eps=0.2):
    """ratio = exp(new_logp - old_logp)
    return -mean( min(ratio * adv, clamp(ratio, 1-clip_eps, 1+clip_eps) * adv) )"""
    raise NotImplementedError("TODO 4: ppo_policy_loss")


old = torch.log(torch.tensor([0.2])); new = torch.log(torch.tensor([0.3])).requires_grad_(True)
loss = ppo_policy_loss(new, old, torch.tensor([1.0]))
loss.backward()
check(abs(loss.item() + 1.2) < 1e-6 and abs(new.grad.item()) < 1e-9, "ratio 1.5 with Â=+1 is clipped to 1.2 and has zero gradient (Lesson 5.6 quiz)")


# %% PPO training loop (given, but read it! the recipe from Lesson 5.6)
def train_ppo(total_steps=60_000, rollout=1024, epochs=10, minibatch=64, lr=3e-4, gamma=0.99, lam=0.95,
              clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, seed=SEED, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    actor, critic = mlp(4, 2), mlp(4, 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    s, _ = env.reset(seed=seed)
    ep_ret, history, step = 0.0, [], 0
    while step < total_steps:
        S, Aa, LP, R, D, V = [], [], [], [], [], []
        for _ in range(rollout):
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                dist = Categorical(logits=actor(st)); a = dist.sample(); v = critic(st).item()
            s2, r, term, trunc, _ = env.step(a.item())
            S.append(st); Aa.append(a); LP.append(dist.log_prob(a)); R.append(r); D.append(float(term)); V.append(v)
            ep_ret += r; s = s2; step += 1
            if term or trunc:
                history.append(ep_ret); ep_ret = 0.0
                if trunc:  # time limit: the episode didn't really end; bootstrap from the cut-off state
                    with torch.no_grad():
                        R[-1] += gamma * critic(torch.as_tensor(s2, dtype=torch.float32)).item()
                D[-1] = 1.0
                s, _ = env.reset()
        with torch.no_grad():
            V.append(critic(torch.as_tensor(s, dtype=torch.float32)).item())
        adv, targets = gae(R, V, D, gamma, lam)
        S, Aa, LP = torch.stack(S), torch.stack(Aa), torch.stack(LP)
        adv, targets = torch.as_tensor(adv), torch.as_tensor(targets)
        for _ in range(epochs):
            for idx in torch.randperm(rollout).split(minibatch):
                dist = Categorical(logits=actor(S[idx]))
                new_lp, ent = dist.log_prob(Aa[idx]), dist.entropy().mean()
                a_mb = adv[idx]; a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)   # advantage normalization
                pl = ppo_policy_loss(new_lp, LP[idx], a_mb, clip_eps)
                vl = ((critic(S[idx]).squeeze(-1) - targets[idx]) ** 2).mean()
                loss = pl + vf_coef * vl - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
                # 🩺 diagnostics (Lesson 5.6). On the FIRST minibatch of each rollout no step has happened yet,
                #    so the ratio should be exactly 1 (up to float rounding) and KL / clip fraction ≈ 0.
                with torch.no_grad():
                    log_ratio = new_lp - LP[idx]
                    approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()   # ≈ KL(old ‖ new), always ≥ 0
                    clip_frac = ((log_ratio.exp() - 1).abs() > clip_eps).float().mean().item()  # share of samples clipped
        if verbose and history and (step // rollout) % 5 == 0:
            print(f"step {step:6d}  avg return (last 10 episodes) {np.mean(history[-10:]):6.1f}  "
                  f"| approx KL {approx_kl:.4f}  clip frac {clip_frac:.2f}  entropy {ent.item():.2f}")
    return actor, history


t0 = time.time()
_, hist_ppo = train_ppo()
print(f"PPO: final avg return (last 20) {np.mean(hist_ppo[-20:]):.1f}  [{time.time() - t0:.0f}s]")
check(np.mean(hist_ppo[-20:]) >= 300, "PPO balances the pole well (last-20 avg ≥ 300)")
# 🩺 PPO < 300? Read the diagnostics printed above, in order:
#   1. approx KL per update: small (usually below ~0.02). Much bigger = steps too large: lower lr or epochs.
#   2. clip frac: stuck at 0 = the clip never engages; near 1 = the policy runs away from its data every epoch.
#   3. entropy starts near 0.69 (= log 2, a 50/50 coin over 2 actions) and should fall SLOWLY.
#      Near 0 early = it stopped exploring (raise ent_coef).
#   4. Old log-probs (LP) must come from the policy that collected the data, computed under no_grad.
#   5. Truncation (the 500-step limit) bootstraps from the critic; only `term` is a real ending.
#   6. Try 3 seeds before concluding anything.

plt.figure(figsize=(7, 3.5))
plt.plot(np.convolve(hist_rf, np.ones(20) / 20, "valid"), label="REINFORCE (with baseline)")
plt.plot(np.convolve(hist_ppo, np.ones(20) / 20, "valid"), label="PPO")
plt.xlabel("episode"); plt.ylabel("return (20-episode avg)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 🔬 Mini research: which PPO detail matters most?
# HYPOTHESIS first. Then rerun train_ppo with ONE change at a time, 3 seeds each:
#   clip_eps=10.0 (≈ no clipping) · ent_coef=0.0 · epochs=1 · lam=0.0 (pure TD advantages)
# Report mean ± spread of the last-20 return for each. Which one hurts most? Why?

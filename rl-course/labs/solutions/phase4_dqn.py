# %% [markdown]
# # Lab 4 — SOLUTION
# Try the exercise file first!

# %%
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=SEED)
print("observation:", obs, "| actions:", env.action_space.n)


# %% TODO 1 — the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.as_tensor(np.array(s), dtype=torch.float32),
                torch.as_tensor(a, dtype=torch.int64),
                torch.as_tensor(r, dtype=torch.float32),
                torch.as_tensor(np.array(s2), dtype=torch.float32),
                torch.as_tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


rb = ReplayBuffer(3)
for i in range(5):
    rb.add(np.zeros(4) + i, i % 2, 1.0, np.zeros(4), False)
check(len(rb) == 3, "buffer keeps only the most recent `capacity` items")
s, a, r, s2, d = rb.sample(2)
check(s.shape == (2, 4) and a.dtype == torch.int64, "sample returns batched tensors")


# %% TODO 2 — the Q-network
class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


check(QNet()(torch.zeros(5, 4)).shape == (5, 2), "QNet maps a batch of states to one Q-value per action")


# %% TODO 3 — the DQN loss (with optional Double DQN)
def dqn_loss(q_net, target_net, batch, gamma=0.99, double=False):
    s, a, r, s2, d = batch
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if double:
            a2 = q_net(s2).argmax(1, keepdim=True)          # online net chooses
            q_next = target_net(s2).gather(1, a2).squeeze(1)  # target net scores
        else:
            q_next = target_net(s2).max(1).values
        target = r + gamma * (1 - d) * q_next
    return F.smooth_l1_loss(q_sa, target)


qn, tn = QNet(), QNet()
tn.load_state_dict(qn.state_dict())
fake = (torch.zeros(3, 4), torch.tensor([0, 1, 0]), torch.ones(3), torch.zeros(3, 4), torch.tensor([1.0, 1.0, 1.0]))
with torch.no_grad():
    q0 = qn(torch.zeros(1, 4))[0]
expected = F.smooth_l1_loss(torch.stack([q0[0], q0[1], q0[0]]), torch.ones(3))
check(abs(dqn_loss(qn, tn, fake).item() - expected.item()) < 1e-6, "terminal transitions use target = r (no bootstrap)")


# %% Training loop (given, but read it! it's the pseudocode from Lesson 4.3)
def train_dqn(total_steps=50_000, double=False, use_target=True, seed=SEED, lr=2.5e-4,
              buffer_size=50_000, batch_size=128, gamma=0.99, target_every=500,
              eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000, warmup=1_000, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)  # the action space has its OWN random generator: seed it too, or runs aren't reproducible
    q_net, target_net = QNet(), QNet()
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)
    s, _ = env.reset(seed=seed)
    ep_ret, returns, t0 = 0.0, [], time.time()
    for step in range(total_steps):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, r, term, trunc, _ = env.step(a)
        buffer.add(s, a, r, s2, float(term))   # only true terminals stop bootstrapping (not time limits)
        s, ep_ret = s2, ep_ret + r
        if term or trunc:
            returns.append(ep_ret)
            s, _ = env.reset()
            ep_ret = 0.0
        if len(buffer) >= warmup:
            loss = dqn_loss(q_net, target_net if use_target else q_net, buffer.sample(batch_size), gamma, double)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            opt.step()
        if use_target and step % target_every == 0:
            target_net.load_state_dict(q_net.state_dict())
        if verbose and step % 5000 == 0 and returns:
            print(f"step {step:6d}  eps {eps:.2f}  avg return (last 10 episodes) {np.mean(returns[-10:]):6.1f}  [{time.time() - t0:.0f}s]")
    return q_net, returns


def evaluate(q_net, episodes=10, seed=123):
    env = gym.make("CartPole-v1")
    rets = []
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(s, dtype=torch.float32)).argmax())
            s, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
    return float(np.mean(rets))


q_net, returns = train_dqn()
score = evaluate(q_net)
print(f"Greedy evaluation over 10 episodes: {score:.1f} (max possible 500)")
# Scored 50–80? Re-run with another seed (train_dqn(seed=1)) before debugging: DQN is seed-sensitive.
check(score >= 80, "your DQN learned to balance (eval avg ≥ 80; random play gets ~22, max is 500)")

# 🩺 DQN isn't learning? Check these, in order:
#   1. The target is computed inside torch.no_grad() (or .detach()): no gradient flows through it.
#   2. Q-values level off at or below ~100. Reward is +1 per step and γ = 0.99, so the most a state
#      can be worth is 1 + 0.99 + 0.99² + … = 1/(1−γ) = 100. Q-values in the thousands = diverging.
#   3. ε actually reaches 0.05 (here at step 10,000); the printout shows it.
#   4. done comes from `term` only, not `term or trunc` (hitting the 500-step limit isn't a real ending).
#   5. Print the loss and the mean Q-value every 5k steps (add to the print in train_dqn: loss.item(), q_net(buffer.sample(256)[0]).mean()).
#   6. Try 3 seeds before concluding anything.

plt.figure(figsize=(7, 3.5))
plt.plot(returns, alpha=0.3, label="episode return")
plt.plot(np.convolve(returns, np.ones(10) / 10, "valid"), label="10-episode average")
plt.xlabel("episode"); plt.ylabel("return"); plt.title("DQN on CartPole"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# %% Ablation — what does the target network buy us? (3 seeds each, ~3 min)
results = {}
for name, kw in [("DQN", {}), ("Double DQN", {"double": True}), ("no target net", {"use_target": False})]:
    scores = [evaluate(train_dqn(seed=s, verbose=False, **kw)[0]) for s in range(3)]
    results[name] = scores
    print(f"{name:<14} eval scores per seed: {scores}  mean {np.mean(scores):.1f}")
# Our run (seeds 0-2): DQN [141, 139, 109], Double DQN [500, 500, 291], no target net [9, 9, 9].
# Double DQN clearly helped here; without the target network every seed collapsed below random play (~22). With only 3 seeds, be careful
# claiming a winner — check the spread (Phase 0 recap: mean vs std).

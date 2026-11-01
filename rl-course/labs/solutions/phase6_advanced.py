# %% [markdown]
# # Lab 6 — SOLUTION
# Try the exercise file first!

# %%
import random
import time

import matplotlib.pyplot as plt
import numpy as np

SEED = 0
random.seed(SEED); np.random.seed(SEED)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# %% Part A — Dyna-Q on Sutton & Barto's maze (Example 8.1). Environment given.
class Maze:
    """6x9 grid, start (2,0), goal (0,8), walls as in S&B Figure 8.2. Reward 1 at the goal, else 0."""
    R, C = 6, 9
    start, goal = (2, 0), (0, 8)
    walls = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def step(self, s, a):
        r, c = s[0] + self.MOVES[a][0], s[1] + self.MOVES[a][1]
        if not (0 <= r < self.R and 0 <= c < self.C) or (r, c) in self.walls:
            r, c = s
        return (r, c), (1.0 if (r, c) == self.goal else 0.0), (r, c) == self.goal


def egreedy(Q, s, eps):
    if random.random() < eps:
        return random.randrange(4)
    q = Q[s]
    return int(random.choice(np.flatnonzero(q == q.max())))


# TODO 1 — Dyna-Q: learn from real steps AND from imagined steps replayed from a learned model
def dyna_q(env, episodes=50, n_planning=5, alpha=0.1, gamma=0.95, eps=0.1):
    Q = np.zeros((env.R, env.C, 4))
    model = {}                      # (s, a) -> (r, s2, done): what happened last time
    steps_per_episode = []
    for _ in range(episodes):
        s, done, steps = env.start, False, 0
        while not done:
            a = egreedy(Q, s, eps)
            s2, r, done = env.step(s, a)
            Q[s][a] += alpha * (r + gamma * (0 if done else Q[s2].max()) - Q[s][a])   # real experience
            model[(s, a)] = (r, s2, done)                                               # learn the model
            for _ in range(n_planning):                                                 # plan: imagined experience
                (ps, pa), (pr, ps2, pdone) = random.choice(list(model.items()))
                Q[ps][pa] += alpha * (pr + gamma * (0 if pdone else Q[ps2].max()) - Q[ps][pa])
            s, steps = s2, steps + 1
        steps_per_episode.append(steps)
    return Q, steps_per_episode


res = {}
for n in [0, 5, 50]:
    runs = np.array([dyna_q(Maze(), n_planning=n)[1] for _ in range(20)])
    res[n] = runs.mean(0)
    print(f"n_planning={n:2d}: avg steps in episode 2 = {res[n][1]:6.1f}   episodes 10-50 avg = {res[n][9:].mean():5.1f}")
check(res[50][1] < res[0][1] / 3, "planning makes episode 2 far shorter (S&B Figure 8.2)")
check(res[0][9:].mean() > res[50][9:].mean(), "with no planning, learning is much slower")
plt.figure(figsize=(6, 3.5))
for n, v in res.items():
    plt.plot(range(2, 51), v[1:], label=f"{n} planning steps")
plt.ylim(0, 800); plt.xlabel("episode"); plt.ylabel("steps per episode"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# %% Part B — SAC on Pendulum (continuous actions). Needs torch + gymnasium (Part A doesn't).
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(SEED)
torch.set_num_threads(1)


class Actor(nn.Module):
    """Gaussian policy squashed by tanh into [-max_a, max_a]."""
    def __init__(self, obs_dim, act_dim, max_a, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.log_std = nn.Linear(hidden, act_dim), nn.Linear(hidden, act_dim)
        self.max_a = max_a

    def forward(self, s):
        h = self.body(s)
        mu, log_std = self.mu(h), self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)                  # reparameterization trick
        a = torch.tanh(u)
        # log-prob of the squashed action (change of variables for tanh)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a * self.max_a, logp


def critic(obs_dim, act_dim, hidden=256):
    return nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


# TODO 2 — the SAC losses
def sac_critic_loss(q1, q2, q1_t, q2_t, actor, batch, alpha, gamma=0.99):
    s, a, r, s2, d = batch
    with torch.no_grad():
        a2, logp2 = actor(s2)
        x2 = torch.cat([s2, a2], -1)
        q_next = torch.min(q1_t(x2), q2_t(x2)).squeeze(-1) - alpha * logp2     # twin min (TD3) + entropy bonus (SAC)
        target = r + gamma * (1 - d) * q_next
    x = torch.cat([s, a], -1)
    return F.mse_loss(q1(x).squeeze(-1), target) + F.mse_loss(q2(x).squeeze(-1), target)


def sac_actor_loss(q1, q2, actor, s, alpha):
    a, logp = actor(s)
    x = torch.cat([s, a], -1)
    return (alpha * logp - torch.min(q1(x), q2(x)).squeeze(-1)).mean()   # maximize Q + α·entropy


# Instant checks on a random batch (no environment, runs in a second)
_od, _ad = 3, 1                                     # Pendulum sizes: 3 observations, 1 torque
_actor, _q1, _q2, _q1t, _q2t = Actor(_od, _ad, 2.0), critic(_od, _ad), critic(_od, _ad), critic(_od, _ad), critic(_od, _ad)
_b = (torch.randn(8, _od), 4 * torch.rand(8, _ad) - 2, torch.randn(8), torch.randn(8, _od), torch.zeros(8))
_lq = sac_critic_loss(_q1, _q2, _q1t, _q2t, _actor, _b, alpha=0.2)
check(torch.is_tensor(_lq) and _lq.dim() == 0 and bool(torch.isfinite(_lq)), "critic loss is a finite scalar")
_lq.backward()
check(_q1[0].weight.grad is not None and _q2[0].weight.grad is not None, "critic loss trains both critics")
check(all(p.grad is None for p in _actor.parameters()), "critic loss sends no gradient into the actor (target built under torch.no_grad())")
_la = sac_actor_loss(_q1, _q2, _actor, _b[0], alpha=0.2)
check(torch.is_tensor(_la) and _la.dim() == 0 and bool(torch.isfinite(_la)), "actor loss is a finite scalar")
_la.backward()
check(_actor.mu.weight.grad is not None and _actor.mu.weight.grad.abs().sum().item() > 0,
      "actor loss gradient reaches the actor's mean layer (reparameterization trick works)")

# Stretch (optional, not checked): auto-tune α like modern SAC (Lesson 6.2) instead of fixing alpha=0.2.
#   log_alpha = torch.zeros(1, requires_grad=True); opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)
#   target_entropy = -act_dim                                  # a common default: −(number of action dims)
#   each update: _, logp = actor(s)
#                alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
#                step opt_alpha on alpha_loss, then use alpha = log_alpha.exp().item()


def train_sac(total_steps=15_000, seed=SEED, alpha=0.2, lr=3e-4, batch=256, tau=0.005, warmup=1_000, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    od, ad, max_a = env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0])
    actor, q1, q2 = Actor(od, ad, max_a), critic(od, ad), critic(od, ad)
    q1_t, q2_t = critic(od, ad), critic(od, ad)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    S, A, R, S2, D = (np.zeros((total_steps, od), np.float32), np.zeros((total_steps, ad), np.float32),
                      np.zeros(total_steps, np.float32), np.zeros((total_steps, od), np.float32), np.zeros(total_steps, np.float32))
    s, _ = env.reset(seed=seed)
    ep_ret, returns, t0 = 0.0, [], time.time()
    for t in range(total_steps):
        if t < warmup:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(s).unsqueeze(0))[0][0].numpy()
        s2, r, term, trunc, _ = env.step(a)
        S[t], A[t], R[t], S2[t], D[t] = s, a, r, s2, float(term)
        s, ep_ret = s2, ep_ret + r
        if term or trunc:
            returns.append(ep_ret); s, _ = env.reset(); ep_ret = 0.0
            if verbose and len(returns) % 10 == 0:
                print(f"step {t:6d}  episode {len(returns):3d}  avg return (last 10) {np.mean(returns[-10:]):7.1f}  [{time.time() - t0:.0f}s]")
        if t >= warmup:
            idx = np.random.randint(0, t + 1, batch)
            b = tuple(torch.as_tensor(x[idx]) for x in (S, A, R, S2, D))
            lq = sac_critic_loss(q1, q2, q1_t, q2_t, actor, b, alpha)
            opt_q.zero_grad(); lq.backward(); opt_q.step()
            la = sac_actor_loss(q1, q2, actor, b[0], alpha)
            opt_a.zero_grad(); la.backward(); opt_a.step()
            with torch.no_grad():
                for net, tgt in ((q1, q1_t), (q2, q2_t)):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - tau).add_(tau * p)
    return actor, returns


actor, rets = train_sac()
print(f"SAC: avg return of last 10 episodes = {np.mean(rets[-10:]):.1f}  (random ≈ -1200, solved ≈ -200)")
check(np.mean(rets[-10:]) > -400, "SAC learns to swing up and balance the pendulum (last-10 avg > -400)")
plt.figure(figsize=(6, 3.5)); plt.plot(rets); plt.xlabel("episode"); plt.ylabel("return"); plt.title("SAC on Pendulum"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

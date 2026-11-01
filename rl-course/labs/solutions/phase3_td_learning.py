# %% [markdown]
# # Lab 3 — SOLUTION
# Try the exercise file first!

# %%
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# Environments the agent can only SAMPLE (no access to P). Given.
class RandomWalk:
    """States 0..6. 0 and 6 are terminal. Start in 3. Reward +1 only when exiting right."""
    n_states = 7
    start = 3
    true_values = np.array([0, 1, 2, 3, 4, 5, 0]) / 6.0

    def step(self, s):
        s2 = s + random.choice([-1, 1])
        return s2, (1.0 if s2 == 6 else 0.0), s2 in (0, 6)


class Cliff:
    """4x12 Cliff Walking. Start bottom-left, goal bottom-right, cliff in between (-100, back to start)."""
    R, C, n_actions = 4, 12, 4
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    start, goal = (3, 0), (3, 11)

    def step(self, s, a):
        r, c = s
        nr = min(self.R - 1, max(0, r + self.MOVES[a][0]))
        nc = min(self.C - 1, max(0, c + self.MOVES[a][1]))
        if nr == 3 and 1 <= nc <= 10:
            return self.start, -100.0, False
        return (nr, nc), -1.0, (nr, nc) == self.goal


def rms(V):
    return float(np.sqrt(np.mean((V[1:6] - RandomWalk.true_values[1:6]) ** 2)))


# %% TODO 1 — Monte Carlo prediction (every-visit, constant α)
def mc_prediction(env, episodes, alpha):
    V = np.full(env.n_states, 0.5); V[0] = V[6] = 0.0
    errors = []
    for _ in range(episodes):
        s, traj, done = env.start, [], False
        while not done:
            s2, r, done = env.step(s)
            traj.append((s, r))
            s = s2
        G = 0.0
        for s, r in reversed(traj):
            G = r + G  # γ = 1
            V[s] += alpha * (G - V[s])
        errors.append(rms(V))
    return V, errors


# %% TODO 2 — TD(0) prediction
def td0_prediction(env, episodes, alpha, gamma=1.0):
    V = np.full(env.n_states, 0.5); V[0] = V[6] = 0.0
    errors = []
    for _ in range(episodes):
        s, done = env.start, False
        while not done:
            s2, r, done = env.step(s)
            V[s] += alpha * (r + gamma * V[s2] - V[s])  # V[terminal] stays 0
            s = s2
        errors.append(rms(V))
    return V, errors


env = RandomWalk()
V_td, _ = td0_prediction(env, 5000, 0.01)
V_mc, _ = mc_prediction(env, 5000, 0.002)  # MC is noisier, so it needs a smaller step size
check(rms(V_td) < 0.05, "TD(0) learns the true values 1/6..5/6 (RMS error < 0.05)")
check(rms(V_mc) < 0.05, "Monte Carlo learns them too")

runs = 100
td_curve = np.mean([td0_prediction(RandomWalk(), 100, 0.1)[1] for _ in range(runs)], axis=0)
mc_curve = np.mean([mc_prediction(RandomWalk(), 100, 0.02)[1] for _ in range(runs)], axis=0)
print(f"RMS error after 100 episodes: TD(α=0.1) {td_curve[-1]:.3f}   MC(α=0.02) {mc_curve[-1]:.3f}")
check(td_curve[-1] < mc_curve[-1], "on the random walk, TD beats MC after 100 episodes (S&B Example 6.2)")
plt.figure(figsize=(6, 3.5))
plt.plot(td_curve, label="TD(0) α=0.1"); plt.plot(mc_curve, label="MC α=0.02")
plt.xlabel("episodes"); plt.ylabel("RMS error"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# %% TODO 3 — SARSA and Q-learning (control)
def egreedy(Q, s, eps):
    if random.random() < eps:
        return random.randrange(4)
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(random.choice(best))


def sarsa(env, episodes, alpha=0.5, eps=0.1, gamma=1.0):
    Q = np.zeros((env.R, env.C, 4))
    returns = []
    for _ in range(episodes):
        s, G, done = env.start, 0.0, False
        a = egreedy(Q, s, eps)
        while not done:
            s2, r, done = env.step(s, a)
            a2 = egreedy(Q, s2, eps)
            target = r + (0.0 if done else gamma * Q[s2][a2])
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, G = s2, a2, G + r
        returns.append(G)
    return Q, returns


class OneStep:
    """Tiny test world: from (0, 0) any action ends the episode with reward -1."""
    R, C, start = 1, 2, (0, 0)

    def step(self, s, a):
        return (0, 1), -1.0, True


Q1, ret1 = sarsa(OneStep(), 1, alpha=0.5, eps=0.0)
check(ret1 == [-1.0] and Q1.min() == -0.5, "SARSA alone: one final step gives Q = 0 + 0.5 × (-1 - 0) = -0.5 (no bootstrap after done)")
Q_s1, ret_s1 = sarsa(Cliff(), 500)
check(np.mean(ret_s1[-100:]) > -60, "SARSA alone on the cliff: average return of the last 100 episodes > -60 (each fall costs -100)")


def q_learning(env, episodes, alpha=0.5, eps=0.1, gamma=1.0):
    Q = np.zeros((env.R, env.C, 4))
    returns = []
    for _ in range(episodes):
        s, G, done = env.start, 0.0, False
        while not done:
            a = egreedy(Q, s, eps)
            s2, r, done = env.step(s, a)
            target = r + (0.0 if done else gamma * Q[s2].max())
            Q[s][a] += alpha * (target - Q[s][a])
            s, G = s2, G + r
        returns.append(G)
    return Q, returns


def greedy_path(Q, env, max_steps=60):
    s, path = env.start, [env.start]
    for _ in range(max_steps):
        s, r, done = env.step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return path


cl = Cliff()
S_runs = [sarsa(cl, 500) for _ in range(30)]
Q_runs = [q_learning(cl, 500) for _ in range(30)]
s_ret = np.mean([r for _, r in S_runs], axis=0)
q_ret = np.mean([r for _, r in Q_runs], axis=0)
print(f"Avg return per episode (last 100, while still exploring ε=0.1): SARSA {s_ret[-100:].mean():.1f}   Q-learning {q_ret[-100:].mean():.1f}")
p_s, p_q = greedy_path(S_runs[0][0], cl), greedy_path(Q_runs[0][0], cl)
print("SARSA greedy path rows visited:", sorted(set(r for r, c in p_s)), "length", len(p_s) - 1)
print("Q-learning greedy path rows visited:", sorted(set(r for r, c in p_q)), "length", len(p_q) - 1)
check(s_ret[-100:].mean() > q_ret[-100:].mean(), "SARSA earns more ONLINE reward while exploring (safe path)")
check(len(p_q) - 1 == 13, "Q-learning's greedy path is the optimal 13-step edge path")
plt.figure(figsize=(6, 3.5))
k = 10
plt.plot(np.convolve(s_ret, np.ones(k) / k, "valid"), label="SARSA")
plt.plot(np.convolve(q_ret, np.ones(k) / k, "valid"), label="Q-learning")
plt.ylim(-120, 0); plt.xlabel("episode"); plt.ylabel("return during training"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# %% TODO 4 — n-step TD on the random walk
def nstep_td(env, episodes, alpha, n, gamma=1.0):
    V = np.full(env.n_states, 0.5); V[0] = V[6] = 0.0
    for _ in range(episodes):
        states, rewards = [env.start], [0.0]
        T, t, done = float("inf"), 0, False
        while True:
            if t < T:
                s2, r, done = env.step(states[t])
                states.append(s2); rewards.append(r)
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, int(min(tau + n, T)) + 1))
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1:
                break
            t += 1
    return V


for n in [1, 2, 4, 8]:
    err = np.mean([rms(nstep_td(RandomWalk(), 10, 0.2, n)) for _ in range(300)])
    print(f"n={n}: RMS error after 10 episodes (α=0.2) = {err:.3f}")
check(np.allclose(nstep_td(RandomWalk(), 3000, 0.01, 3)[1:6], RandomWalk.true_values[1:6], atol=0.06), "n-step TD converges to the true values")

# %% [markdown]
# # Lab 1 — SOLUTION
# Try the exercise file first!

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def argmax(x):
    """argmax with random tie-breaking (np.argmax always picks the first — a subtle bias!)."""
    x = np.asarray(x)
    return int(rng.choice(np.flatnonzero(x == x.max())))


# %% TODO 1 — the environment
class BernoulliBandit:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)
        self.k = len(self.probs)
        self.best = self.probs.max()

    def pull(self, a):
        return 1.0 if rng.random() < self.probs[a] else 0.0


b = BernoulliBandit([0.3, 0.8])
check(abs(np.mean([b.pull(0) for _ in range(10000)]) - 0.3) < 0.02, "machine 0 pays ~30% of the time")
check(set(b.pull(1) for _ in range(100)) <= {0.0, 1.0}, "rewards are 0 or 1")


# %% TODO 2 — the update rule
class Agent:
    def __init__(self, k, alpha=None, q_init=0.0):
        self.k = k
        self.Q = np.full(k, q_init, dtype=float)
        self.N = np.zeros(k)
        self.alpha = alpha
        self.t = 0

    def act(self):
        raise NotImplementedError

    def update(self, a, r):
        self.t += 1
        self.N[a] += 1
        step = 1.0 / self.N[a] if self.alpha is None else self.alpha
        self.Q[a] += step * (r - self.Q[a])


ag = Agent(2)
ag.update(0, 10); ag.update(0, 0)
check(ag.Q[0] == 5.0 and ag.N[0] == 2, "α = 1/N gives the exact average (10, 0 → 5)")
ag = Agent(2, alpha=0.5)
ag.update(1, 10); ag.update(1, 0)
check(ag.Q[1] == 2.5, "fixed α = 0.5: 0 → 5 → 2.5")


class Greedy(Agent):
    def act(self):
        return argmax(self.Q)


# %% TODO 3 — ε-greedy
class EpsilonGreedy(Agent):
    def __init__(self, k, epsilon=0.1, **kw):
        super().__init__(k, **kw)
        self.epsilon = epsilon

    def act(self):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.k))
        return argmax(self.Q)


e = EpsilonGreedy(3, epsilon=0.0); e.Q[:] = [0, 1, 0]
check(all(e.act() == 1 for _ in range(100)), "ε = 0 always exploits")
e = EpsilonGreedy(3, epsilon=1.0)
counts = np.bincount([e.act() for _ in range(3000)], minlength=3)
check(all(abs(c - 1000) < 150 for c in counts), "ε = 1 picks uniformly at random")


# %% TODO 4 — UCB
class UCB(Agent):
    def __init__(self, k, c=2.0, **kw):
        super().__init__(k, **kw)
        self.c = c

    def act(self):
        untried = np.flatnonzero(self.N == 0)
        if len(untried):
            return int(untried[0])
        return argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))


u = UCB(3)
picks = []
for _ in range(3):
    a = u.act(); picks.append(a); u.update(a, 0.0)
check(sorted(picks) == [0, 1, 2], "UCB tries every arm once first")
u = UCB(2); u.Q[:] = [0.5, 0.5]; u.N[:] = [100, 1]; u.t = 101
check(u.act() == 1, "UCB prefers the less-tried arm when estimates are equal")


# %% TODO 5 — Thompson sampling
class Thompson(Agent):
    def __init__(self, k, **kw):
        super().__init__(k, **kw)
        self.W = np.zeros(k)
        self.L = np.zeros(k)

    def act(self):
        return argmax(rng.beta(self.W + 1, self.L + 1))

    def update(self, a, r):
        super().update(a, r)
        if r > 0:
            self.W[a] += 1
        else:
            self.L[a] += 1


ts = Thompson(2); ts.W[:] = [50, 0]; ts.L[:] = [0, 50]
check(sum(ts.act() == 0 for _ in range(200)) >= 198, "Thompson exploits a clearly better arm")
ts.update(1, 1.0)
check(ts.W[1] == 1 and ts.Q[1] == 1.0, "Thompson updates its wins and Q")


# %% Experiment — the race (averaged over many random bandits)
def run(make_agent, n_runs=200, steps=1000, k=10):
    regret = np.zeros((n_runs, steps))
    for i in range(n_runs):
        bandit = BernoulliBandit(rng.random(k))
        agent = make_agent(k)
        cum = 0.0
        for t in range(steps):
            a = agent.act()
            agent.update(a, bandit.pull(a))
            cum += bandit.best - bandit.probs[a]
            regret[i, t] = cum
    return regret


def plot_regret(results, title):
    plt.figure(figsize=(8, 4.5))
    for name, reg in results.items():
        m = reg.mean(0)
        se = reg.std(0) / np.sqrt(len(reg))
        plt.plot(m, label=name)
        plt.fill_between(range(len(m)), m - 1.96 * se, m + 1.96 * se, alpha=0.2)  # 95% CI
    plt.xlabel("step"); plt.ylabel("cumulative regret (lower is better)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


results = {
    "greedy": run(lambda k: Greedy(k)),
    "ε-greedy 0.1": run(lambda k: EpsilonGreedy(k, 0.1)),
    "UCB c=0.5": run(lambda k: UCB(k, c=0.5)),
    "Thompson": run(lambda k: Thompson(k)),
}
for name, reg in results.items():
    print(f"{name:<14} final regret = {reg[:, -1].mean():6.1f} ± {1.96 * reg[:, -1].std() / np.sqrt(len(reg)):.1f}")
plot_regret(results, "Bandit race: 10 arms, 200 runs")


# %% Experiment 2 — a drifting bandit: 1/N vs fixed α (Lesson 1.2). Ready to run, ~2 s.
def run_drift(make_agent, n_runs=50, steps=2000, k=5, drift=0.02):
    reward = np.zeros((n_runs, steps))
    for i in range(n_runs):
        bandit = BernoulliBandit(np.full(k, 0.5))
        agent = make_agent(k)
        for t in range(steps):
            a = agent.act()
            reward[i, t] = bandit.pull(a)
            agent.update(a, reward[i, t])
            bandit.probs = np.clip(bandit.probs + drift * rng.standard_normal(k), 0.0, 1.0)
    return reward


drift = {
    "ε-greedy, α = 1/N": run_drift(lambda k: EpsilonGreedy(k, 0.1)),
    "ε-greedy, fixed α = 0.1": run_drift(lambda k: EpsilonGreedy(k, 0.1, alpha=0.1)),
}
plt.figure(figsize=(8, 4.5))
for name, rw in drift.items():
    print(f"{name:<24} avg reward, 2nd half = {rw[:, 1000:].mean():.3f}")
    plt.plot(np.convolve(rw.mean(0), np.ones(50) / 50, mode="valid"), label=name)
plt.xlabel("step"); plt.ylabel("average reward (smoothed)"); plt.title("Drifting bandit")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
check(drift["ε-greedy, fixed α = 0.1"][:, 1000:].mean() > drift["ε-greedy, α = 1/N"][:, 1000:].mean() + 0.05,
      "on a drifting bandit, fixed α = 0.1 beats 1/N (it forgets stale rewards)")


# %% Experiment 3 — optimistic greedy (Lesson 1.3). Ready to run, ~4 s.
def optimistic_greedy(k):
    agent = Greedy(k, q_init=1.0)
    agent.N[:] = 1  # count the optimistic start as one pretend pull
    return agent


optimism = {
    "greedy": results["greedy"],
    "greedy, q_init=1 (1/N erases it)": run(lambda k: Greedy(k, q_init=1.0)),
    "greedy, q_init=1 as a pretend pull": run(optimistic_greedy),
    "Thompson": results["Thompson"],
}
for name, reg in optimism.items():
    print(f"{name:<36} final regret = {reg[:, -1].mean():6.1f}")
plot_regret(optimism, "Optimistic starting values")
check(optimism["greedy, q_init=1 as a pretend pull"][:, -1].mean() < results["greedy"][:, -1].mean() / 3,
      "optimism alone removes most of greedy's regret")
# Typical result: plain q_init=1 still tries every arm once (an untried arm keeps its 1.0); the
# pretend pull keeps exploring a little longer and lands close to Thompson here. But once the
# early optimism is used up it never explores again, so it would fail on the drifting bandit.


# %% Mini research — does decaying ε beat fixed ε?
class DecayingEpsilonGreedy(Agent):
    def __init__(self, k, c=5.0, **kw):
        super().__init__(k, **kw)
        self.c = c

    def act(self):
        eps = min(1.0, self.c / (self.t + 1))
        if rng.random() < eps:
            return int(rng.integers(self.k))
        return argmax(self.Q)


research = {
    "ε = 0.1 fixed": results["ε-greedy 0.1"],
    "ε = 0.01 fixed": run(lambda k: EpsilonGreedy(k, 0.01)),
    "ε = 5/t decaying": run(lambda k: DecayingEpsilonGreedy(k, c=5.0)),
}
for name, reg in research.items():
    print(f"{name:<18} final regret = {reg[:, -1].mean():6.1f}")
plot_regret(research, "Fixed vs decaying ε")
# Typical conclusion: decaying ε explores early and exploits later, so its regret curve bends
# (sub-linear) while fixed ε keeps paying a constant exploration tax. But the decay schedule
# (c) matters a lot — too fast and it behaves like greedy. That sensitivity is exactly why
# UCB/Thompson (no schedule to tune) are preferred.

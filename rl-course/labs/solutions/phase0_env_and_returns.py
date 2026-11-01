# %% [markdown]
# # Lab 0 — SOLUTION
# Try the exercise file first. Looking here too early robs you of the learning!

# %%
import random
import statistics

random.seed(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# %% TODO 1 — discounted return
def discounted_return(rewards, gamma):
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


check(discounted_return([0, 0, 10], 0.5) == 2.5, "discounted_return([0,0,10], 0.5) == 2.5")
check(discounted_return([1, 1, 1], 1.0) == 3.0, "γ = 1 is a plain sum")
check(discounted_return([], 0.9) == 0.0, "empty episode has return 0")
check(abs(discounted_return([1] * 500, 0.9) - 10) < 1e-6, "1 forever with γ=0.9 → 10")


# %% TODO 2 — the environment
class GridWorld:
    """5x5 grid. Start bottom-left, 💎 top-right (+10), 🔥 lava (-10), every step -1."""

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
    NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(self, slippery=0.0, max_steps=50):
        self.size = 5
        self.start = (4, 0)
        self.walls = {(1, 1), (2, 1), (3, 3)}
        self.gem = (0, 4)
        self.lava = (1, 3)
        self.slippery = slippery
        self.max_steps = max_steps

    def reset(self):
        self.pos = self.start
        self.t = 0
        return self.pos

    def step(self, action):
        if random.random() < self.slippery:
            action = random.randrange(4)
        dr, dc = self.ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls:
            self.pos = (r, c)
        self.t += 1
        if self.pos == self.gem:
            return self.pos, 10, True
        if self.pos == self.lava:
            return self.pos, -10, True
        return self.pos, -1, self.t >= self.max_steps


env = GridWorld()
check(env.reset() == (4, 0), "reset() returns the start state")
check(env.step(3) == ((4, 0), -1, False), "moving left into the edge: stay put, reward -1")
check(env.step(0) == ((3, 0), -1, False), "moving up works")
env.pos = (0, 3)
check(env.step(1) == ((0, 4), 10, True), "reaching the gem gives +10 and ends the episode")
env.pos = (1, 2)
check(env.step(1) == ((1, 3), -10, True), "lava gives -10 and ends the episode")
env.pos = (1, 0)
check(env.step(1)[0] == (1, 0), "walls block movement")


# %% TODO 3 — the agent–environment loop
def run_episode(env, policy):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        a = policy(s)
        s, r, done = env.step(a)
        rewards.append(r)
    return rewards


def random_policy(state):
    return random.randrange(4)


rw = run_episode(GridWorld(), random_policy)
check(len(rw) > 0 and rw[-1] in (10, -10, -1) and all(r == -1 for r in rw[:-1]), "run_episode returns the list of rewards")


# %% TODO 4 — write a policy by hand (a dict: state -> action)
MY_POLICY = {
    (4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 0,
    (3, 4): 0, (2, 4): 0, (1, 4): 0,
}


def my_policy(state):
    return MY_POLICY.get(state, random.randrange(4))


check(sum(run_episode(GridWorld(), my_policy)) == 3, "your policy reaches 💎 by the shortest path (return = +3)")


# %% Experiment — compare policies (averaged over many episodes!)
def evaluate(env, policy, n=1000):
    """Run n episodes. mean = the typical return; std = how much returns vary between
    episodes (0 = identical every time). Explained in the Phase 0 recap: "Reading your results"."""
    returns = [sum(run_episode(env, policy)) for _ in range(n)]
    return statistics.mean(returns), statistics.stdev(returns)


print(f"\n{'policy':<10} {'floor':<10} {'mean return':>12} {'std':>8}")
for floor, slip in [("normal", 0.0), ("slippery", 0.2)]:
    for name, pol in [("random", random_policy), ("mine", my_policy)]:
        m, s = evaluate(GridWorld(slippery=slip), pol)
        print(f"{name:<10} {floor:<10} {m:>12.2f} {s:>8.2f}")

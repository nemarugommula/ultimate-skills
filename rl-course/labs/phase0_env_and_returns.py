# %% [markdown]
# # Lab 0: Your first RL environment 🤖
#
# You'll build the gridworld from Lesson 0.2 using a simplified version of the
# reset()/step() interface real RL libraries use (Gymnasium's step() also returns
# `truncated` and `info`), then measure some policies.
#
# How to use: run one cell at a time ("Run Cell" above each `# %%`),
# or run the whole file with `python3 phase0_env_and_returns.py`.
# Fill in each TODO until its ✅ checks pass. Solutions: labs/solutions/

# %%
import random
import statistics

random.seed(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# %% TODO 1: discounted return
# G = r_1 + γ·r_2 + γ²·r_3 + ...   (rewards[0] is r_1)
# Hint: walk BACKWARDS through the rewards using G = r + gamma * G
def discounted_return(rewards, gamma):
    G = 0
    for r in rewards[::-1]:
        G = r + gamma*G

    return G

check(discounted_return([0, 0, 10], 0.5) == 2.5, "discounted_return([0,0,10], 0.5) == 2.5")
check(discounted_return([1, 1, 1], 1.0) == 3.0, "γ = 1 is a plain sum")
check(discounted_return([], 0.9) == 0.0, "empty episode has return 0")
check(abs(discounted_return([1] * 500, 0.9) - 10) < 1e-6, "1 forever with γ=0.9 → 10")


# %% TODO 2: the environment
class GridWorld:
    """5x5 grid. Start bottom-left, 💎 top-right (+10), 🔥 lava (-10), every step -1.

        col: 0  1  2  3  4
    row 0:   .  .  .  .  💎
    row 1:   .  🧱 .  🔥 .
    row 2:   .  🧱 .  .  .
    row 3:   .  .  .  🧱 .
    row 4:   🤖 .  .  .  .
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
    NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(self, slippery=0.0, max_steps=50):
        self.size = 5
        self.start = (4, 0)
        self.walls = {(1, 1), (2, 1), (3, 3)}
        self.gem = (0, 4)
        self.lava = (1, 3)
        self.slippery = slippery  # probability the action is replaced by a random one
        self.max_steps = max_steps

    def reset(self):
        self.pos = self.start
        self.t = 0
        return self.pos

    def step(self, action):
        """Apply `action`. Return (next_state, reward, done).

        Rules:
          1. With probability self.slippery, replace action with a random one (0-3).
          2. Compute the new (row, col). If it's off the grid or a wall, stay put.
          3. self.t += 1
          4. Gem  → reward +10, done=True
             Lava → reward -10, done=True
             else → reward  -1, done = (self.t >= self.max_steps)
        """
        if self.slippery:
            action = random.randint(0,3)
        
        rc,cc = self.ACTIONS(action)
        
    
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


# %% TODO 3: the agent–environment loop
# Reset the env, then repeat: action = policy(state); state, r, done = env.step(action)
# Collect every reward in a list and return it when done.
def run_episode(env, policy):
    raise NotImplementedError("TODO 3: implement run_episode")


def random_policy(state):
    return random.randrange(4)


rw = run_episode(GridWorld(), random_policy)
check(len(rw) > 0 and rw[-1] in (10, -10, -1) and all(r == -1 for r in rw[:-1]), "run_episode returns the list of rewards")


# %% TODO 4: write a policy by hand (a dict: state -> action)
# Fill in the shortest route from (4,0) to the gem (0,4). Actions: 0↑ 1→ 2↓ 3←
MY_POLICY = {
    (4, 0): 1,
    # TODO: add the rest of the path
}


def my_policy(state):
    return MY_POLICY.get(state, random.randrange(4))  # random if you forgot a cell


check(sum(run_episode(GridWorld(), my_policy)) == 3, "your policy reaches 💎 by the shortest path (return = +3)")


# %% Experiment: compare policies (averaged over many episodes!)
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

# %% [markdown]
# ## 🤔 Think about it (write your answers as comments)
# 1. Why is the std of "mine" exactly 0 on the normal floor but not on the slippery one?
# 2. On the slippery floor your policy only covers cells on the path. What happens when it
#    slips off the path? Add arrows for EVERY cell. Does the mean return go up?
# 3. Bonus: on the slippery floor, is it better to go along the bottom row then up the right
#    column, or up the left column then along the top (past the lava)? Measure it, don't guess.
#    You've just discovered why Phase 2 (computing the best policy automatically) is useful.

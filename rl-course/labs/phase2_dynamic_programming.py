# %% [markdown]
# # Lab 2: Solve the gridworld with dynamic programming 🗺️
#
# Same world as Lab 0, but now you have the rulebook (the model P). You'll build it,
# then compute exact values and the optimal policy, and finally answer Lab 0's bonus
# question: what's the best way across the slippery floor?
#
# Run cell by cell ("Run Cell" above each `# %%`) or `python3 phase2_dynamic_programming.py`.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 2.2 (P format), 2.4 (Bellman), 2.5 (policy iteration), 2.6 (value iteration)

# %%
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# The same world as Lab 0 (given)
SIZE = 5
START, GEM, LAVA = (4, 0), (0, 4), (1, 3)
WALLS = {(1, 1), (2, 1), (3, 3)}
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
ARROWS = "↑→↓←"
STATES = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS]
TERMINAL = {GEM, LAVA}


def move(s, a):
    """Where you'd land moving in direction a (walls and edges block you)."""
    r, c = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    if 0 <= r < SIZE and 0 <= c < SIZE and (r, c) not in WALLS:
        return (r, c)
    return s


# %% TODO 1: the model P[s][a] = [(prob, next_state, reward, done), ...]
# This is the same format as Gymnasium's FrozenLake `env.P`.
def build_model(slip=0.0, step_reward=-1.0):
    """For every state s and action a, list the possible outcomes.

    Slip rule (same as Lab 0): with probability `slip` the action is replaced by a
    uniformly random one, so the intended direction has probability (1 - slip) + slip/4
    and each other direction slip/4.
    Rewards: landing on GEM → +10 (done), LAVA → -10 (done), anything else → step_reward.
    Merge outcomes that land in the same cell (e.g. bumping into two different walls).
    Hint: build a dict {next_state: total_prob} for each (s, a), then convert to the list.
    """
    raise NotImplementedError("TODO 1: build_model")


P0 = build_model(slip=0.0)
P2 = build_model(slip=0.2)
check(all(abs(sum(p for p, *_ in P2[s][a]) - 1) < 1e-9 for s in STATES for a in ACTIONS), "probabilities sum to 1")
check(P0[(4, 0)][3] == [(1.0, (4, 0), -1.0, False)], "bumping the edge: stay put, -1")
check(P0[(0, 3)][1] == [(1.0, GEM, 10.0, True)], "stepping onto the gem: +10, done")
check(len(P2[(2, 2)][0]) == 4, "slippery floor: 4 possible outcomes from an open cell")


# %% TODO 2: policy evaluation (Lesson 2.5)
def policy_evaluation(P, policy, gamma=1.0, theta=1e-8, max_sweeps=10_000):
    """policy[s] = list of 4 action probabilities. Return (V, number_of_sweeps).

    Start with V[s] = 0. Each sweep, compute a NEW dict where for non-terminal s:
        V_new[s] = Σ_a policy[s][a] · Σ_(p,s2,r,done) p · (r + gamma · (0 if done else V[s2]))
    Terminal states stay 0. Stop when the biggest change in a sweep < theta.
    """
    raise NotImplementedError("TODO 2: policy_evaluation")


def deterministic(actions):
    """Turn {state: action} into probability lists (default action 0 = up)."""
    return {s: [1.0 if a == actions.get(s, 0) else 0.0 for a in ACTIONS] for s in STATES}


RANDOM = {s: [0.25] * 4 for s in STATES}
SHORTEST = {(4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 0, (3, 4): 0, (2, 4): 0, (1, 4): 0}
V_short, _ = policy_evaluation(P0, deterministic(SHORTEST))
check(abs(V_short[START] - 3) < 1e-6, "shortest-path policy on a normal floor: V(start) = +3")
V_rand, n = policy_evaluation(P0, RANDOM, gamma=0.9)
check(abs(V_rand[START] - (-9.86)) < 0.01, "random policy, γ=0.9: V(start) ≈ -9.86 (matches Lesson 2.3)")


# %% TODO 3: one-step lookahead, Q from V (Lesson 2.4)
def q_from_v(P, V, s, gamma=1.0):
    """Return [Q(s,0), Q(s,1), Q(s,2), Q(s,3)] where Q(s,a) = Σ p · (r + gamma · V[s2] (0 if done))."""
    raise NotImplementedError("TODO 3: q_from_v")


check(q_from_v(P0, V_short, (0, 3))[1] == 10.0, "Q((0,3), →) = 10")


# %% TODO 4: policy iteration (Lesson 2.5)
def greedy_policy(P, V, gamma=1.0):
    """For every state, put probability 1 on argmax_a q_from_v(...)."""
    raise NotImplementedError("TODO 4a: greedy_policy")


def policy_iteration(P, gamma=0.9):
    """Start from deterministic({}) (always up). Repeat: evaluate → greedy improve,
    until the policy stops changing. Return (policy, V, number_of_iterations).

    Stopping: compare `new == policy` (both are dicts of lists of 0.0/1.0, so == works).
    It can't cycle: each round is strictly better somewhere until nothing changes, and
    np.argmax always breaks ties the same way (first best action), so equal actions don't flip-flop.
    Keep gamma < 1 here: with gamma = 1, "always up" never reaches the gem from most cells,
    its true values are -infinity, and policy_evaluation only stops at max_sweeps (V ≈ -10,000)."""
    raise NotImplementedError("TODO 4b: policy_iteration")


pi_chk, V_chk, n_chk = policy_iteration(P0, gamma=0.9)
check(abs(V_chk[START] - (-0.43)) < 0.01, "policy iteration, normal floor, γ=0.9: V(start) ≈ -0.43 (8 steps to the gem)")


# %% TODO 5: value iteration (Lesson 2.6)
def value_iteration(P, gamma=1.0, theta=1e-8):
    """Repeat V_new[s] = max(q_from_v(P, V, s)) (0 for terminals) until the max change < theta.
    Return (greedy_policy(P, V), V, number_of_sweeps)."""
    raise NotImplementedError("TODO 5: value_iteration")


def show(policy, V):
    for r in range(SIZE):
        row = ""
        for c in range(SIZE):
            s = (r, c)
            if s in WALLS: row += "  ██   "
            elif s == GEM: row += "  💎   "
            elif s == LAVA: row += "  🔥   "
            else: row += f"{ARROWS[int(np.argmax(policy[s]))]}{V[s]:+6.2f}"
        print(row)


for name, P in [("normal floor", P0), ("slippery 0.2", P2)]:
    pi_pi, V_pi, iters = policy_iteration(P, gamma=0.9)
    pi_vi, V_vi, sweeps = value_iteration(P, gamma=0.9)
    print(f"\n== {name} (γ=0.9): policy iteration {iters} iterations, value iteration {sweeps} sweeps")
    show(pi_vi, V_vi)
    check(max(abs(V_pi[s] - V_vi[s]) for s in STATES) < 1e-5, f"{name}: policy iteration and value iteration agree")
    res = max(abs(V_vi[s] - (0 if s in TERMINAL else max(q_from_v(P, V_vi, s, 0.9)))) for s in STATES)
    check(res < 1e-6, f"{name}: V satisfies the Bellman optimality equation")


# %% Experiment: answer Lab 0's bonus question on the slippery floor (given; just run it)
def simulate(choose, slip, n=20000):
    """Average return (γ=1, 50-step cap, like Lab 0). choose(s) -> action."""
    total = 0.0
    for _ in range(n):
        s, G = START, 0.0
        for t in range(50):
            a = choose(s)
            if random.random() < slip:
                a = random.randrange(4)
            s = move(s, a)
            G += 10.0 if s == GEM else -10.0 if s == LAVA else -1.0
            if s in TERMINAL:
                break
        total += G
    return total / n


def hand(route):  # Lab 0 style: follow the route, random if you're off it
    return lambda s: route.get(s, random.randrange(4))


LEFT_TOP = {(4, 0): 0, (3, 0): 0, (2, 0): 0, (1, 0): 0, (0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1}
pi_opt, V_opt, _ = value_iteration(P2, gamma=1.0)
print("\nSlippery floor (0.2), γ=1, average return over 20k simulated episodes:")
for name, choose in [("bottom-then-up (Lab 0)", hand(SHORTEST)),
                     ("up-then-along-top", hand(LEFT_TOP)),
                     ("optimal (value iteration)", lambda s: int(np.argmax(pi_opt[s])))]:
    print(f"  {name:<28} {simulate(choose, 0.2):+.2f}")
print(f"  V*(start) predicted by DP     {V_opt[START]:+.2f}")
print("\nOptimal policy on the slippery floor (γ=1):")
show(pi_opt, V_opt)

# %% [markdown]
# ## 🤔 Think about it
# 1. Both hand-drawn routes score about the same. The optimal policy beats them by ~2 points.
#    Where does that gain come from? (Hint: look at the arrows in cells that are NOT on any route.)
# 2. Does the DP prediction V*(start) match the simulation? Why might they differ slightly?
#    (Hint: noise, and the 50-step cap.)

# %% Research: how does slipperiness change the value of the start?
# HYPOTHESIS (write it first!): as slip goes 0 → 0.6, I expect V*(start) to ...
slips = np.linspace(0, 0.6, 13)
vals = [value_iteration(build_model(slip=x), gamma=1.0)[1][START] for x in slips]
plt.figure(figsize=(6, 3.5))
plt.plot(slips, vals, "o-")
plt.xlabel("slip probability"); plt.ylabel("V*(start)"); plt.title("A slipperier world is worth less")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
# FOLLOW-UP: at what slip does the optimal route switch? Method: for slip in 0, 0.1, ..., 0.5 run
#   value_iteration(build_model(slip=x), gamma=1.0), turn each policy into arrows
#   {s: ARROWS[int(np.argmax(policy[s]))]}, and print the cells whose arrow differs from slip 0.
#   Before believing a "switch", print q_from_v at that cell on the dry floor: ties are common there,
#   and np.argmax just shows the first of several equally good arrows.
# FOLLOW-UP 2: count value-iteration sweeps for γ = 0.5, 0.9, 0.99. How does it scale? Why?

# %% [markdown]
# # Lab 2 — SOLUTION
# Try the exercise file first!

# %%
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


# The same world as Lab 0
SIZE = 5
START, GEM, LAVA = (4, 0), (0, 4), (1, 3)
WALLS = {(1, 1), (2, 1), (3, 3)}
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
ARROWS = "↑→↓←"
STATES = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS]
TERMINAL = {GEM, LAVA}


def move(s, a):
    r, c = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
    if 0 <= r < SIZE and 0 <= c < SIZE and (r, c) not in WALLS:
        return (r, c)
    return s


# %% TODO 1 — the model P[s][a] = [(prob, next_state, reward, done), ...]
def build_model(slip=0.0, step_reward=-1.0):
    P = {}
    for s in STATES:
        P[s] = {}
        for a in ACTIONS:
            outcomes = {}
            for actual in ACTIONS:
                p = slip / 4 + (1 - slip if actual == a else 0.0)
                if p == 0:
                    continue
                s2 = move(s, actual)
                outcomes[s2] = outcomes.get(s2, 0.0) + p
            P[s][a] = []
            for s2, p in outcomes.items():
                if s2 == GEM:
                    P[s][a].append((p, s2, 10.0, True))
                elif s2 == LAVA:
                    P[s][a].append((p, s2, -10.0, True))
                else:
                    P[s][a].append((p, s2, step_reward, False))
    return P


P0 = build_model(slip=0.0)
P2 = build_model(slip=0.2)
check(all(abs(sum(p for p, *_ in P2[s][a]) - 1) < 1e-9 for s in STATES for a in ACTIONS), "probabilities sum to 1")
check(P0[(4, 0)][3] == [(1.0, (4, 0), -1.0, False)], "bumping the edge: stay put, -1")
check(P0[(0, 3)][1] == [(1.0, GEM, 10.0, True)], "stepping onto the gem: +10, done")
check(len(P2[(2, 2)][0]) == 4, "slippery floor: 4 possible outcomes from an open cell")


# %% TODO 2 — policy evaluation
def policy_evaluation(P, policy, gamma=1.0, theta=1e-8, max_sweeps=10_000):
    """policy[s] = list of 4 action probabilities. Returns (V, sweeps)."""
    V = {s: 0.0 for s in STATES}
    for sweep in range(1, max_sweeps + 1):
        delta = 0.0
        newV = {}
        for s in STATES:
            if s in TERMINAL:
                newV[s] = 0.0
                continue
            v = 0.0
            for a, pi_a in enumerate(policy[s]):
                for p, s2, r, done in P[s][a]:
                    v += pi_a * p * (r + gamma * (0.0 if done else V[s2]))
            newV[s] = v
            delta = max(delta, abs(v - V[s]))
        V = newV
        if delta < theta:
            return V, sweep
    return V, max_sweeps


def deterministic(actions):
    return {s: [1.0 if a == actions.get(s, 0) else 0.0 for a in ACTIONS] for s in STATES}


RANDOM = {s: [0.25] * 4 for s in STATES}
SHORTEST = {(4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 0, (3, 4): 0, (2, 4): 0, (1, 4): 0}
V_short, _ = policy_evaluation(P0, deterministic(SHORTEST))
check(abs(V_short[START] - 3) < 1e-6, "shortest-path policy on a normal floor: V(start) = +3")
V_rand, n = policy_evaluation(P0, RANDOM, gamma=0.9)
check(abs(V_rand[START] - (-9.86)) < 0.01, "random policy, γ=0.9: V(start) ≈ -9.86 (matches Lesson 2.3)")


# %% TODO 3 — one-step lookahead: Q from V
def q_from_v(P, V, s, gamma=1.0):
    return [sum(p * (r + gamma * (0.0 if done else V[s2])) for p, s2, r, done in P[s][a]) for a in ACTIONS]


check(q_from_v(P0, V_short, (0, 3))[1] == 10.0, "Q((0,3), →) = 10")


# %% TODO 4 — policy iteration
def greedy_policy(P, V, gamma=1.0):
    pol = {}
    for s in STATES:
        q = q_from_v(P, V, s, gamma)
        best = int(np.argmax(q))
        pol[s] = [1.0 if a == best else 0.0 for a in ACTIONS]
    return pol


def policy_iteration(P, gamma=0.9):
    """Keep gamma < 1: with gamma = 1, "always up" never reaches the gem from most cells,
    so its values are -infinity and policy_evaluation only stops at max_sweeps."""
    policy = deterministic({})  # start: always "up"
    for it in range(1, 100):
        V, _ = policy_evaluation(P, policy, gamma)
        new = greedy_policy(P, V, gamma)
        if new == policy:  # np.argmax breaks ties the same way every time, so no flip-flopping
            return policy, V, it
        policy = new
    return policy, V, it


pi_chk, V_chk, n_chk = policy_iteration(P0, gamma=0.9)
check(abs(V_chk[START] - (-0.43)) < 0.01, "policy iteration, normal floor, γ=0.9: V(start) ≈ -0.43 (8 steps to the gem)")


# %% TODO 5 — value iteration
def value_iteration(P, gamma=1.0, theta=1e-8):
    V = {s: 0.0 for s in STATES}
    for sweep in range(1, 10_000):
        delta = 0.0
        newV = {}
        for s in STATES:
            newV[s] = 0.0 if s in TERMINAL else max(q_from_v(P, V, s, gamma))
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if delta < theta:
            break
    return greedy_policy(P, V, gamma), V, sweep


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


# %% Experiment — answer Lab 0's bonus question on the slippery floor
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


# %% Research — how does slipperiness change the value of the start?
slips = np.linspace(0, 0.6, 13)
vals = [value_iteration(build_model(slip=x), gamma=1.0)[1][START] for x in slips]
plt.figure(figsize=(6, 3.5))
plt.plot(slips, vals, "o-")
plt.xlabel("slip probability"); plt.ylabel("V*(start)"); plt.title("A slipperier world is worth less")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# Follow-up: when does the route switch? Diff the greedy arrows against the dry floor.
def arrows(policy):
    return {s: ARROWS[int(np.argmax(policy[s]))] for s in STATES if s not in TERMINAL}


pi_dry, V_dry, _ = value_iteration(P0, gamma=1.0)
base = arrows(pi_dry)
for x in [0.1, 0.2, 0.3, 0.4, 0.5]:
    now = arrows(value_iteration(build_model(slip=x), gamma=1.0)[0])
    print(f"slip {x}: cells that changed", {s: base[s] + " to " + now[s] for s in now if now[s] != base[s]})
print("Dry floor, Q at the start:", [round(q, 2) for q in q_from_v(P0, V_dry, START)])
# ↑ and → tie at the start on the dry floor (argmax shows ↑), so ANY slip breaks the tie toward the bottom route.

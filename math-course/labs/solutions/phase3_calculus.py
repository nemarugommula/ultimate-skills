# %% [markdown]
# # Lab 3: Calculus from scratch 📈
#
# (SOLUTION file: every TODO is filled in. Try labs/phase3_calculus.py first!)
#
# You'll build the calculus that trains every neural network, with nothing but NumPy:
# a slope-by-nudging tool and a gradient check, the chain rule on one neuron, the gradient
# of a 2-input loss, gradient descent in 1D and 2D (and the learning rate that makes it
# explode), a tiny autograd engine (a mini PyTorch `loss.backward()`) that fits a line,
# and the area under a bell curve.
#
# Setup (once, from the math-course folder): pip install -r labs/requirements.txt
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase3_calculus.py`.
# Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 3.1 (derivatives), 3.2 (chain rule), 3.3 (gradients),
# 3.4 and 3.4b (gradient descent, the 2/c speed limit), 3.5 (backprop), 3.6 (integrals)

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)  # one seeded random generator → reproducible results


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def close(a, b, tol=1e-6):
    """True when every number in a is within tol of the matching number in b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) < tol))


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: the slope by nudging (Lesson 3.1)
def derivative(f, x, h=1e-5):
    """The slope of f at x, measured by nudging x a tiny bit both ways (the 'centered difference')."""
    # TODO: return (f(x + h) − f(x − h)) / (2h)
    # Hint: rise = f(x + h) − f(x − h). Run = the distance between the two nudges = 2 * h.
    rise = f(x + h) - f(x - h)
    run = 2 * h
    return rise / run


check(close(derivative(lambda x: x ** 2, 3.0), 6.0), "slope of x² at x = 3 is 6")
check(close(derivative(lambda w: (2 * w - 3) ** 2, 2.0), 4.0), "slope of the loss (2w − 3)² at w = 2 is 4")
check(close(derivative(np.exp, 0.0), 1.0), "slope of eˣ at x = 0 is 1 (eˣ is its own slope)")


# %% See it: why nudge both ways, and the gradient check engineers use
# f(x) = x³ at x = 2. The exact slope (power rule, Lesson 3.2) is 3 × 2² = 12.
cube = lambda x: x ** 3
print("     h   forward (one way)   centered (both ways)   exact = 12")
for h in [0.1, 0.01, 0.001]:
    forward = (cube(2 + h) - cube(2)) / h
    print(f"{h:6}   {forward:14.6f}   {derivative(cube, 2.0, h):16.6f}")
# Make h 10× smaller: the forward error shrinks 10×, the centered error 100×.

# Gradient checking (Lesson 3.1): compare a hand-derived formula with the nudge.
L = lambda w: (2 * w - 3) ** 2
good_formula = lambda w: 4 * (2 * w - 3)   # chain rule: 2(2w − 3) × 2
buggy_formula = lambda w: 2 * (2 * w - 3)  # someone forgot the inner × 2
nudge = derivative(L, 2.0)
print(f"\nnudge says {nudge:.6f}; good formula says {good_formula(2.0)}; buggy formula says {buggy_formula(2.0)}")
check(abs(good_formula(2.0) - nudge) < 1e-6, "gradient check passes for the right formula")
check(abs(buggy_formula(2.0) - nudge) > 1, "gradient check catches the buggy formula")
# An absurdly tiny h is WORSE: the computer runs out of decimal places (Lesson 8.5).
print("with h = 1e-13 the nudge says", derivative(L, 2.0, 1e-13), "(should be 4). Engineers use h ≈ 1e-5.")


# %% TODO 2: the chain rule on one neuron (Lesson 3.2)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def neuron_slope(w, x):
    """The slope of y = σ(w · x) with respect to the weight w, using the chain rule."""
    # TODO: follow the chain one link at a time, then multiply the slopes (like gear ratios):
    #   u = w * x               inner machine,  du/dw = x
    #   y = sigmoid(u)          outer machine,  dy/du = y * (1 − y)
    #   return dy/du * du/dw
    u = w * x
    du_dw = x
    y = sigmoid(u)
    dy_du = y * (1 - y)
    return dy_du * du_dw


check(abs(neuron_slope(2.0, 1.0) - 0.1049936) < 1e-6, "σ(w·x) at w = 2, x = 1 has slope 0.105 (Lesson 3.2's worked example)")
check(close(neuron_slope(2.0, 1.0), derivative(lambda w: sigmoid(w * 1.0), 2.0)), "the chain rule agrees with the nudge")
check(neuron_slope(5.0, 0.0) == 0.0, "input x = 0 → the weight can't matter (slope 0)")


# %% See it: vanishing gradients through 10 sigmoid layers (Lesson 3.2)
# Each layer computes h ← σ(h). The learning signal for the first layer is the product of all
# 10 local slopes σ(h)(1 − σ(h)), and each one is at most 0.25.
def ten_sigmoids(x):
    h = x
    for _ in range(10):
        h = sigmoid(h)
    return h


h = 0.0
signal = 1.0
for layer in range(1, 11):
    y = sigmoid(h)
    signal = signal * y * (1 - y)  # chain rule: multiply by this layer's local slope
    h = y
    print(f"through {layer:2} layers: learning signal = {signal:.2e}")
print(f"best case 0.25¹⁰ = {0.25 ** 10:.2e}. This is why deep sigmoid nets were hard to train before ReLU and ResNets.")
check(abs(signal - derivative(ten_sigmoids, 0.0)) < 1e-3 * signal, "the product of 10 local slopes = the nudged slope of the whole stack")


# %% TODO 3: the gradient of a 2-input function (Lesson 3.3)
def numeric_gradient(f, p, h=1e-5):
    """The gradient of f at point p (a NumPy array): one centered nudge per input."""
    p = np.asarray(p, dtype=float)
    grad = np.zeros_like(p)
    for i in range(len(p)):
        # TODO: nudge ONLY input i, up and down, keeping every other input frozen:
        #   up = p.copy();   up[i] = up[i] + h
        #   down = p.copy(); down[i] = down[i] − h
        #   grad[i] = (f(up) − f(down)) / (2h)
        # Hint: .copy() matters! Without it you'd change p itself.
        up = p.copy()
        up[i] = up[i] + h
        down = p.copy()
        down[i] = down[i] - h
        grad[i] = (f(up) - f(down)) / (2 * h)
    return grad


bowl = lambda p: p[0] ** 2 + 3 * p[1] ** 2  # f(x, y) = x² + 3y²
check(close(numeric_gradient(bowl, [1.0, 1.0]), [2.0, 6.0]), "∇f at (1, 1) = [2, 6]")
check(close(numeric_gradient(bowl, [2.0, -1.0]), [4.0, -6.0]), "∇f at (2, −1) = [4, −6] (Lesson 3.3's worked example)")
check(close(numeric_gradient(lambda p: p[0] * p[1] * p[2], [1.0, 2.0, 3.0]), [6.0, 3.0, 2.0]),
      "works for 3 inputs too: ∇(x·y·z) at (1, 2, 3) = [6, 3, 2]")


# %% See it: a gradient with respect to the INPUT fools a classifier (FGSM, Lesson 3.3)
# A 3-pixel detector scores an image with z = w · x. Which way should each pixel move to lower z most?
w_pixels = np.array([2.0, -1.0, 0.5])
image = np.array([0.5, 0.5, 0.5])
score = lambda img: w_pixels @ img
g = numeric_gradient(score, image)  # the gradient of the score with respect to the pixels = the weights
attacked = image - 0.1 * np.sign(g)  # FGSM: move every pixel by ε = 0.1 against its gradient's sign
print("gradient wrt pixels:", g.round(6), " pixel changes:", (attacked - image).round(6))
print(f"score {score(image):.2f} → {score(attacked):.2f}")
check(close(score(attacked) - score(image), -0.35), "every pixel helps: z drops by 0.1 × (2 + 1 + 0.5) = 0.35")


# %% TODO 4: gradient descent in 1D (Lesson 3.4)
def gradient_descent_1d(slope, w0, eta, steps):
    """Repeat w ← w − η · slope(w). Return every w visited: [w0, w1, ..., w_steps]."""
    w = w0
    history = [w0]
    for _ in range(steps):
        # TODO: take one step downhill, then remember where you are:
        #   w = w − eta * slope(w)
        #   history.append(w)
        w = w - eta * slope(w)
        history.append(w)
    return history


slope = lambda w: 2 * (w - 3)  # the slope of the bowl L(w) = (w − 3)²
check(close(gradient_descent_1d(slope, 0.0, 0.1, 3), [0, 0.6, 1.08, 1.464]), "η = 0.1: 0 → 0.6 → 1.08 → 1.464 (Lesson 3.4's table)")
check(close(gradient_descent_1d(slope, 0.0, 0.5, 1), [0, 3]), "η = 0.5: one jump straight to the bottom")
check(close(gradient_descent_1d(slope, 0.0, 1.0, 4), [0, 6, 0, 6, 0]), "η = 1.0: bounces between 0 and 6 forever")
check(abs(gradient_descent_1d(slope, 0.0, 1.1, 30)[-1]) > 100, "η = 1.1: every jump lands farther away (divergence)")


# %% See it: curvature sets the speed limit, η < 2/c (Lesson 3.4b)
# On the bowl L(w) = ½ · c · w² the slope is c · w, so every step multiplies w by (1 − η c).
# |1 − η c| < 1 exactly when η < 2/c. Here c = 4, so the limit is 2/4 = 0.5.
c = 4.0
plt.figure(figsize=(7, 4))
for eta in [0.1, 0.25, 0.4, 0.49, 0.51, 0.6]:
    ws = gradient_descent_1d(lambda w: c * w, 1.0, eta, 30)
    print(f"η = {eta:4}: factor 1 − ηc = {1 - eta * c:5.2f}, |w| after 30 steps = {abs(ws[-1]):.3g}")
    plt.semilogy(np.abs(ws) + 1e-12, label=f"η = {eta}")
plt.axhline(1, color="gray", lw=0.8)
plt.xlabel("step"); plt.ylabel("|w| (distance from the bottom, log scale)")
plt.title("Bowl with curvature c = 4: the speed limit is η = 2/c = 0.5")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); show()
check(abs(gradient_descent_1d(lambda w: c * w, 1.0, 0.49, 30)[-1]) < 1, "η = 0.49 (just under 2/c) still converges")
check(abs(gradient_descent_1d(lambda w: c * w, 1.0, 0.51, 30)[-1]) > 1, "η = 0.51 (just over 2/c) blows up")


# %% TODO 5: gradient descent in 2D (Lesson 3.4)
def gradient_descent(grad, p0, eta, steps):
    """Gradient descent on any number of knobs. Returns all points visited, shape (steps + 1, d)."""
    p = np.array(p0, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        # TODO: the same rule as TODO 4, but p is a whole vector now:
        #   p = p − eta * grad(p)       (NumPy subtracts element by element)
        #   path.append(p.copy())
        p = p - eta * grad(p)
        path.append(p.copy())
    return np.array(path)


bowl_grad = lambda p: np.array([2 * p[0], 6 * p[1]])  # ∇(x² + 3y²) = [2x, 6y]
path = gradient_descent(bowl_grad, [1.0, 1.0], 0.1, 100)
check(close(path[1], [0.8, 0.4]), "one step from (1, 1) with η = 0.1: (1 − 0.1·2, 1 − 0.1·6) = (0.8, 0.4)")
check(close(path[-1], [0.0, 0.0]), "100 steps reach the bottom (0, 0)")
path_nudged = gradient_descent(lambda p: numeric_gradient(bowl, p), [1.0, 1.0], 0.1, 100)
check(close(path_nudged, path), "the nudged gradient from TODO 3 walks the same path")


# %% See it: the narrow valley (Lesson 3.4). Curvature 2 across x, 20 across y → η must stay below 2/20 = 0.1
valley = lambda p: p[0] ** 2 + 10 * p[1] ** 2
valley_grad = lambda p: np.array([2 * p[0], 20 * p[1]])
start = [2.0, 1.0]
runs = {}
for eta in [0.01, 0.09, 0.105]:
    runs[eta] = gradient_descent(valley_grad, start, eta, 60)
    print(f"η = {eta:5}: loss after 60 steps = {valley(runs[eta][-1]):.3g}")

gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-1.5, 1.5, 200))
plt.figure(figsize=(7, 4.5))
plt.contour(gx, gy, gx ** 2 + 10 * gy ** 2, levels=20, cmap="Greys", linewidths=0.6)
names = {0.01: "η = 0.01 (crawls)", 0.09: "η = 0.09 (zig-zags, arrives)", 0.105: "η = 0.105 (flies off)"}
for eta, pts in runs.items():
    shown = pts[np.all(np.abs(pts) < 3, axis=1)]  # only draw the part that's still on the map
    plt.plot(shown[:, 0], shown[:, 1], ".-", label=names[eta])
plt.xlim(-2.5, 2.5); plt.ylim(-1.5, 1.5); plt.xlabel("x"); plt.ylabel("y")
plt.title("Gradient descent on x² + 10y²"); plt.legend(); plt.tight_layout(); show()
check(valley(runs[0.09][-1]) < 1e-6, "η = 0.09 zig-zags across the steep direction but gets there")
check(valley(runs[0.105][-1]) > 1000, "η = 0.105 > 2/20 explodes, even though x alone would be fine")
check(valley(runs[0.01][-1]) > 0.1, "η = 0.01 is safe but slow: still not at the bottom after 60 steps")
# This tension (steep in one direction, flat in another) is what momentum and Adam fix (Lesson 7.3).


# %% TODO 6: a tiny autograd engine, part 1: the local rules (Lesson 3.5)
class Value:
    """A number that remembers how it was made, so it can pass gradients backwards (like micrograd)."""

    def __init__(self, data, children=()):
        self.data = float(data)  # the number itself (computed in the forward pass)
        self.grad = 0.0  # ∂L/∂(this number), filled in by backward()
        self._children = children  # the Values this one was made from
        self._backward = lambda: None  # how to hand my gradient to my children (set by + and ×)

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)  # allow v + 2: wrap the plain number
        out = Value(self.data + other.data, (self, other))

        def _backward():
            # TODO 6a: the add box is "the distributor": both inputs get the gradient that arrived at out.
            #   self.grad += out.grad
            #   other.grad += out.grad
            # Hint: use += (not =). A Value used twice must collect gradient from both uses.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            # TODO 6b: the multiply box is "the swapper": each input gets out.grad × the OTHER input's value.
            #   self.grad += out.grad * other.data
            #   other.grad += out.grad * self.data
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    # Ready-made helpers. They're all built from + and ×, so they need no new backward rule.
    def __neg__(self):  # -v
        return self * -1

    def __sub__(self, other):  # v - 2
        return self + (-other)

    def __radd__(self, other):  # 2 + v
        return self + other

    def __rmul__(self, other):  # 2 * v
        return self * other

    def square(self):  # v²
        return self * self

    def backward(self):
        """Backpropagation: fill in .grad for every Value that led to this one."""
        # Ready-made: list every Value in the graph so that each one comes AFTER its children
        # (a "topological order"). Walking that list backwards then starts at the loss.
        order = []
        visited = set()

        def visit(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    visit(child)
                order.append(v)

        visit(self)
        # TODO 7: run the backward pass.
        #   1. self.grad = 1.0             (∂L/∂L = 1: the loss moves 1-for-1 with itself)
        #   2. for v in reversed(order):   (from the loss back towards the inputs)
        #          v._backward()           (each box hands its gradient to its children)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


a, b = Value(2.0), Value(3.0)
c = a + b
c.grad = 1.0  # pretend c is the loss: ∂c/∂c = 1
c._backward()  # run just this one box backwards
check(c.data == 5.0 and a.grad == 1.0 and b.grad == 1.0, "add box: 2 + 3 = 5, and both inputs get gradient 1")
a, b = Value(2.0), Value(3.0)
d = a * b
d.grad = 1.0
d._backward()
check(d.data == 6.0 and a.grad == 3.0 and b.grad == 2.0, "multiply box: 2 × 3 = 6, and each input gets the OTHER one (3, 2)")


# %% TODO 7: backward() for a whole graph (fill in the end of Value.backward above, then run this cell)
w, x, b = Value(2.0), Value(3.0), Value(1.0)
y = 5.0
L = (w * x + b - y).square()  # Lesson 3.5's graph: L = (w·x + b − y)²
L.backward()
check(L.data == 4.0, "forward pass: L = (2·3 + 1 − 5)² = 4")
check(w.grad == 12.0 and b.grad == 4.0 and x.grad == 8.0, "backward pass: ∂L/∂w = 12, ∂L/∂b = 4, ∂L/∂x = 8 (Lesson 3.5)")
v = Value(3.0)
v.square().backward()
check(v.grad == 6.0, "a Value used twice collects both gradients: d(v²)/dv = 2v = 6")


# %% See it: train a line with your autograd (Lesson 3.5's mini training loop)
# Three points on y = 2x + 1. Start from w = b = 0 and let backprop + gradient descent find the line.
xs = [1.0, 2.0, 3.0]
ys = [3.0, 5.0, 7.0]
w = Value(0.0)
b = Value(0.0)
eta = 0.05
for step in range(1, 1001):
    loss = Value(0.0)
    for x_i, y_i in zip(xs, ys):
        loss = loss + (w * x_i + b - y_i).square()  # forward pass, one point at a time
    loss = loss * (1 / len(xs))  # mean squared error
    w.grad = 0.0  # forget last step's gradients (PyTorch: optimizer.zero_grad())
    b.grad = 0.0
    loss.backward()  # backward pass: fills w.grad and b.grad
    w.data = w.data - eta * w.grad  # gradient descent step (Lesson 3.4)
    b.data = b.data - eta * b.grad
    if step in (1, 10, 100, 1000):
        print(f"step {step:4}: loss = {loss.data:8.5f}   w = {w.data:.4f}   b = {b.data:.4f}")
check(abs(w.data - 2) < 0.01 and abs(b.data - 1) < 0.02, "backprop + gradient descent found the line y = 2x + 1")
# This is what `loss.backward(); optimizer.step()` does in PyTorch, for billions of weights at once.


# %% TODO 8: area by adding up thin rectangles (Lesson 3.6)
def area_under(f, a, b, n):
    """Area under f from a to b: n thin strips, each as tall as f at the MIDDLE of its strip."""
    width = (b - a) / n
    total = 0.0
    for i in range(n):
        # TODO: the middle of strip i is at a + (i + 0.5) * width.
        #   total = total + f(middle) * width        (height × width = one rectangle)
        middle = a + (i + 0.5) * width
        total = total + f(middle) * width
    return total


check(close(area_under(lambda x: x ** 2, 0, 1, 4), 0.328125),
      "4 strips under x² on [0, 1]: 0.25 × (0.125² + 0.375² + 0.625² + 0.875²) = 0.328125")
check(close(area_under(lambda x: x ** 2, 0, 1, 1000), 1 / 3), "1,000 strips → 1/3, the exact area")
bell = lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)  # the standard bell curve N(0, 1)
check(close(area_under(bell, -1, 1, 1000), 0.682689, 1e-5), "68.3% of a bell curve lies within 1 standard deviation")
check(close(area_under(bell, -2, 2, 1000), 0.954500, 1e-5), "95.4% lies within 2 standard deviations")
check(close(area_under(bell, -10, 10, 1000), 1.0), "the whole bell has area 1: it's a probability density (Lesson 4.3)")


# %% See it: area = probability
print("strips n   area under the bell on [−1, 1]   error")
for n in [2, 10, 100]:
    est = area_under(bell, -1, 1, n)
    print(f"{n:8}   {est:.6f}                        {abs(est - 0.682689):.1e}")
# 10× more strips → about 100× smaller error (the middle-of-the-strip rule is that good).

xx = np.linspace(-4, 4, 400)
plt.figure(figsize=(7, 3.8))
plt.plot(xx, bell(xx), color="purple", label="bell curve N(0, 1)")
n = 10
width = 2 / n
middles = -1 + (np.arange(n) + 0.5) * width
plt.bar(middles, bell(middles), width=width, alpha=0.35, edgecolor="purple", label=f"{n} rectangles on [−1, 1]")
plt.title(f"P(−1 < X < 1) = area ≈ {area_under(bell, -1, 1, n):.4f}")
plt.xlabel("x"); plt.ylabel("density"); plt.legend(); plt.tight_layout(); show()
# The same "add up thin strips" gives a classifier's AUC (Lesson 3.6) and every probability of a range.


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. The bowl L(w) = (w − 3)² from TODO 4 has curvature c = 2. Predict the η where it starts
#    to bounce (2/c), then check with `gradient_descent_1d`. What about 10(w − 3)²?
# 2. Add a `relu()` method to `Value` (forward: max(0, x); backward: the "gate", pass the
#    gradient only if the input was positive). Check it against `derivative`.
# 3. Run TODO 5's `gradient_descent` on the two-valley loss f(w) = w⁴ − 3w² + w from a few
#    random starts (`rng.uniform(-2, 2)`). Which valley does each one end in, and why?

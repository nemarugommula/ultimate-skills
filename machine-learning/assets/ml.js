/* ============ ML: a tiny offline machine-learning kit for playgrounds (no dependencies) ============
 * Everything is small, seeded and fast enough to refit on every slider move (n ≤ ~400 points, d ≤ ~10).
 * The API copies scikit-learn on purpose: new Model(opts).fit(X, y) → .predict(X) / .proba(X) / .score1(x).
 *
 * Random + data
 *   ML.rng(seed) → () => [0,1)        ML.randn(rng)          ML.shuffle(arr, rng)       ML.range(n)
 *   ML.blobs({n, centers:[[x,y],...], std, seed})            → {X, y}   (y = index of the center)
 *   ML.moons({n, noise, seed}) · ML.circles({n, noise, factor, seed}) · ML.xor({n, noise, seed}) · ML.spiral({n, noise, turns, seed})
 *   ML.linear1d({n, f: x => ..., noise, xMin, xMax, seed})    → {X: [[x]], y}   (regression)
 *   ML.split(X, y, {test: 0.3, seed, stratify})               → {Xtr, ytr, Xte, yte, itr, ite}
 *   ML.kfold(n, k, {seed, shuffle, y (stratify)})             → [{train: [i...], test: [i...]}, ...]
 *   ML.bootstrap(n, rng) → {idx, oob}
 *   ML.standardize(X) → {Z, mean, std, transform(X)}          ML.poly1d(X, degree) → [[x, x², ...]]
 *
 * Models (classification labels are integers 0..K-1; score1(x) = P(class 1), handy for binary fields)
 *   new ML.KNN({k, weights: "uniform"|"distance", task: "clf"|"reg"})
 *   new ML.Logistic({lr, epochs, l2, features: x => [...]})            binary, gradient descent
 *   new ML.Linear({l2, l1, features, iters})                           least squares / ridge (closed form) / lasso+elastic net (coordinate descent)
 *   new ML.Tree({maxDepth, minLeaf, criterion: "gini"|"entropy"|"mse", maxFeatures, task, seed})
 *        .root nodes {f, t, left, right, value, n, imp, depth, leaf} · .nLeaves · .depthReached · .apply(x) → leaf
 *   new ML.Forest({nTrees, maxDepth, minLeaf, maxFeatures ("sqrt"|number|null), bootstrap, task, seed})   .oobScore · .trees
 *   new ML.GBM({nRounds, lr, maxDepth, minLeaf, task: "reg"|"clf"})  .stage(k) → model with the first k rounds · .history
 *   new ML.AdaBoost({nRounds})                                         stumps on reweighted data · .alphas · .weightsHistory
 *   new ML.GaussianNB() · new ML.MultinomialNB({alpha})                NB on token lists: fit(docs: [[tok...]], y)
 *   new ML.SVM({C, kernel: "linear"|"rbf"|"poly", gamma, degree, coef0, iters, seed})  simplified SMO
 *        .decision(x) · .supportIdx (indices with α > 0) · .alpha · .b
 *   new ML.KMeans({k, init: "random"|"++", seed}) .init(X) .step() (one assign+update; returns moved) .fit(X) .centers .labels .inertia
 *   ML.pca(X, k) → {components, explained, ratio, mean, transform(X)}   ML.eigSym(A) → {values, vectors} (Jacobi)
 *
 * Metrics
 *   ML.accuracy(y, yhat) · ML.mse · ML.rmse · ML.mae · ML.r2 · ML.logloss(y, p) · ML.brier(y, p)
 *   ML.confusion(y, scores, thr) → {tp, fp, tn, fn, precision, recall, fpr, tpr, f1, accuracy}
 *   ML.roc(y, scores) → [[fpr, tpr], ...] · ML.prCurve(y, scores) → [[recall, precision], ...] · ML.auc(points)
 *   ML.tokenize(text) → lowercase word tokens
 * All functions are pure and seeded: the same seed gives the same data and the same model.
 */
(function () {
  const ML = {};

  /* ---------- random ---------- */
  ML.rng = seed => { // mulberry32
    let a = (seed == null ? 1 : seed) >>> 0;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  ML.randn = rng => { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  ML.shuffle = (arr, rng) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  ML.range = n => Array.from({ length: n }, (_, i) => i);
  const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, z))));
  ML.sigmoid = sigmoid;
  const argmax = a => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  ML.argmax = argmax;
  const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);

  /* ---------- toy datasets ---------- */
  ML.blobs = ({ n = 100, centers = [[-1.5, -1], [1.5, 1]], std = 0.8, seed = 1 } = {}) => {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const k = i % centers.length, s = Array.isArray(std) ? std[k] : std; X.push([centers[k][0] + s * ML.randn(r), centers[k][1] + s * ML.randn(r)]); y.push(k); }
    return { X, y };
  };
  ML.moons = ({ n = 100, noise = 0.2, seed = 1 } = {}) => {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) {
      const k = i % 2, t = Math.PI * r();
      const x = k ? 1 - Math.cos(t) : Math.cos(t), z = k ? 0.5 - Math.sin(t) : Math.sin(t);
      X.push([(x - 0.5) * 2 + noise * ML.randn(r), (z - 0.25) * 2 + noise * ML.randn(r)]); y.push(k);
    }
    return { X, y };
  };
  ML.circles = ({ n = 100, noise = 0.1, factor = 0.45, seed = 1 } = {}) => {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const k = i % 2, t = 2 * Math.PI * r(), R = (k ? factor : 1) * 2.4; X.push([R * Math.cos(t) + noise * 2.4 * ML.randn(r), R * Math.sin(t) + noise * 2.4 * ML.randn(r)]); y.push(k); }
    return { X, y };
  };
  ML.xor = ({ n = 100, noise = 0.35, seed = 1 } = {}) => {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const a = r() < 0.5 ? -1 : 1, b = r() < 0.5 ? -1 : 1; X.push([1.5 * a + noise * 2 * ML.randn(r), 1.5 * b + noise * 2 * ML.randn(r)]); y.push(a * b > 0 ? 0 : 1); }
    return { X, y };
  };
  ML.spiral = ({ n = 120, noise = 0.15, turns = 1.5, seed = 1 } = {}) => {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const k = i % 2, t = 0.3 + r() * turns * 2 * Math.PI, R = t / (turns * 2 * Math.PI) * 4, s = k ? Math.PI : 0; X.push([R * Math.cos(t + s) + noise * ML.randn(r), R * Math.sin(t + s) + noise * ML.randn(r)]); y.push(k); }
    return { X, y };
  };
  ML.linear1d = ({ n = 30, f = x => 0.5 * x + 1, noise = 0.5, xMin = -4, xMax = 4, seed = 1 } = {}) => {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const x = xMin + (xMax - xMin) * r(); X.push([x]); y.push(f(x) + noise * ML.randn(r)); }
    return { X, y };
  };

  /* ---------- splitting ---------- */
  ML.split = (X, y, { test = 0.3, seed = 1, stratify = false } = {}) => {
    const r = ML.rng(seed); let ite = [], itr = [];
    if (stratify) {
      const by = {}; y.forEach((v, i) => (by[v] = by[v] || []).push(i));
      Object.values(by).forEach(ix => { const s = ML.shuffle(ix, r), m = Math.round(s.length * test); ite.push(...s.slice(0, m)); itr.push(...s.slice(m)); });
    } else { const s = ML.shuffle(ML.range(X.length), r), m = Math.round(X.length * test); ite = s.slice(0, m); itr = s.slice(m); }
    return { Xtr: itr.map(i => X[i]), ytr: itr.map(i => y[i]), Xte: ite.map(i => X[i]), yte: ite.map(i => y[i]), itr, ite };
  };
  ML.kfold = (n, k, { seed = 1, shuffle = true, y = null } = {}) => {
    const r = ML.rng(seed), fold = new Array(n);
    if (y) { const by = {}; y.forEach((v, i) => (by[v] = by[v] || []).push(i)); let c = 0; Object.values(by).forEach(ix => (shuffle ? ML.shuffle(ix, r) : ix).forEach(i => (fold[i] = c++ % k))); }
    else (shuffle ? ML.shuffle(ML.range(n), r) : ML.range(n)).forEach((i, j) => (fold[i] = shuffle ? j % k : Math.min(k - 1, Math.floor(j * k / n))));
    return ML.range(k).map(f => ({ train: ML.range(n).filter(i => fold[i] !== f), test: ML.range(n).filter(i => fold[i] === f) }));
  };
  ML.bootstrap = (n, rng) => { const idx = [], seen = new Uint8Array(n); for (let i = 0; i < n; i++) { const j = Math.floor(rng() * n); idx.push(j); seen[j] = 1; } return { idx, oob: ML.range(n).filter(i => !seen[i]) }; };
  ML.standardize = X => {
    const d = X[0].length, mu = ML.range(d).map(j => mean(X.map(r => r[j])));
    const sd = ML.range(d).map(j => Math.sqrt(mean(X.map(r => (r[j] - mu[j]) ** 2))) || 1);
    const transform = A => A.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
    return { Z: transform(X), mean: mu, std: sd, transform };
  };
  ML.poly1d = (X, deg) => X.map(r => ML.range(deg).map(p => r[0] ** (p + 1)));

  /* ---------- small linear algebra ---------- */
  function solve(A, b) { // Gaussian elimination with partial pivoting; A is copied
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) M[c][c] = 1e-12;
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
  ML.solve = solve;
  ML.eigSym = A0 => { // Jacobi rotations; returns eigenvalues (descending) and unit eigenvectors (as rows)
    const n = A0.length, A = A0.map(r => r.slice()), V = ML.range(n).map(i => ML.range(n).map(j => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
      if (off < 1e-18) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    const order = ML.range(n).sort((a, b) => A[b][b] - A[a][a]);
    return { values: order.map(i => A[i][i]), vectors: order.map(i => V.map(r => r[i])) };
  };
  ML.pca = (X, k = 2) => {
    const d = X[0].length, mu = ML.range(d).map(j => mean(X.map(r => r[j])));
    const C = ML.range(d).map(a => ML.range(d).map(b => X.reduce((s, r) => s + (r[a] - mu[a]) * (r[b] - mu[b]), 0) / (X.length - 1)));
    const { values, vectors } = ML.eigSym(C), total = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
    const comps = vectors.slice(0, k);
    return { components: comps, explained: values.slice(0, k), ratio: values.slice(0, k).map(v => v / total), allValues: values, mean: mu,
      transform: A => A.map(r => comps.map(c => c.reduce((s, w, j) => s + w * (r[j] - mu[j]), 0))) };
  };

  /* ---------- base helpers ---------- */
  const nClasses = y => Math.max(...y) + 1;
  class Base {
    predictAll(X) { return X.map(x => this.predict(x)); }
    probaAll(X) { return X.map(x => this.proba(x)); }
    score1(x) { const p = this.proba(x); return p[1] ?? 0; }
  }

  /* ---------- k-nearest neighbours ---------- */
  ML.KNN = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ k: 5, weights: "uniform", task: "clf" }, o); }
    fit(X, y) { this.X = X; this.y = y; this.K = this.o.task === "clf" ? nClasses(y) : 0; return this; }
    neighbors(x, k = this.o.k) {
      return this.X.map((r, i) => ({ i, d: Math.hypot(...r.map((v, j) => v - x[j])) })).sort((a, b) => a.d - b.d).slice(0, Math.min(k, this.X.length));
    }
    proba(x) {
      const nb = this.neighbors(x), p = new Array(this.K).fill(0);
      nb.forEach(({ i, d }) => (p[this.y[i]] += this.o.weights === "distance" ? 1 / (d + 1e-9) : 1));
      const s = p.reduce((a, b) => a + b, 0) || 1; return p.map(v => v / s);
    }
    predict(x) {
      if (this.o.task === "reg") { const nb = this.neighbors(x); let ws = 0, s = 0; nb.forEach(({ i, d }) => { const w = this.o.weights === "distance" ? 1 / (d + 1e-9) : 1; ws += w; s += w * this.y[i]; }); return s / ws; }
      return argmax(this.proba(x));
    }
  };

  /* ---------- logistic regression (binary) ---------- */
  ML.Logistic = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ lr: 0.5, epochs: 400, l2: 0, features: null, classWeight: null }, o); }
    phi(x) { return this.o.features ? this.o.features(x) : x; }
    fit(X, y) {
      const F = X.map(x => this.phi(x)), d = F[0].length, n = F.length, cw = this.o.classWeight || [1, 1];
      this.w = new Array(d).fill(0); this.b = 0; this.lossHistory = [];
      for (let e = 0; e < this.o.epochs; e++) {
        const gw = new Array(d).fill(0); let gb = 0, loss = 0, W = 0;
        for (let i = 0; i < n; i++) {
          const p = sigmoid(this.b + F[i].reduce((s, v, j) => s + v * this.w[j], 0)), c = cw[y[i]], r = c * (p - y[i]);
          W += c; gb += r; F[i].forEach((v, j) => (gw[j] += r * v));
          loss -= c * (y[i] ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12));
        }
        this.w = this.w.map((w, j) => w - this.o.lr * (gw[j] / W + this.o.l2 * w));
        this.b -= this.o.lr * gb / W;
        this.lossHistory.push(loss / W);
      }
      return this;
    }
    logit(x) { return this.b + this.phi(x).reduce((s, v, j) => s + v * this.w[j], 0); }
    proba(x) { const p = sigmoid(this.logit(x)); return [1 - p, p]; }
    predict(x, thr = 0.5) { return this.proba(x)[1] >= thr ? 1 : 0; }
  };

  /* ---------- linear regression: OLS, ridge (closed form), lasso / elastic net (coordinate descent) ---------- */
  ML.Linear = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ l2: 0, l1: 0, features: null, iters: 500 }, o); }
    phi(x) { return this.o.features ? this.o.features(x) : x; }
    fit(X, y) {
      const F = X.map(x => this.phi(x)), d = F[0].length, n = F.length;
      const mx = ML.range(d).map(j => mean(F.map(r => r[j]))), my = mean(y);
      const C = F.map(r => r.map((v, j) => v - mx[j])), yc = y.map(v => v - my);
      if (!this.o.l1) {
        const A = ML.range(d).map(a => ML.range(d).map(b => C.reduce((s, r) => s + r[a] * r[b], 0) + (a === b ? this.o.l2 * n : 0)));
        this.w = solve(A, ML.range(d).map(a => C.reduce((s, r, i) => s + r[a] * yc[i], 0)));
      } else { // minimise (1/2n)‖y − Xw‖² + l1‖w‖₁ + (l2/2)‖w‖²
        const w = new Array(d).fill(0), z = ML.range(d).map(j => C.reduce((s, r) => s + r[j] * r[j], 0) / n);
        const res = yc.slice();
        for (let it = 0; it < this.o.iters; it++) {
          let delta = 0;
          for (let j = 0; j < d; j++) {
            let rho = 0; for (let i = 0; i < n; i++) rho += C[i][j] * (res[i] + C[i][j] * w[j]); rho /= n;
            const nw = Math.sign(rho) * Math.max(0, Math.abs(rho) - this.o.l1) / (z[j] + this.o.l2 || 1e-12);
            if (nw !== w[j]) { for (let i = 0; i < n; i++) res[i] -= C[i][j] * (nw - w[j]); delta = Math.max(delta, Math.abs(nw - w[j])); w[j] = nw; }
          }
          if (delta < 1e-7) break;
        }
        this.w = w;
      }
      this.b = my - this.w.reduce((s, w, j) => s + w * mx[j], 0);
      return this;
    }
    predict(x) { return this.b + this.phi(x).reduce((s, v, j) => s + v * this.w[j], 0); }
  };

  /* ---------- decision tree (CART) ---------- */
  function impurity(stats, crit) {
    if (crit === "mse") { const { w, s, s2 } = stats; return w ? s2 / w - (s / w) ** 2 : 0; }
    const tot = stats.reduce((a, b) => a + b, 0); if (!tot) return 0;
    if (crit === "entropy") return -stats.reduce((h, c) => (c > 0 ? h + (c / tot) * Math.log2(c / tot) : h), 0);
    return 1 - stats.reduce((g, c) => g + (c / tot) ** 2, 0);
  }
  ML.impurity = (counts, crit = "gini") => impurity(counts, crit);
  ML.Tree = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ maxDepth: 4, minLeaf: 1, criterion: null, maxFeatures: null, task: "clf", seed: 1, minGain: 0 }, o); }
    fit(X, y, sw) {
      this.task = this.o.task; this.crit = this.o.criterion || (this.task === "reg" ? "mse" : "gini");
      this.K = this.task === "clf" ? Math.max(this.o.nClasses || 0, nClasses(y)) : 0;
      this.X = X; this.y = y; this.sw = sw || X.map(() => 1); this.d = X[0].length; this.r = ML.rng(this.o.seed);
      this.nLeaves = 0; this.depthReached = 0; this.importances = new Array(this.d).fill(0);
      this.root = this.grow(ML.range(X.length), 0);
      const t = this.importances.reduce((a, b) => a + b, 0) || 1; this.importances = this.importances.map(v => v / t);
      delete this.X; delete this.y; delete this.sw;
      return this;
    }
    stats(idx) {
      if (this.task === "reg") { let w = 0, s = 0, s2 = 0; idx.forEach(i => { const q = this.sw[i]; w += q; s += q * this.y[i]; s2 += q * this.y[i] ** 2; }); return { w, s, s2 }; }
      const c = new Array(this.K).fill(0); idx.forEach(i => (c[this.y[i]] += this.sw[i])); return c;
    }
    leafValue(st) {
      if (this.task === "reg") return st.w ? st.s / st.w : 0;
      const t = st.reduce((a, b) => a + b, 0) || 1; return st.map(c => c / t);
    }
    grow(idx, depth) {
      const st = this.stats(idx), W = this.task === "reg" ? st.w : st.reduce((a, b) => a + b, 0), imp = impurity(st, this.crit);
      const node = { n: idx.length, w: W, imp, depth, value: this.leafValue(st), leaf: true };
      this.depthReached = Math.max(this.depthReached, depth);
      if (depth >= this.o.maxDepth || idx.length < 2 * this.o.minLeaf || imp < 1e-12) { this.nLeaves++; return node; }
      let feats = ML.range(this.d), mf = this.o.maxFeatures;
      if (mf === "sqrt") mf = Math.max(1, Math.round(Math.sqrt(this.d)));
      if (mf && mf < this.d) feats = ML.shuffle(feats, this.r).slice(0, mf);
      let best = null;
      feats.forEach(f => {
        const s = idx.slice().sort((a, b) => this.X[a][f] - this.X[b][f]);
        let L = this.task === "reg" ? { w: 0, s: 0, s2: 0 } : new Array(this.K).fill(0);
        let R = this.task === "reg" ? { ...st } : st.slice();
        for (let k = 0; k < s.length - 1; k++) {
          const i = s[k], q = this.sw[i];
          if (this.task === "reg") { const v = this.y[i]; L.w += q; L.s += q * v; L.s2 += q * v * v; R.w -= q; R.s -= q * v; R.s2 -= q * v * v; }
          else { L[this.y[i]] += q; R[this.y[i]] -= q; }
          const a = this.X[i][f], b = this.X[s[k + 1]][f];
          if (a === b || k + 1 < this.o.minLeaf || s.length - k - 1 < this.o.minLeaf) continue;
          const wl = this.task === "reg" ? L.w : L.reduce((x, z) => x + z, 0), wr = W - wl;
          const child = (wl * impurity(L, this.crit) + wr * impurity(R, this.crit)) / W;
          const gain = imp - child;
          if (!best || gain > best.gain + 1e-12) best = { f, t: (a + b) / 2, gain };
        }
      });
      if (!best || best.gain <= this.o.minGain) { this.nLeaves++; return node; }
      this.importances[best.f] += W * best.gain;
      const li = idx.filter(i => this.X[i][best.f] <= best.t), ri = idx.filter(i => this.X[i][best.f] > best.t);
      return Object.assign(node, { leaf: false, f: best.f, t: best.t, gain: best.gain, left: this.grow(li, depth + 1), right: this.grow(ri, depth + 1) });
    }
    apply(x) { let n = this.root; while (!n.leaf) n = x[n.f] <= n.t ? n.left : n.right; return n; }
    proba(x) { return this.apply(x).value; }
    predict(x) { const v = this.apply(x).value; return this.task === "reg" ? v : argmax(v); }
    // leaf rectangles for 2-D drawing: [{x0, x1, y0, y1, value}] inside the given box
    regions(box) {
      const out = [];
      const walk = (n, b) => {
        if (n.leaf) { out.push({ ...b, value: n.value, n: n.n }); return; }
        const L = { ...b }, R = { ...b };
        if (n.f === 0) { L.x1 = Math.min(b.x1, n.t); R.x0 = Math.max(b.x0, n.t); } else { L.y1 = Math.min(b.y1, n.t); R.y0 = Math.max(b.y0, n.t); }
        walk(n.left, L); walk(n.right, R);
      };
      walk(this.root, box); return out;
    }
  };

  /* ---------- bagging / random forest ---------- */
  ML.Forest = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ nTrees: 50, maxDepth: 12, minLeaf: 1, maxFeatures: "sqrt", bootstrap: true, task: "clf", seed: 1 }, o); }
    fit(X, y) {
      const r = ML.rng(this.o.seed), n = X.length; this.task = this.o.task; this.K = this.task === "clf" ? nClasses(y) : 0;
      this.trees = []; this.oob = [];
      const votes = X.map(() => (this.task === "clf" ? new Array(this.K).fill(0) : { s: 0, c: 0 }));
      for (let t = 0; t < this.o.nTrees; t++) {
        const bs = this.o.bootstrap ? ML.bootstrap(n, r) : { idx: ML.range(n), oob: [] };
        const tree = new ML.Tree({ maxDepth: this.o.maxDepth, minLeaf: this.o.minLeaf, maxFeatures: this.o.maxFeatures, task: this.task, seed: Math.floor(r() * 1e9), nClasses: this.K })
          .fit(bs.idx.map(i => X[i]), bs.idx.map(i => y[i]));
        this.trees.push(tree); this.oob.push(bs.oob);
        bs.oob.forEach(i => { if (this.task === "clf") tree.proba(X[i]).forEach((p, k) => (votes[i][k] += p)); else { votes[i].s += tree.predict(X[i]); votes[i].c++; } });
      }
      const has = votes.map(v => (this.task === "clf" ? v.some(p => p > 0) : v.c > 0));
      const idx = ML.range(n).filter(i => has[i]);
      this.oobScore = !idx.length ? NaN : this.task === "clf" ? mean(idx.map(i => (argmax(votes[i]) === y[i] ? 1 : 0))) : ML.r2(idx.map(i => y[i]), idx.map(i => votes[i].s / votes[i].c));
      this.importances = ML.range(X[0].length).map(j => mean(this.trees.map(t => t.importances[j])));
      return this;
    }
    use(m) { const f = Object.create(this); f.trees = this.trees.slice(0, m); return f; } // the first m trees only
    proba(x) { const p = new Array(this.K).fill(0); this.trees.forEach(t => t.proba(x).forEach((v, k) => (p[k] += v / this.trees.length))); return p; }
    predict(x) { return this.task === "reg" ? mean(this.trees.map(t => t.predict(x))) : argmax(this.proba(x)); }
  };

  /* ---------- gradient boosting (squared loss for regression, log-loss for binary classification) ---------- */
  ML.GBM = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ nRounds: 50, lr: 0.1, maxDepth: 2, minLeaf: 1, task: "reg", subsample: 1, seed: 1 }, o); }
    fit(X, y) {
      const n = X.length, r = ML.rng(this.o.seed); this.task = this.o.task; this.trees = []; this.history = [];
      if (this.task === "reg") this.f0 = mean(y); else { const p = Math.min(0.999, Math.max(0.001, mean(y))); this.f0 = Math.log(p / (1 - p)); }
      const F = new Array(n).fill(this.f0);
      for (let m = 0; m < this.o.nRounds; m++) {
        const p = F.map(sigmoid), g = this.task === "reg" ? y.map((v, i) => v - F[i]) : y.map((v, i) => v - p[i]); // negative gradient = residual
        let idx = ML.range(n); if (this.o.subsample < 1) idx = idx.filter(() => r() < this.o.subsample);
        const tree = new ML.Tree({ maxDepth: this.o.maxDepth, minLeaf: this.o.minLeaf, task: "reg" }).fit(idx.map(i => X[i]), idx.map(i => g[i]));
        if (this.task === "clf") { // Newton step per leaf: Σ residual / Σ p(1 − p)
          const acc = new Map();
          idx.forEach(i => { const leaf = tree.apply(X[i]); const a = acc.get(leaf) || { g: 0, h: 0 }; a.g += g[i]; a.h += p[i] * (1 - p[i]); acc.set(leaf, a); });
          acc.forEach((a, leaf) => (leaf.value = a.g / Math.max(a.h, 1e-6)));
        }
        for (let i = 0; i < n; i++) F[i] += this.o.lr * tree.predict(X[i]);
        this.trees.push(tree);
        this.history.push(this.task === "reg" ? ML.mse(y, F) : ML.logloss(y, F.map(sigmoid)));
      }
      return this;
    }
    stage(k) { const g = Object.create(this); g.trees = this.trees.slice(0, k); return g; }
    raw(x) { return this.trees.reduce((s, t) => s + this.o.lr * t.predict(x), this.f0); }
    proba(x) { const p = sigmoid(this.raw(x)); return [1 - p, p]; }
    predict(x) { return this.task === "reg" ? this.raw(x) : (this.raw(x) >= 0 ? 1 : 0); }
  };

  /* ---------- AdaBoost with decision stumps (binary) ---------- */
  ML.AdaBoost = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ nRounds: 20, maxDepth: 1 }, o); }
    fit(X, y) {
      const n = X.length; let w = new Array(n).fill(1 / n); this.stumps = []; this.alphas = []; this.weightsHistory = [w.slice()]; this.errors = [];
      for (let m = 0; m < this.o.nRounds; m++) {
        const s = new ML.Tree({ maxDepth: this.o.maxDepth, task: "clf", nClasses: 2 }).fit(X, y, w);
        const miss = X.map((x, i) => (s.predict(x) !== y[i] ? 1 : 0));
        const err = Math.min(0.999, Math.max(1e-6, miss.reduce((a, v, i) => a + v * w[i], 0)));
        const alpha = 0.5 * Math.log((1 - err) / err);
        w = w.map((wi, i) => wi * Math.exp(miss[i] ? alpha : -alpha)); const Z = w.reduce((a, b) => a + b, 0); w = w.map(v => v / Z);
        this.stumps.push(s); this.alphas.push(alpha); this.errors.push(err); this.weightsHistory.push(w.slice());
      }
      return this;
    }
    stage(k) { const a = Object.create(this); a.stumps = this.stumps.slice(0, k); a.alphas = this.alphas.slice(0, k); return a; }
    raw(x) { return this.stumps.reduce((s, t, m) => s + this.alphas[m] * (t.predict(x) ? 1 : -1), 0); }
    proba(x) { const p = sigmoid(2 * this.raw(x)); return [1 - p, p]; }
    predict(x) { return this.raw(x) >= 0 ? 1 : 0; }
  };

  /* ---------- naive Bayes ---------- */
  ML.GaussianNB = class extends Base {
    fit(X, y) {
      this.K = nClasses(y); const d = X[0].length;
      this.prior = ML.range(this.K).map(k => y.filter(v => v === k).length / y.length);
      this.mu = ML.range(this.K).map(k => ML.range(d).map(j => mean(X.filter((_, i) => y[i] === k).map(r => r[j]))));
      this.var = ML.range(this.K).map(k => ML.range(d).map(j => mean(X.filter((_, i) => y[i] === k).map(r => (r[j] - this.mu[k][j]) ** 2)) + 1e-6));
      return this;
    }
    logJoint(x) { return ML.range(this.K).map(k => Math.log(this.prior[k] || 1e-12) + x.reduce((s, v, j) => s - 0.5 * Math.log(2 * Math.PI * this.var[k][j]) - (v - this.mu[k][j]) ** 2 / (2 * this.var[k][j]), 0)); }
    proba(x) { const l = this.logJoint(x), m = Math.max(...l), e = l.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
    predict(x) { return argmax(this.logJoint(x)); }
  };
  ML.tokenize = text => (String(text).toLowerCase().match(/[a-z0-9£$€']+/g) || []);
  ML.MultinomialNB = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ alpha: 1 }, o); }
    fit(docs, y) { // docs: arrays of tokens
      this.K = nClasses(y); this.vocab = new Set(); docs.forEach(d => d.forEach(t => this.vocab.add(t)));
      this.prior = ML.range(this.K).map(k => y.filter(v => v === k).length / y.length);
      this.count = ML.range(this.K).map(() => new Map()); this.total = new Array(this.K).fill(0);
      docs.forEach((d, i) => d.forEach(t => { const c = this.count[y[i]]; c.set(t, (c.get(t) || 0) + 1); this.total[y[i]]++; }));
      return this;
    }
    logp(t, k) { return Math.log(((this.count[k].get(t) || 0) + this.o.alpha) / (this.total[k] + this.o.alpha * this.vocab.size)); }
    logJoint(doc) { return ML.range(this.K).map(k => Math.log(this.prior[k] || 1e-12) + doc.filter(t => this.vocab.has(t)).reduce((s, t) => s + this.logp(t, k), 0)); }
    proba(doc) { const l = this.logJoint(doc), m = Math.max(...l), e = l.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
    predict(doc) { return argmax(this.logJoint(doc)); }
  };

  /* ---------- support vector machine (simplified SMO, binary) ---------- */
  ML.SVM = class extends Base {
    constructor(o = {}) { super(); this.o = Object.assign({ C: 1, kernel: "linear", gamma: 0.5, degree: 3, coef0: 1, iters: 30, tol: 1e-3, seed: 1 }, o); }
    k(a, b) {
      const dot = a.reduce((s, v, j) => s + v * b[j], 0);
      if (this.o.kernel === "rbf") return Math.exp(-this.o.gamma * a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0));
      if (this.o.kernel === "poly") return (this.o.gamma * dot + this.o.coef0) ** this.o.degree;
      return dot;
    }
    fit(X, y) {
      const n = X.length, Y = y.map(v => (v ? 1 : -1)), C = this.o.C, r = ML.rng(this.o.seed);
      const Kx = X.map(a => X.map(b => this.k(a, b)));
      const a = new Array(n).fill(0); let b = 0, passes = 0, guard = 0;
      const f = i => { let s = b; for (let j = 0; j < n; j++) if (a[j]) s += a[j] * Y[j] * Kx[j][i]; return s; };
      while (passes < this.o.iters && guard++ < 2000) {
        let changed = 0;
        for (let i = 0; i < n; i++) {
          const Ei = f(i) - Y[i];
          if ((Y[i] * Ei < -this.o.tol && a[i] < C) || (Y[i] * Ei > this.o.tol && a[i] > 0)) {
            let j = Math.floor(r() * (n - 1)); if (j >= i) j++;
            const Ej = f(j) - Y[j], ai = a[i], aj = a[j];
            const L = Y[i] !== Y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - C), H = Y[i] !== Y[j] ? Math.min(C, C + aj - ai) : Math.min(C, ai + aj);
            if (L === H) continue;
            const eta = 2 * Kx[i][j] - Kx[i][i] - Kx[j][j]; if (eta >= 0) continue;
            a[j] = Math.min(H, Math.max(L, aj - Y[j] * (Ei - Ej) / eta));
            if (Math.abs(a[j] - aj) < 1e-5) continue;
            a[i] = ai + Y[i] * Y[j] * (aj - a[j]);
            const b1 = b - Ei - Y[i] * (a[i] - ai) * Kx[i][i] - Y[j] * (a[j] - aj) * Kx[i][j];
            const b2 = b - Ej - Y[i] * (a[i] - ai) * Kx[i][j] - Y[j] * (a[j] - aj) * Kx[j][j];
            b = a[i] > 0 && a[i] < C ? b1 : a[j] > 0 && a[j] < C ? b2 : (b1 + b2) / 2;
            changed++;
          }
        }
        passes = changed ? 0 : passes + 1;
      }
      this.alpha = a; this.b = b; this.Y = Y;
      this.supportIdx = ML.range(n).filter(i => a[i] > 1e-6);
      this.SV = this.supportIdx.map(i => ({ x: X[i], ay: a[i] * Y[i] }));
      if (this.o.kernel === "linear") { this.w = ML.range(X[0].length).map(j => this.SV.reduce((s, v) => s + v.ay * v.x[j], 0)); }
      return this;
    }
    decision(x) { return this.SV.reduce((s, v) => s + v.ay * this.k(v.x, x), this.b); }
    proba(x) { const p = sigmoid(2 * this.decision(x)); return [1 - p, p]; }
    predict(x) { return this.decision(x) >= 0 ? 1 : 0; }
  };

  /* ---------- k-means ---------- */
  ML.KMeans = class {
    constructor(o = {}) { this.o = Object.assign({ k: 3, init: "++", seed: 1, maxIter: 100 }, o); }
    init(X) {
      this.X = X; const r = ML.rng(this.o.seed), k = this.o.k;
      if (this.o.init === "random") this.centers = ML.shuffle(ML.range(X.length), r).slice(0, k).map(i => X[i].slice());
      else {
        this.centers = [X[Math.floor(r() * X.length)].slice()];
        while (this.centers.length < k) {
          const d2 = X.map(x => Math.min(...this.centers.map(c => x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0))));
          let u = r() * d2.reduce((a, b) => a + b, 0), i = 0; for (; i < X.length - 1; i++) { u -= d2[i]; if (u <= 0) break; }
          this.centers.push(X[i].slice());
        }
      }
      this.labels = X.map(() => -1); this.iter = 0; this.assign(); return this;
    }
    assign() {
      let moved = 0; this.inertia = 0;
      this.X.forEach((x, i) => {
        let best = 0, bd = Infinity;
        this.centers.forEach((c, k) => { const d = x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0); if (d < bd) { bd = d; best = k; } });
        if (this.labels[i] !== best) moved++; this.labels[i] = best; this.inertia += bd;
      });
      return moved;
    }
    update() { this.centers = this.centers.map((c, k) => { const m = this.X.filter((_, i) => this.labels[i] === k); return m.length ? c.map((_, j) => mean(m.map(r => r[j]))) : c; }); }
    step() { this.update(); this.iter++; return this.assign(); } // returns how many points changed cluster
    fit(X) { this.init(X); for (let i = 0; i < this.o.maxIter; i++) if (!this.step()) break; return this; }
    predict(x) { let best = 0, bd = Infinity; this.centers.forEach((c, k) => { const d = x.reduce((s, v, j) => s + (v - c[j]) ** 2, 0); if (d < bd) { bd = d; best = k; } }); return best; }
  };

  /* ---------- metrics ---------- */
  ML.accuracy = (y, yh) => mean(y.map((v, i) => (v === yh[i] ? 1 : 0)));
  ML.mse = (y, yh) => mean(y.map((v, i) => (v - yh[i]) ** 2));
  ML.rmse = (y, yh) => Math.sqrt(ML.mse(y, yh));
  ML.mae = (y, yh) => mean(y.map((v, i) => Math.abs(v - yh[i])));
  ML.r2 = (y, yh) => { const m = mean(y), ss = y.reduce((s, v) => s + (v - m) ** 2, 0); return 1 - y.reduce((s, v, i) => s + (v - yh[i]) ** 2, 0) / (ss || 1); };
  ML.logloss = (y, p) => -mean(y.map((v, i) => { const q = Math.min(1 - 1e-12, Math.max(1e-12, p[i])); return v ? Math.log(q) : Math.log(1 - q); }));
  ML.brier = (y, p) => mean(y.map((v, i) => (p[i] - v) ** 2));
  ML.confusion = (y, s, thr = 0.5) => {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    y.forEach((v, i) => { const h = s[i] >= thr; if (h && v) tp++; else if (h) fp++; else if (v) fn++; else tn++; });
    const precision = tp + fp ? tp / (tp + fp) : 1, recall = tp + fn ? tp / (tp + fn) : 0;
    return { tp, fp, tn, fn, precision, recall, tpr: recall, fpr: fp + tn ? fp / (fp + tn) : 0, f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0, accuracy: (tp + tn) / (y.length || 1) };
  };
  ML.roc = (y, s) => {
    const o = ML.range(y.length).sort((a, b) => s[b] - s[a]), P = y.filter(v => v).length, N = y.length - P, pts = [[0, 0]];
    let tp = 0, fp = 0;
    o.forEach((i, k) => { if (y[i]) tp++; else fp++; if (k === o.length - 1 || s[o[k + 1]] !== s[i]) pts.push([fp / (N || 1), tp / (P || 1)]); });
    return pts;
  };
  ML.prCurve = (y, s) => {
    const o = ML.range(y.length).sort((a, b) => s[b] - s[a]), P = y.filter(v => v).length, pts = [];
    let tp = 0, fp = 0;
    o.forEach((i, k) => { if (y[i]) tp++; else fp++; if (k === o.length - 1 || s[o[k + 1]] !== s[i]) pts.push([tp / (P || 1), tp / (tp + fp)]); });
    return [[0, pts.length ? pts[0][1] : 1], ...pts];
  };
  ML.auc = pts => { let a = 0; for (let i = 1; i < pts.length; i++) a += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2; return a; };

  if (typeof window !== "undefined") window.ML = ML;
  if (typeof module !== "undefined") module.exports = ML;
})();

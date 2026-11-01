/* ============ NN: a tiny deep-learning toy kit for playgrounds (no dependencies, plain JS) ============
 * Everything runs in the browser on toy sizes (2D points, 8×8 images, 6-token sentences). Real models are huge;
 * the mechanics are the same. Load after course.js:  <script src="../assets/nn.js"></script>
 *
 * Random:      NN.rng(seed) → { u() uniform [0,1), n() standard normal, int(k) }
 * Activations: NN.act[name] = { f(z), d(z, a) slope, label, formula, range }   names: linear step sigmoid tanh relu
 *              leaky (slope NN.LEAK = 0.01, or pass {leak} to MLP) elu gelu silu softplus. NN.erf(x), NN.phi(x) (normal CDF).
 * Data (2D):   NN.data.xor|circle|spiral|moons|gauss|linear(n, {noise, seed}) → [{ x: [x1, x2], y: 0|1 }], coords ≈ [-1, 1]
 * Network:     const net = new NN.MLP({ sizes: [2, 8, 8, 1], act: "relu", out: "sigmoid"|"linear"|"softmax",
 *                init: "he"|"xavier"|"lecun"|"normal"|"zeros", initScale: 1, seed: 1, leak: 0.01 });
 *              net.predict([x1, x2]) → output array · net.forward(x) → { zs, as } (pre-activations, activations per layer)
 *              net.train(batch, { lr, opt: "sgd"|"momentum"|"adam"|"adamw", loss: "bce"|"mse"|"ce", wd, clip })
 *                → { loss, gradNorms[] (per layer, input→output), gradNorm (global, before clipping) }
 *              net.stats(batch) → per hidden layer { dead (fraction of units off on every input), sat (fraction of
 *                (unit, input) pairs with slope < 5% of the max), mean, std, values[] (all activations, for histograms) }
 *              net.nParams(), net.layers (W: Float64Array out×in row-major, b, nIn, nOut), net.reset()
 * Optimizers:  NN.optim(name, { lr, beta, beta1, beta2, eps, wd }) → { step(params: Float64Array|number[], grads, key) }
 *              names sgd momentum nesterov rmsprop adam adamw (wd = L2 for sgd/momentum/adam, decoupled for adamw)
 * Surfaces:    NN.surfaces.bowl|ravine|rosenbrock|saddle|bumpy = { f(x, y), g(x, y) → [gx, gy], xr, yr, start, min }
 * Images:      NN.conv2d(img, k, { stride, pad }) · NN.pool(img, size, stride, "max"|"avg") · NN.img.digit(d) (8×8 0..1)
 *              NN.img.make(name, n) names: "seven" "cross" "square" "smile" "stripes" "edge" "checker"
 * Boxes:       NN.iou(a, b) with boxes [x1, y1, x2, y2] · NN.nms(dets [{ box, score }], thr) → kept indices (by score)
 * Attention:   NN.attention(Q, K, V, { causal, scale }) → { scores, weights, out } (Q n×d, K m×d, V m×dv, plain arrays)
 * Audio:       NN.fft(re, im) in place (length 2^k) · NN.stft(signal, { n: 256, hop: 64 }) → frames × (n/2) magnitudes
 * Drawing:     NN.drawBoundary(canvas, f(x, y) → p∈[0,1], data, { range, res, classColors })
 *              NN.heat(canvas, M, { lo, hi, cmap: "seq"|"div"|"gray", color: "--c-at", text: true, fmt, rowLabels,
 *                colLabels, highlight: [r, c] }) · NN.onCells(canvas, rows, cols, (r, c, e) => …) (click + drag painting)
 *              NN.cmap.seq(t, color) / NN.cmap.div(t) / NN.cmap.gray(t) → CSS color · NN.histogram(values, bins, lo, hi)
 * Loop:        const L = NN.loop(stepFn, { onStop }); L.start(); L.stop(); L.running   (requestAnimationFrame)
 * All randomness is seeded so the prose can promise what the learner will see.
 */
(function () {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const col = c => (c && c.startsWith("--") ? css(c) : c);

  /* ---------- seeded random ---------- */
  function rng(seed) {
    let a = (seed == null ? 1 : seed) >>> 0, spare = null;
    const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const n = () => {
      if (spare != null) { const s = spare; spare = null; return s; }
      let x = 0, y = 0; while (x === 0) x = u(); y = u();
      const r = Math.sqrt(-2 * Math.log(x)); spare = r * Math.sin(2 * Math.PI * y); return r * Math.cos(2 * Math.PI * y);
    };
    return { u, n, int: k => Math.floor(u() * k) };
  }

  /* ---------- activations ---------- */
  function erf(x) { // Abramowitz–Stegun 7.1.26, |error| < 1.5e-7
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const phi = x => 0.5 * (1 + erf(x / Math.SQRT2));           // standard normal CDF Φ
  const pdf = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  const sig = z => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  const act = {
    linear:   { label: "Linear (none)", formula: "f(z) = z", range: "(−∞, ∞)", f: z => z, d: () => 1 },
    step:     { label: "Step", formula: "f(z) = 1 if z > 0 else 0", range: "{0, 1}", f: z => (z > 0 ? 1 : 0), d: () => 0 },
    sigmoid:  { label: "Sigmoid", formula: "σ(z) = 1 / (1 + e^−z)", range: "(0, 1)", f: sig, d: (z, a) => { const s = a == null ? sig(z) : a; return s * (1 - s); } },
    tanh:     { label: "Tanh", formula: "tanh(z) = (e^z − e^−z) / (e^z + e^−z)", range: "(−1, 1)", f: Math.tanh, d: (z, a) => { const t = a == null ? Math.tanh(z) : a; return 1 - t * t; } },
    relu:     { label: "ReLU", formula: "max(0, z)", range: "[0, ∞)", f: z => (z > 0 ? z : 0), d: z => (z > 0 ? 1 : 0) },
    leaky:    { label: "Leaky ReLU", formula: "z if z > 0 else 0.01·z", range: "(−∞, ∞)", f: (z, k) => (z > 0 ? z : (k == null ? NN.LEAK : k) * z), d: (z, a, k) => (z > 0 ? 1 : (k == null ? NN.LEAK : k)) },
    elu:      { label: "ELU", formula: "z if z > 0 else e^z − 1", range: "(−1, ∞)", f: z => (z > 0 ? z : Math.expm1(z)), d: z => (z > 0 ? 1 : Math.exp(z)) },
    gelu:     { label: "GELU", formula: "z · Φ(z)", range: "(≈ −0.17, ∞)", f: z => z * phi(z), d: z => phi(z) + z * pdf(z) },
    silu:     { label: "SiLU / Swish", formula: "z · σ(z)", range: "(≈ −0.28, ∞)", f: z => z * sig(z), d: z => { const s = sig(z); return s * (1 + z * (1 - s)); } },
    softplus: { label: "Softplus", formula: "ln(1 + e^z)", range: "(0, ∞)", f: z => (z > 30 ? z : Math.log1p(Math.exp(z))), d: z => sig(z) },
  };
  // the largest slope each activation can have (used to define "saturated")
  const maxSlope = { linear: 1, step: 1, sigmoid: 0.25, tanh: 1, relu: 1, leaky: 1, elu: 1, gelu: 1.13, silu: 1.1, softplus: 1 };

  /* ---------- 2D toy datasets (coordinates roughly in [-1, 1]) ---------- */
  const data = {
    xor(n = 200, o = {}) { const r = rng(o.seed ?? 3), nz = o.noise ?? 0.12, out = [];
      for (let i = 0; i < n; i++) { const sx = r.u() < 0.5 ? -1 : 1, sy = r.u() < 0.5 ? -1 : 1;
        out.push({ x: [sx * (0.2 + 0.6 * r.u()) + nz * r.n() * 0.5, sy * (0.2 + 0.6 * r.u()) + nz * r.n() * 0.5], y: sx * sy > 0 ? 1 : 0 }); }
      return out; },
    circle(n = 200, o = {}) { const r = rng(o.seed ?? 4), nz = o.noise ?? 0.05, out = [];
      for (let i = 0; i < n; i++) { const inner = i % 2 === 0, rad = inner ? 0.45 * Math.sqrt(r.u()) : 0.65 + 0.3 * r.u(), t = 2 * Math.PI * r.u();
        out.push({ x: [rad * Math.cos(t) + nz * r.n(), rad * Math.sin(t) + nz * r.n()], y: inner ? 1 : 0 }); }
      return out; },
    spiral(n = 200, o = {}) { const r = rng(o.seed ?? 5), nz = o.noise ?? 0.04, out = [];
      for (let i = 0; i < n; i++) { const c = i % 2, t = 0.25 + 2.6 * (Math.floor(i / 2) / (n / 2)), ang = t * 2.2 + c * Math.PI, rad = t / 2.9;
        out.push({ x: [rad * Math.cos(ang) + nz * r.n(), rad * Math.sin(ang) + nz * r.n()], y: c }); }
      return out; },
    moons(n = 200, o = {}) { const r = rng(o.seed ?? 6), nz = o.noise ?? 0.08, out = [];
      for (let i = 0; i < n; i++) { const c = i % 2, t = Math.PI * r.u();
        const x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? 0.5 - Math.sin(t) : Math.sin(t);
        out.push({ x: [(x - 0.5) * 0.75 + nz * r.n(), (y - 0.25) * 0.9 + nz * r.n()], y: c }); }
      return out; },
    gauss(n = 200, o = {}) { const r = rng(o.seed ?? 7), nz = o.noise ?? 0.2, out = [];
      for (let i = 0; i < n; i++) { const c = i % 2, m = c ? 0.45 : -0.45; out.push({ x: [m + nz * r.n(), m + nz * r.n()], y: c }); }
      return out; },
    linear(n = 200, o = {}) { const r = rng(o.seed ?? 8), out = [];
      for (let i = 0; i < n; i++) { const x = [2 * r.u() - 1, 2 * r.u() - 1]; out.push({ x, y: x[0] * 0.6 + x[1] > 0.1 ? 1 : 0 }); }
      return out; },
  };

  /* ---------- optimizers (work on flat arrays; `key` keeps separate state per tensor) ---------- */
  function optim(name, o = {}) {
    const lr = o.lr ?? 0.01, beta = o.beta ?? 0.9, b1 = o.beta1 ?? 0.9, b2 = o.beta2 ?? 0.999, eps = o.eps ?? 1e-8, wd = o.wd ?? 0;
    const st = {};
    return {
      name, o, t: 0,
      tick() { this.t++; },               // call once per update of all tensors (Adam bias correction)
      step(p, g, key = "p") {
        const s = st[key] || (st[key] = { m: new Float64Array(p.length), v: new Float64Array(p.length) });
        const t = Math.max(1, this.t), lrNow = this.o.lr ?? lr;
        for (let i = 0; i < p.length; i++) {
          let gi = g[i] + (name !== "adamw" && wd ? wd * p[i] : 0);
          if (name === "sgd") p[i] -= lrNow * gi;
          else if (name === "momentum") { s.m[i] = beta * s.m[i] + gi; p[i] -= lrNow * s.m[i]; }
          else if (name === "nesterov") { s.m[i] = beta * s.m[i] + gi; p[i] -= lrNow * (gi + beta * s.m[i]); }
          else if (name === "rmsprop") { s.v[i] = b2 * s.v[i] + (1 - b2) * gi * gi; p[i] -= lrNow * gi / (Math.sqrt(s.v[i]) + eps); }
          else { // adam / adamw
            s.m[i] = b1 * s.m[i] + (1 - b1) * gi; s.v[i] = b2 * s.v[i] + (1 - b2) * gi * gi;
            const mh = s.m[i] / (1 - Math.pow(b1, t)), vh = s.v[i] / (1 - Math.pow(b2, t));
            p[i] -= lrNow * (mh / (Math.sqrt(vh) + eps) + (name === "adamw" ? wd * p[i] : 0));
          }
        }
      },
    };
  }

  /* ---------- the MLP ---------- */
  class MLP {
    constructor(o = {}) {
      this.o = Object.assign({ sizes: [2, 8, 8, 1], act: "relu", out: "sigmoid", init: "he", initScale: 1, seed: 1, leak: 0.01 }, o);
      this.reset();
    }
    reset(seed) {
      const o = this.o, r = rng(seed ?? o.seed);
      this.layers = [];
      for (let l = 0; l < o.sizes.length - 1; l++) {
        const nIn = o.sizes[l], nOut = o.sizes[l + 1];
        const std = { he: Math.sqrt(2 / nIn), xavier: Math.sqrt(2 / (nIn + nOut)), lecun: Math.sqrt(1 / nIn), normal: 1, zeros: 0 }[o.init] ?? Math.sqrt(2 / nIn);
        const W = new Float64Array(nIn * nOut).map(() => std * o.initScale * r.n());
        this.layers.push({ W, b: new Float64Array(nOut), nIn, nOut });
      }
      this.opt = null; this.optKey = "";
      return this;
    }
    nParams() { return this.layers.reduce((s, L) => s + L.W.length + L.b.length, 0); }
    f(z) { return act[this.o.act].f(z, this.o.leak); }
    d(z, a) { return act[this.o.act].d(z, a, this.o.leak); }
    forward(x) {
      const zs = [], as = [Float64Array.from(x)], last = this.layers.length - 1;
      this.layers.forEach((L, l) => {
        const a = as[l], z = new Float64Array(L.nOut);
        for (let j = 0; j < L.nOut; j++) { let s = L.b[j]; const off = j * L.nIn; for (let i = 0; i < L.nIn; i++) s += L.W[off + i] * a[i]; z[j] = s; }
        zs.push(z);
        let out;
        if (l < last) out = z.map(v => this.f(v));
        else if (this.o.out === "sigmoid") out = z.map(sig);
        else if (this.o.out === "softmax") { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((p, q) => p + q, 0); out = e.map(v => v / s); }
        else out = Float64Array.from(z);
        as.push(out);
      });
      return { zs, as };
    }
    predict(x) { const { as } = this.forward(x); return Array.from(as[as.length - 1]); }
    // one full-batch gradient step; returns the loss before the step
    train(batch, o = {}) {
      const loss = o.loss || (this.o.out === "softmax" ? "ce" : this.o.out === "sigmoid" ? "bce" : "mse");
      const key = (o.opt || "sgd") + "|" + (o.wd || 0);
      if (!this.opt || this.optKey !== key) { this.opt = optim(o.opt || "sgd", { lr: o.lr, wd: o.wd, beta: o.beta }); this.optKey = key; }
      this.opt.o.lr = o.lr ?? 0.1;
      const gW = this.layers.map(L => new Float64Array(L.W.length)), gb = this.layers.map(L => new Float64Array(L.b.length));
      const last = this.layers.length - 1;
      let total = 0;
      batch.forEach(s => {
        const { zs, as } = this.forward(s.x), p = as[as.length - 1];
        const ys = this.o.out === "softmax" ? p.map((_, k) => (k === s.y ? 1 : 0)) : Array.isArray(s.y) ? s.y : [s.y];
        let delta = new Float64Array(p.length);
        for (let k = 0; k < p.length; k++) {
          const pk = Math.min(1 - 1e-12, Math.max(1e-12, p[k]));
          if (loss === "bce") total += -(ys[k] * Math.log(pk) + (1 - ys[k]) * Math.log(1 - pk));
          else if (loss === "ce") total += ys[k] ? -Math.log(pk) : 0;
          else total += (p[k] - ys[k]) ** 2;
          if ((loss === "bce" && this.o.out === "sigmoid") || (loss === "ce" && this.o.out === "softmax")) delta[k] = p[k] - ys[k];
          else if (loss === "mse") delta[k] = 2 * (p[k] - ys[k]) * (this.o.out === "sigmoid" ? p[k] * (1 - p[k]) : 1);
          else delta[k] = (pk - ys[k]) / (pk * (1 - pk)) * (this.o.out === "sigmoid" ? p[k] * (1 - p[k]) : 1);
        }
        for (let l = last; l >= 0; l--) {
          const L = this.layers[l], a = as[l];
          for (let j = 0; j < L.nOut; j++) { const dj = delta[j]; if (!dj) continue; gb[l][j] += dj; const off = j * L.nIn; for (let i = 0; i < L.nIn; i++) gW[l][off + i] += dj * a[i]; }
          if (l > 0) {
            const prev = new Float64Array(L.nIn), zp = zs[l - 1], ap = as[l];
            for (let j = 0; j < L.nOut; j++) { const dj = delta[j]; if (!dj) continue; const off = j * L.nIn; for (let i = 0; i < L.nIn; i++) prev[i] += L.W[off + i] * dj; }
            for (let i = 0; i < L.nIn; i++) prev[i] *= this.d(zp[i], ap[i]);
            delta = prev;
          }
        }
      });
      const n = batch.length;
      gW.forEach(g => { for (let i = 0; i < g.length; i++) g[i] /= n; });
      gb.forEach(g => { for (let i = 0; i < g.length; i++) g[i] /= n; });
      const gradNorms = gW.map((g, l) => Math.sqrt(g.reduce((s, v) => s + v * v, 0) + gb[l].reduce((s, v) => s + v * v, 0)));
      const gradNorm = Math.sqrt(gradNorms.reduce((s, v) => s + v * v, 0));
      if (o.clip && gradNorm > o.clip) { const k = o.clip / gradNorm; gW.forEach(g => g.forEach((v, i) => (g[i] = v * k))); gb.forEach(g => g.forEach((v, i) => (g[i] = v * k))); }
      this.opt.tick();
      this.layers.forEach((L, l) => { this.opt.step(L.W, gW[l], "W" + l); this.opt.step(L.b, gb[l], "b" + l); });
      return { loss: total / n, gradNorms, gradNorm };
    }
    stats(batch) {
      const hidden = this.layers.length - 1, ms = maxSlope[this.o.act] || 1;
      const out = Array.from({ length: hidden }, (_, l) => ({ on: new Uint8Array(this.layers[l].nOut), satN: 0, n: 0, values: [] }));
      batch.forEach(s => {
        const { zs, as } = this.forward(s.x);
        for (let l = 0; l < hidden; l++) {
          const z = zs[l], a = as[l + 1], S = out[l];
          for (let j = 0; j < z.length; j++) {
            if (Math.abs(this.d(z[j], a[j])) > 1e-6) S.on[j] = 1;   // this input lets gradient through this unit
            if (Math.abs(this.d(z[j], a[j])) < 0.05 * ms) S.satN++;
            S.n++; S.values.push(a[j]);
          }
        }
      });
      return out.map(S => {
        const m = S.values.reduce((p, q) => p + q, 0) / (S.values.length || 1);
        const sd = Math.sqrt(S.values.reduce((p, q) => p + (q - m) ** 2, 0) / (S.values.length || 1));
        return { dead: S.on.reduce((p, q) => p + (q ? 0 : 1), 0) / S.on.length, sat: S.satN / (S.n || 1), mean: m, std: sd, values: S.values };
      });
    }
  }

  /* ---------- loss surfaces for optimizer races ---------- */
  const surfaces = {
    bowl:       { f: (x, y) => x * x + y * y, g: (x, y) => [2 * x, 2 * y], xr: [-3, 3], yr: [-3, 3], start: [-2.5, 2.2], min: [0, 0] },
    ravine:     { f: (x, y) => 0.05 * x * x + 2 * y * y, g: (x, y) => [0.1 * x, 4 * y], xr: [-6, 6], yr: [-3, 3], start: [-5.5, 1.5], min: [0, 0] },
    rosenbrock: { f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)], xr: [-2, 2], yr: [-1, 3], start: [-1.5, 2.5], min: [1, 1] },
    saddle:     { f: (x, y) => x * x - y * y + 0.25 * y ** 4, g: (x, y) => [2 * x, -2 * y + y ** 3], xr: [-2.5, 2.5], yr: [-2.5, 2.5], start: [-2, 0.02], min: [0, Math.SQRT2] },
    bumpy:      { f: (x, y) => 0.1 * (x * x + y * y) - Math.cos(2 * x) * Math.cos(2 * y) + 1, g: (x, y) => [0.2 * x + 2 * Math.sin(2 * x) * Math.cos(2 * y), 0.2 * y + 2 * Math.cos(2 * x) * Math.sin(2 * y)], xr: [-4, 4], yr: [-4, 4], start: [-3.2, 3.0], min: [0, 0] },
  };

  /* ---------- images: convolution, pooling, tiny pictures ---------- */
  function conv2d(img, k, o = {}) {
    const s = o.stride || 1, p = o.pad || 0, H = img.length, W = img[0].length, kh = k.length, kw = k[0].length;
    const oh = Math.floor((H + 2 * p - kh) / s) + 1, ow = Math.floor((W + 2 * p - kw) / s) + 1, out = [];
    for (let r = 0; r < oh; r++) { const row = [];
      for (let c = 0; c < ow; c++) { let v = 0;
        for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) { const y = r * s + i - p, x = c * s + j - p; if (y >= 0 && y < H && x >= 0 && x < W) v += img[y][x] * k[i][j]; }
        row.push(v); }
      out.push(row); }
    return out;
  }
  function pool(img, size = 2, stride = size, mode = "max") {
    const H = img.length, W = img[0].length, out = [];
    for (let r = 0; r + size <= H; r += stride) { const row = [];
      for (let c = 0; c + size <= W; c += stride) { const v = []; for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) v.push(img[r + i][c + j]);
        row.push(mode === "avg" ? v.reduce((a, b) => a + b, 0) / v.length : Math.max(...v)); }
      out.push(row); }
    return out;
  }
  const PICS = {
    seven:   ["........", ".######.", "......#.", ".....#..", "....#...", "...#....", "...#....", "........"],
    cross:   ["........", "...##...", "...##...", ".######.", ".######.", "...##...", "...##...", "........"],
    square:  ["........", ".######.", ".#....#.", ".#....#.", ".#....#.", ".#....#.", ".######.", "........"],
    smile:   ["........", "..#..#..", "..#..#..", "........", ".#....#.", "..#..#..", "...##...", "........"],
    stripes: ["#.#.#.#.", "#.#.#.#.", "#.#.#.#.", "#.#.#.#.", "#.#.#.#.", "#.#.#.#.", "#.#.#.#.", "#.#.#.#."],
    edge:    ["....####", "....####", "....####", "....####", "....####", "....####", "....####", "....####"],
    checker: ["##..##..", "##..##..", "..##..##", "..##..##", "##..##..", "##..##..", "..##..##", "..##..##"],
  };
  const DIGITS = [
    ["..####..", ".#....#.", ".#....#.", ".#....#.", ".#....#.", ".#....#.", "..####..", "........"],
    ["...##...", "..###...", "...##...", "...##...", "...##...", "...##...", "..####..", "........"],
    ["..####..", ".#....#.", "......#.", ".....#..", "...##...", "..#.....", ".######.", "........"],
    ["..####..", ".#....#.", "......#.", "...###..", "......#.", ".#....#.", "..####..", "........"],
    [".....#..", "....##..", "...#.#..", "..#..#..", ".######.", ".....#..", ".....#..", "........"],
    [".######.", ".#......", ".#####..", "......#.", "......#.", ".#....#.", "..####..", "........"],
    ["..####..", ".#......", ".#......", ".#####..", ".#....#.", ".#....#.", "..####..", "........"],
    PICS.seven,
    ["..####..", ".#....#.", ".#....#.", "..####..", ".#....#.", ".#....#.", "..####..", "........"],
    ["..####..", ".#....#.", ".#....#.", "..#####.", "......#.", "......#.", "..####..", "........"],
  ];
  const fromRows = rows => rows.map(r => [...r].map(ch => (ch === "#" ? 1 : 0)));
  const img = {
    make(name, n) { const m = fromRows(PICS[name] || PICS.seven); if (!n || n === 8) return m;
      return Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => m[Math.floor(r * 8 / n)][Math.floor(c * 8 / n)])); },
    digit(d) { return fromRows(DIGITS[((d % 10) + 10) % 10]); },
    names: Object.keys(PICS),
  };

  /* ---------- boxes ---------- */
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy, ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ua > 0 ? inter / ua : 0;
  }
  function nms(dets, thr = 0.5) {
    const order = dets.map((d, i) => i).sort((i, j) => dets[j].score - dets[i].score), keep = [];
    order.forEach(i => { if (keep.every(k => iou(dets[i].box, dets[k].box) <= thr)) keep.push(i); });
    return keep;
  }

  /* ---------- attention ---------- */
  function attention(Q, K, V, o = {}) {
    const d = Q[0].length, scale = o.scale ?? 1 / Math.sqrt(d);
    const scores = Q.map((q, i) => K.map((k, j) => (o.causal && j > i ? -Infinity : scale * q.reduce((s, v, t) => s + v * k[t], 0))));
    const weights = scores.map(row => { const m = Math.max(...row), e = row.map(v => (v === -Infinity ? 0 : Math.exp(v - m))), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); });
    const out = weights.map(w => V[0].map((_, t) => w.reduce((s, wj, j) => s + wj * V[j][t], 0)));
    return { scores, weights, out };
  }

  /* ---------- audio ---------- */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        } }
    }
  }
  function stft(signal, o = {}) {
    const n = o.n || 256, hop = o.hop || n / 4, frames = [];
    for (let s = 0; s + n <= signal.length; s += hop) {
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) re[i] = signal[s + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1))); // Hann window
      fft(re, im);
      frames.push(Array.from({ length: n / 2 }, (_, k) => Math.hypot(re[k], im[k])));
    }
    return frames;
  }

  /* ---------- colors and drawing ---------- */
  const ctx2 = document.createElement("canvas").getContext("2d");
  function rgb(c) { ctx2.fillStyle = "#000"; ctx2.fillStyle = col(c) || c; const v = ctx2.fillStyle;
    if (v[0] === "#") { const n = parseInt(v.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
    const m = v.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [0, 0, 0]; }
  const mixRGB = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const str = c => `rgb(${c[0]},${c[1]},${c[2]})`;
  const clamp01 = t => Math.max(0, Math.min(1, t));
  const cmap = {
    seq: (t, c) => str(mixRGB(rgb("--bg2"), rgb(c || "--c-at"), clamp01(t))),
    div: t => { t = Math.max(-1, Math.min(1, t)); return str(mixRGB(rgb("--bg2"), rgb(t < 0 ? "#2f80ed" : "#e0474c"), Math.abs(t))); },
    gray: t => { const v = Math.round(255 * clamp01(t)); return `rgb(${v},${v},${v})`; },
  };
  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || canvas.width || 300, h = canvas.clientHeight || canvas.height || 300;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, w, h };
  }
  function drawBoundary(canvas, f, pts, o = {}) {
    const { ctx, w, h } = fit(canvas), R = o.range || 1.25, res = o.res || 50;
    const c0 = rgb((o.classColors || [])[0] || "#e17055"), c1 = rgb((o.classColors || [])[1] || "#2f80ed"), bg = rgb("--bg2");
    const cw = w / res, ch = h / res;
    for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
      const x = -R + (2 * R) * (i + 0.5) / res, y = R - (2 * R) * (j + 0.5) / res, p = clamp01(f(x, y));
      const base = p > 0.5 ? c1 : c0, k = Math.min(1, Math.abs(p - 0.5) * 2) * 0.55;
      ctx.fillStyle = str(mixRGB(bg, base, k)); ctx.fillRect(i * cw, j * ch, cw + 1, ch + 1);
    }
    (pts || []).forEach(s => {
      const px = (s.x[0] + R) / (2 * R) * w, py = (R - s.x[1]) / (2 * R) * h;
      ctx.beginPath(); ctx.arc(px, py, o.r || 3.2, 0, 7); ctx.fillStyle = str(s.y ? c1 : c0); ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = css("--bg2"); ctx.stroke();
    });
  }
  function heat(canvas, M, o = {}) {
    const { ctx, w, h } = fit(canvas), rows = M.length, cols = M[0].length;
    const flat = M.flat().filter(v => isFinite(v));
    const lo = o.lo ?? Math.min(...flat), hi = o.hi ?? Math.max(...flat);
    const padL = o.rowLabels ? (o.padLeft || 56) : 0, padT = o.colLabels ? (o.padTop || 22) : 0;
    const cw = (w - padL) / cols, chh = (h - padT) / rows;
    ctx.clearRect(0, 0, w, h);
    ctx.font = (o.fontSize || 11) + "px " + css("--mono");
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = M[r][c];
      let fill;
      if (!isFinite(v)) fill = css("--panel");
      else if (o.cmap === "div") { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; fill = cmap.div(v / m); }
      else if (o.cmap === "gray") fill = cmap.gray((v - lo) / ((hi - lo) || 1));
      else fill = cmap.seq((v - lo) / ((hi - lo) || 1), o.color);
      ctx.fillStyle = fill; ctx.fillRect(padL + c * cw, padT + r * chh, cw - (o.gap ?? 1), chh - (o.gap ?? 1));
      if (o.text && isFinite(v) && cw > 22) {
        const t = o.cmap === "gray" ? (v - lo) / ((hi - lo) || 1) : 0;
        ctx.fillStyle = o.cmap === "gray" && t > 0.55 ? "#111" : o.cmap === "gray" ? "#eee" : css("--text");
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(o.fmt ? o.fmt(v) : (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(o.digits ?? 2)), padL + c * cw + cw / 2, padT + r * chh + chh / 2);
      }
    }
    if (o.highlight) { const [r, c] = o.highlight; ctx.strokeStyle = css("--accent"); ctx.lineWidth = 3; ctx.strokeRect(padL + c * cw + 1.5, padT + r * chh + 1.5, cw - 4, chh - 4); }
    ctx.fillStyle = css("--muted");
    if (o.rowLabels) { ctx.textAlign = "right"; ctx.textBaseline = "middle"; o.rowLabels.forEach((t, r) => ctx.fillText(t, padL - 6, padT + r * chh + chh / 2)); }
    if (o.colLabels) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; o.colLabels.forEach((t, c) => ctx.fillText(t, padL + c * cw + cw / 2, padT - 4)); }
    canvas._heat = { rows, cols, padL, padT, w, h };
  }
  function onCells(canvas, rows, cols, cb) {
    let down = false;
    const hit = e => {
      const r = canvas.getBoundingClientRect(), g = canvas._heat || { padL: 0, padT: 0, w: r.width, h: r.height };
      const x = e.clientX - r.left - g.padL, y = e.clientY - r.top - g.padT;
      const c = Math.floor(x / ((r.width - g.padL) / cols)), rr = Math.floor(y / ((r.height - g.padT) / rows));
      if (rr >= 0 && rr < rows && c >= 0 && c < cols) cb(rr, c, e);
    };
    canvas.addEventListener("pointerdown", e => { down = true; canvas.setPointerCapture(e.pointerId); hit(e); e.preventDefault(); });
    canvas.addEventListener("pointermove", e => { if (down) hit(e); });
    const end = () => (down = false);
    canvas.addEventListener("pointerup", end); canvas.addEventListener("pointercancel", end);
  }
  function histogram(values, bins = 20, lo, hi) {
    lo = lo ?? Math.min(...values); hi = hi ?? Math.max(...values);
    const out = new Array(bins).fill(0), wdt = (hi - lo) / bins || 1;
    values.forEach(v => { const k = Math.floor((v - lo) / wdt); if (k >= 0 && k < bins) out[k]++; else if (v === hi) out[bins - 1]++; });
    return out;
  }
  function loop(stepFn, o = {}) {
    let id = null;
    const L = {
      get running() { return id != null; },
      start() { if (id != null) return; const tick = () => { if (stepFn() === false) { L.stop(); return; } id = requestAnimationFrame(tick); }; id = requestAnimationFrame(tick); },
      stop() { if (id != null) cancelAnimationFrame(id); id = null; if (o.onStop) o.onStop(); },
    };
    return L;
  }

  const NN = { LEAK: 0.01, rng, erf, phi, sigmoid: sig, act, maxSlope, data, optim, MLP, surfaces, conv2d, pool, img, iou, nms, attention, fft, stft, cmap, rgb, drawBoundary, heat, onCells, histogram, loop, fit };
  window.NN = NN;
})();

/* ============ Gridworld MDP engine (Phase 2+) ============
 * Same world as Phase 0: 5x5, start (4,0), 💎 (0,4) +10, 🔥 (1,3) −10, walls, step −1.
 * GW.model({slip, step}) → P[k][a] = [{p, s2, r, done}]
 * GW.evalSweep / GW.viSweep / GW.q / GW.greedy / GW.evaluate / GW.valueIteration
 * new GW.Grid(el, {onClick}) .draw({V, pi, highlight, vscale, showValues})
 */
(function () {
  const style = document.createElement("style");
  style.textContent = `
  .gw2 { display:grid; grid-template-columns:repeat(5, var(--cell,66px)); gap:4px; user-select:none; }
  .gw2-cell { position:relative; width:var(--cell,66px); height:var(--cell,66px); border-radius:10px; border:1px solid var(--line);
    background:var(--bg2); display:flex; align-items:center; justify-content:center; transition:background .25s; cursor:pointer; }
  .gw2-cell.wall { background:var(--panel); cursor:default; }
  .gw2-cell.hl { outline:3px solid var(--accent); outline-offset:1px; z-index:2; }
  .gw2-cell.hl2 { outline:3px dashed var(--c-value); outline-offset:1px; }
  .gw2-cell .val { font-family:var(--mono); font-size:.78rem; font-weight:700; color:var(--text); }
  .gw2-cell .emo { font-size:1.5rem; }
  .gw2-cell .start { position:absolute; top:2px; left:4px; font-size:.75rem; }
  .gw2-cell .ar { position:absolute; font-size:.85rem; font-weight:900; color:var(--c-policy); line-height:1; }
  .gw2-cell .ar.u { top:3px; left:50%; transform:translateX(-50%); }
  .gw2-cell .ar.d { bottom:3px; left:50%; transform:translateX(-50%); }
  .gw2-cell .ar.l { left:4px; top:50%; transform:translateY(-50%); }
  .gw2-cell .ar.r { right:4px; top:50%; transform:translateY(-50%); }
  .gw2-cell .bot { position:absolute; font-size:1.5rem; z-index:3; }
  `;
  document.head.appendChild(style);

  const N = 5, START = "4,0", GEM = "0,4", LAVA = "1,3";
  const WALLS = new Set(["1,1", "2,1", "3,3"]);
  const DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]], ARROWS = "↑→↓←", ACTION_NAMES = ["up", "right", "down", "left"];
  const key = (r, c) => r + "," + c;
  const STATES = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (!WALLS.has(key(r, c))) STATES.push(key(r, c));
  const isTerm = k => k === GEM || k === LAVA;

  function move(k, a) {
    const [r, c] = k.split(",").map(Number);
    const nr = r + DIRS[a][0], nc = c + DIRS[a][1];
    return nr >= 0 && nr < N && nc >= 0 && nc < N && !WALLS.has(key(nr, nc)) ? key(nr, nc) : k;
  }

  function model({ slip = 0, step = -1, gem = 10, lava = -10 } = {}) {
    const P = {};
    STATES.forEach(k => {
      P[k] = [0, 1, 2, 3].map(a => {
        const out = {};
        for (let actual = 0; actual < 4; actual++) {
          const p = slip / 4 + (actual === a ? 1 - slip : 0);
          if (p <= 0) continue;
          const s2 = move(k, actual);
          out[s2] = (out[s2] || 0) + p;
        }
        return Object.entries(out).map(([s2, p]) => ({
          p, s2, r: s2 === GEM ? gem : s2 === LAVA ? lava : step, done: isTerm(s2),
        }));
      });
    });
    return P;
  }

  const zeros = () => Object.fromEntries(STATES.map(k => [k, 0]));
  const backup = (P, V, k, a, g) => P[k][a].reduce((s, o) => s + o.p * (o.r + g * (o.done ? 0 : V[o.s2])), 0);
  const q = (P, V, k, g) => [0, 1, 2, 3].map(a => backup(P, V, k, a, g));

  function evalSweep(P, V, pi, g) {
    const nv = {}; let delta = 0;
    STATES.forEach(k => {
      nv[k] = isTerm(k) ? 0 : pi[k].reduce((s, pa, a) => s + (pa ? pa * backup(P, V, k, a, g) : 0), 0);
      delta = Math.max(delta, Math.abs(nv[k] - V[k]));
    });
    return { V: nv, delta };
  }
  function viSweep(P, V, g) {
    const nv = {}; let delta = 0;
    STATES.forEach(k => {
      nv[k] = isTerm(k) ? 0 : Math.max(...q(P, V, k, g));
      delta = Math.max(delta, Math.abs(nv[k] - V[k]));
    });
    return { V: nv, delta };
  }
  function greedy(P, V, g, tol = 1e-6) {
    const pi = {};
    STATES.forEach(k => {
      if (isTerm(k)) { pi[k] = [0, 0, 0, 0]; return; }
      const qs = q(P, V, k, g), m = Math.max(...qs);
      const best = qs.map(v => (m - v < tol ? 1 : 0)), n = best.reduce((a, b) => a + b, 0);
      pi[k] = best.map(b => b / n);
    });
    return pi;
  }
  function evaluate(P, pi, g, theta = 1e-6, max = 5000) {
    let V = zeros(), sweeps = 0, d;
    do { ({ V, delta: d } = evalSweep(P, V, pi, g)); sweeps++; } while (d > theta && sweeps < max);
    return { V, sweeps };
  }
  function valueIteration(P, g, theta = 1e-6, max = 5000) {
    let V = zeros(), sweeps = 0, d;
    do { ({ V, delta: d } = viSweep(P, V, g)); sweeps++; } while (d > theta && sweeps < max);
    return { V, sweeps };
  }
  const randomPolicy = () => Object.fromEntries(STATES.map(k => [k, isTerm(k) ? [0, 0, 0, 0] : [0.25, 0.25, 0.25, 0.25]]));
  const detPolicy = (map, fallback = 0) => Object.fromEntries(STATES.map(k => [k, [0, 1, 2, 3].map(a => (isTerm(k) ? 0 : a === (map[k] ?? fallback) ? 1 : 0))]));

  function sample(P, k, a) {
    let u = Math.random();
    for (const o of P[k][a]) { if ((u -= o.p) <= 0) return o; }
    return P[k][a][P[k][a].length - 1];
  }
  function sampleAction(probs) {
    let u = Math.random();
    for (let a = 0; a < 4; a++) { if ((u -= probs[a]) <= 0) return a; }
    return 3;
  }
  // one Monte Carlo episode from state k (optionally forcing the first action)
  function rollout(P, pi, k, g, firstAction = null, maxT = 100) {
    let G = 0, disc = 1, s = k;
    for (let t = 0; t < maxT && !isTerm(s); t++) {
      const a = t === 0 && firstAction != null ? firstAction : sampleAction(pi[s]);
      const o = sample(P, s, a);
      G += disc * o.r; disc *= g; s = o.s2;
    }
    return G;
  }

  function heat(v, scale) {
    if (v == null || !isFinite(v)) return "";
    const a = Math.min(1, Math.abs(v) / scale) * 0.6;
    return v >= 0 ? `rgba(39,174,96,${a})` : `rgba(224,71,76,${a})`;
  }

  class Grid {
    constructor(el, opts = {}) {
      this.el = el; this.opts = opts;
      el.classList.add("gw2");
      if (opts.cell) el.style.setProperty("--cell", opts.cell + "px");
      this.cells = {};
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const k = key(r, c), d = document.createElement("div");
        d.className = "gw2-cell" + (WALLS.has(k) ? " wall" : "");
        d.dataset.k = k;
        if (!WALLS.has(k)) d.onclick = () => opts.onClick && opts.onClick(k);
        el.appendChild(d); this.cells[k] = d;
      }
    }
    draw({ V = null, pi = null, highlight = null, highlight2 = null, vscale = 10, showValues = true, bot = null, digits = 1 } = {}) {
      const hl = new Set([].concat(highlight || [])), hl2 = new Set([].concat(highlight2 || []));
      for (const [k, d] of Object.entries(this.cells)) {
        d.classList.toggle("hl", hl.has(k)); d.classList.toggle("hl2", hl2.has(k));
        if (WALLS.has(k)) { d.innerHTML = `<span class="emo">🧱</span>`; continue; }
        let html = "";
        if (k === GEM) html = `<span class="emo">💎</span>`;
        else if (k === LAVA) html = `<span class="emo">🔥</span>`;
        else {
          if (k === START) html += `<span class="start">🤖</span>`;
          if (V && showValues && V[k] != null) html += `<span class="val">${V[k] >= 0 ? "+" : ""}${V[k].toFixed(digits)}</span>`;
          if (pi && pi[k]) ["u", "r", "d", "l"].forEach((cls, a) => {
            if (pi[k][a] > 0.001) html += `<span class="ar ${cls}" style="opacity:${0.25 + 0.75 * pi[k][a]}">${ARROWS[a]}</span>`;
          });
        }
        if (bot === k) html += `<span class="bot">🤖</span>`;
        d.innerHTML = html;
        d.style.background = V && !isTerm(k) ? heat(V[k], vscale) : "";
      }
    }
  }

  window.GW = {
    N, START, GEM, LAVA, WALLS, STATES, ARROWS, ACTION_NAMES, key, isTerm, move, model, zeros, backup, q,
    evalSweep, viSweep, greedy, evaluate, valueIteration, randomPolicy, detPolicy, sample, sampleAction, rollout, Grid,
  };
})();

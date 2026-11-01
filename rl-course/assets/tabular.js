/* ============ Tabular RL helpers (Phase 3+) ============
 * Small environments the agent can only *sample* (no model access), tabular agents,
 * and a generic grid renderer.
 *   TAB.randomWalk(n)           Sutton & Barto's random walk (states 1..n, terminals 0 and n+1)
 *   TAB.cliff()                 4×12 Cliff Walking (S&B Example 6.6)
 *   TAB.gridworld({slip})       the Phase 0 gridworld, sample-only interface
 *   TAB.egreedy(qRow, eps)      ε-greedy action from one row of a Q-table
 *   new TAB.Grid(el, rows, cols, {cell})  .draw(fn(r,c) → {html, bg, cls})
 */
(function () {
  const style = document.createElement("style");
  style.textContent = `
  .tg { display:grid; gap:3px; user-select:none; }
  .tg-cell { position:relative; border-radius:7px; border:1px solid var(--line); background:var(--bg2);
    display:flex; align-items:center; justify-content:center; font-size:.72rem; font-family:var(--mono); font-weight:700; transition:background .2s; }
  .tg-cell.cliff { background:color-mix(in srgb, var(--bad) 30%, var(--bg2)); }
  .tg-cell.path { outline:2px solid var(--accent); outline-offset:-2px; }
  .tg-cell .ar { color:var(--c-policy); font-size:1rem; font-weight:900; }
  .tg-cell .emo { font-size:1.1rem; }
  `;
  document.head.appendChild(style);

  const ARROWS = "↑→↓←", DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]];

  function randomWalk(n = 5) {
    return {
      n, start: Math.ceil(n / 2), nS: n + 2,
      isTerm: s => s === 0 || s === n + 1,
      step(s) { const s2 = s + (Math.random() < 0.5 ? -1 : 1); return { s2, r: s2 === n + 1 ? 1 : 0, done: s2 === 0 || s2 === n + 1 }; },
      trueV: Array.from({ length: n + 2 }, (_, i) => (i === 0 || i === n + 1 ? 0 : i / (n + 1))),
    };
  }

  function cliff() {
    const R = 4, C = 12, start = [3, 0], goal = [3, 11];
    const id = (r, c) => r * C + c;
    return {
      R, C, nS: R * C, nA: 4, start: id(...start), goal: id(...goal), id,
      rc: s => [Math.floor(s / C), s % C],
      isCliff: (r, c) => r === 3 && c >= 1 && c <= 10,
      isTerm(s) { return s === this.goal; },
      step(s, a) {
        let [r, c] = this.rc(s);
        const nr = Math.min(R - 1, Math.max(0, r + DIRS[a][0])), nc = Math.min(C - 1, Math.max(0, c + DIRS[a][1]));
        if (this.isCliff(nr, nc)) return { s2: this.start, r: -100, done: false, fell: true };
        const s2 = id(nr, nc);
        return { s2, r: -1, done: s2 === this.goal };
      },
    };
  }

  // Phase 0 world, but the agent may only call step() (no access to P)
  function gridworld({ slip = 0 } = {}) {
    const P = GW.model({ slip });
    const S = GW.STATES, index = Object.fromEntries(S.map((k, i) => [k, i]));
    return {
      nS: S.length, nA: 4, keys: S, index, start: index[GW.START],
      isTerm: s => GW.isTerm(S[s]),
      step(s, a, t = 0) { const o = GW.sample(P, S[s], a); return { s2: index[o.s2], r: o.r, done: o.done }; },
    };
  }

  function argmaxRand(row) {
    let best = -Infinity, idx = [];
    row.forEach((v, i) => { if (v > best + 1e-12) { best = v; idx = [i]; } else if (Math.abs(v - best) <= 1e-12) idx.push(i); });
    return idx[Math.floor(Math.random() * idx.length)];
  }
  const egreedy = (row, eps) => (Math.random() < eps ? Math.floor(Math.random() * row.length) : argmaxRand(row));
  const zerosQ = (nS, nA) => Array.from({ length: nS }, () => new Array(nA).fill(0));

  class Grid {
    constructor(el, rows, cols, { cell = 34 } = {}) {
      this.el = el; this.rows = rows; this.cols = cols;
      el.classList.add("tg");
      el.style.gridTemplateColumns = `repeat(${cols}, ${cell}px)`;
      this.cells = [];
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const d = document.createElement("div");
        d.className = "tg-cell"; d.style.width = d.style.height = cell + "px";
        el.appendChild(d); this.cells.push(d);
      }
    }
    draw(fn) {
      for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
        const d = this.cells[r * this.cols + c], o = fn(r, c) || {};
        d.innerHTML = o.html || ""; d.style.background = o.bg || ""; d.className = "tg-cell " + (o.cls || "");
      }
    }
  }
  const heat = (v, scale) => {
    const a = Math.min(1, Math.abs(v) / scale) * 0.6;
    return v >= 0 ? `rgba(39,174,96,${a})` : `rgba(224,71,76,${a})`;
  };

  window.TAB = { ARROWS, DIRS, randomWalk, cliff, gridworld, argmaxRand, egreedy, zerosQ, Grid, heat };
})();

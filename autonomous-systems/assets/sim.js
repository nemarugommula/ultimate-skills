/* ============ Sim: a tiny 2D world for autonomy playgrounds (no dependencies) ============
 * const W = new Sim(canvas, { xRange: [0, 20], yRange: [0, 11.25], grid: 1, onPointer(type, [x, y], e) {} });
 *   World units are meters, x right, y UP. If yRange is omitted it's fitted to the canvas aspect ratio.
 *   Use <canvas class="sim"> (16:9), "sim square" or "sim tall" (4:3).
 * Draw inside a function you hand to W.render(fn): it runs now and again on resize / theme change.
 *   W.clear()                                    W.gridLines(step)
 *   W.rect(x, y, w, h, {fill, stroke, width, alpha, angle})    x, y = lower-left corner (or center with center:true)
 *   W.circle(x, y, r, {fill, stroke, width, alpha})            r in meters
 *   W.line([[x, y], ...], {color, width, dash, alpha, close, fill})
 *   W.arrow([x0, y0], [x1, y1], {color, width, head})          W.dot([x, y], {color, r (px), alpha})
 *   W.text([x, y], "label", {color, size, align, base, bold})
 *   W.ellipse([x, y], P, {k, color, fill, alpha})              P = 2×2 covariance; k = how many σ (default 2)
 *   W.car(pose, {color, L, width, steer, alpha})  pose = {x, y, th} (rear-axle center, heading in radians)
 *   W.robot(pose, {color, r})    W.drone(pose, {color, size})
 *   W.obstacles(list, {color})   list items: {type:"rect", x, y, w, h} | {type:"circle", x, y, r} | {type:"poly", pts}
 *   W.rays(pose, scan, {color, alpha, hits})                   draws a scan from Sim.lidar
 *   W.cells(grid, {color, empty, unknown})   grid = {x0, y0, res, nx, ny, data} with data[i + nx*j] in [0, 1] or null
 * Animation: W.run((dt, t) => { ...; return false to stop }, {dt: 1/30}); W.stop(); W.running.
 *   Each frame calls your step, then the render fn. Simulation time advances by dt per frame (fixed step).
 * Coordinates: W.X(x), W.Y(y) → pixels; W.toWorld(px, py) → [x, y]; W.px(meters) → pixels.
 * Static helpers (no canvas needed, also handy in node for checking numbers):
 *   Sim.rng(seed) → () => [0, 1)        Sim.randn(rng) → standard normal
 *   Sim.raycast(obstacles, [x, y], angle, maxRange, bounds) → distance (maxRange if nothing hit)
 *       bounds = [x0, y0, x1, y1] adds the world's outer walls
 *   Sim.lidar(obstacles, pose, {beams, fov, range, noise, rng, bounds}) → [{a, r, hit}] (a = world angle)
 *   Sim.collides(obstacles, [x, y], radius) → true if the disc touches any obstacle
 *   Sim.bicycle({x, y, th, v}, {accel, steer, L, dt, maxSteer}) → next state (kinematic bicycle, rear axle)
 *   Sim.unicycle({x, y, th}, {v, w, dt}) → next state
 *   Sim.covEllipse(P, k) → {a, b, angle} semi-axes in meters for k σ
 *   Sim.dist(a, b), Sim.wrap(angle)
 * Colors may be CSS variable names ("--c-se") or plain colors.
 */
(function () {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const col = c => (c && c.startsWith("--") ? css(c) : c || css("--text"));
  const sims = new Set();
  const reduced = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  class Sim {
    constructor(canvas, o) {
      this.c = canvas;
      this.o = Object.assign({ xRange: [0, 20], grid: 0 }, o || {});
      this.ctx = canvas.getContext("2d");
      this.renderFn = null; this.running = false; this._raf = 0; this.t = 0;
      sims.add(this);
      this.resize();
      const pos = e => { const r = canvas.getBoundingClientRect(); return this.toWorld(e.clientX - r.left, e.clientY - r.top); };
      if (this.o.onPointer) {
        let down = false;
        canvas.addEventListener("pointerdown", e => { down = true; canvas.setPointerCapture(e.pointerId); this.o.onPointer("down", pos(e), e); e.preventDefault(); });
        canvas.addEventListener("pointermove", e => this.o.onPointer(down ? "drag" : "move", pos(e), e));
        canvas.addEventListener("pointerup", e => { down = false; this.o.onPointer("up", pos(e), e); });
        canvas.addEventListener("pointercancel", () => { down = false; });
      }
    }
    resize() {
      const dpr = window.devicePixelRatio || 1;
      this.w = this.c.clientWidth || 640; this.h = this.c.clientHeight || 360;
      this.c.width = this.w * dpr; this.c.height = this.h * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const [x0, x1] = this.o.xRange;
      this.x0 = x0; this.x1 = x1;
      if (this.o.yRange) { this.y0 = this.o.yRange[0]; this.y1 = this.o.yRange[1]; }
      else { this.y0 = 0; this.y1 = (x1 - x0) * this.h / this.w; }
      this.sx = this.w / (this.x1 - this.x0); this.sy = this.h / (this.y1 - this.y0);
    }
    X(x) { return (x - this.x0) * this.sx; }
    Y(y) { return this.h - (y - this.y0) * this.sy; }
    px(m) { return m * this.sx; }
    toWorld(px, py) { return [this.x0 + px / this.sx, this.y0 + (this.h - py) / this.sy]; }
    render(fn) { if (fn) this.renderFn = fn; if (this.renderFn) this.renderFn(this); return this; }
    redraw() { this.resize(); this.render(); }

    // ---------- primitives ----------
    clear() {
      const g = this.ctx; g.clearRect(0, 0, this.w, this.h);
      if (this.o.grid) this.gridLines(this.o.grid);
      return this;
    }
    gridLines(step) {
      const g = this.ctx; g.save(); g.strokeStyle = css("--line"); g.lineWidth = 1; g.globalAlpha = 0.7;
      for (let x = Math.ceil(this.x0 / step) * step; x <= this.x1; x += step) { g.beginPath(); g.moveTo(this.X(x), 0); g.lineTo(this.X(x), this.h); g.stroke(); }
      for (let y = Math.ceil(this.y0 / step) * step; y <= this.y1; y += step) { g.beginPath(); g.moveTo(0, this.Y(y)); g.lineTo(this.w, this.Y(y)); g.stroke(); }
      g.restore(); return this;
    }
    _style(o, defStroke) {
      const g = this.ctx;
      g.globalAlpha = o.alpha ?? 1;
      g.lineWidth = o.width ?? 2;
      g.setLineDash(o.dash || []);
      g.strokeStyle = col(o.stroke || o.color || defStroke);
      if (o.fill) g.fillStyle = col(o.fill === true ? o.color : o.fill);
    }
    rect(x, y, w, h, o) {
      o = o || {}; const g = this.ctx; g.save(); this._style(o, "--muted");
      const cx = o.center ? x : x + w / 2, cy = o.center ? y : y + h / 2;
      g.translate(this.X(cx), this.Y(cy)); g.rotate(-(o.angle || 0));
      const pw = this.px(w), ph = h * this.sy;
      if (o.fill) g.fillRect(-pw / 2, -ph / 2, pw, ph);
      if (o.stroke || !o.fill) g.strokeRect(-pw / 2, -ph / 2, pw, ph);
      g.restore(); return this;
    }
    circle(x, y, r, o) {
      o = o || {}; const g = this.ctx; g.save(); this._style(o, "--muted");
      g.beginPath(); g.arc(this.X(x), this.Y(y), Math.max(0.5, this.px(r)), 0, 2 * Math.PI);
      if (o.fill) g.fill(); if (o.stroke || !o.fill) g.stroke();
      g.restore(); return this;
    }
    line(pts, o) {
      o = o || {}; if (!pts || pts.length < 2) return this;
      const g = this.ctx; g.save(); this._style(o, "--text"); g.lineJoin = "round"; g.lineCap = "round";
      g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(this.X(p[0]), this.Y(p[1])) : g.moveTo(this.X(p[0]), this.Y(p[1]))));
      if (o.close) g.closePath();
      if (o.fill) g.fill(); if (!o.fill || o.stroke) g.stroke();
      g.restore(); return this;
    }
    arrow(a, b, o) {
      o = o || {}; const g = this.ctx; g.save(); this._style(o, "--text");
      const x0 = this.X(a[0]), y0 = this.Y(a[1]), x1 = this.X(b[0]), y1 = this.Y(b[1]);
      const ang = Math.atan2(y1 - y0, x1 - x0), hd = o.head ?? 9;
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
      g.setLineDash([]); g.fillStyle = g.strokeStyle;
      g.beginPath(); g.moveTo(x1, y1);
      g.lineTo(x1 - hd * Math.cos(ang - 0.4), y1 - hd * Math.sin(ang - 0.4));
      g.lineTo(x1 - hd * Math.cos(ang + 0.4), y1 - hd * Math.sin(ang + 0.4)); g.closePath(); g.fill();
      g.restore(); return this;
    }
    dot(p, o) {
      o = o || {}; const g = this.ctx; g.save(); g.globalAlpha = o.alpha ?? 1; g.fillStyle = col(o.color || "--text");
      g.beginPath(); g.arc(this.X(p[0]), this.Y(p[1]), o.r ?? 3, 0, 2 * Math.PI); g.fill();
      if (o.ring) { g.strokeStyle = col(o.ring); g.lineWidth = 2; g.stroke(); }
      g.restore(); return this;
    }
    text(p, s, o) {
      o = o || {}; const g = this.ctx; g.save(); g.globalAlpha = o.alpha ?? 1; g.fillStyle = col(o.color || "--text");
      g.font = `${o.bold ? 700 : 500} ${o.size || 12}px ${css("--font")}`;
      g.textAlign = o.align || "center"; g.textBaseline = o.base || "middle";
      g.fillText(s, this.X(p[0]) + (o.dx || 0), this.Y(p[1]) + (o.dy || 0));
      g.restore(); return this;
    }
    ellipse(c, P, o) {
      o = o || {}; const e = Sim.covEllipse(P, o.k ?? 2), g = this.ctx;
      g.save(); this._style(Object.assign({ color: "--c-un", width: 2 }, o), "--c-un");
      g.beginPath(); g.ellipse(this.X(c[0]), this.Y(c[1]), Math.max(1, this.px(e.a)), Math.max(1, e.b * this.sy), -e.angle, 0, 2 * Math.PI);
      if (o.fill) { g.globalAlpha = o.fillAlpha ?? 0.15; g.fillStyle = col(o.fill === true ? (o.color || "--c-un") : o.fill); g.fill(); g.globalAlpha = o.alpha ?? 1; }
      g.stroke(); g.restore(); return this;
    }
    car(p, o) {
      o = o || {}; const L = o.L ?? 2.7, Wd = o.width ?? 1.8, g = this.ctx, c = col(o.color || "--c-be");
      g.save(); g.globalAlpha = o.alpha ?? 1;
      g.translate(this.X(p.x), this.Y(p.y)); g.rotate(-p.th);
      const s = this.sx, over = 0.25 * L;
      g.fillStyle = c; g.strokeStyle = css("--bg2"); g.lineWidth = 1.5;
      g.beginPath(); g.roundRect ? g.roundRect(-over * s, -Wd / 2 * s, (L + 2 * over) * s, Wd * s, 0.25 * Wd * s) : g.rect(-over * s, -Wd / 2 * s, (L + 2 * over) * s, Wd * s);
      g.globalAlpha = (o.alpha ?? 1) * 0.85; g.fill(); g.globalAlpha = o.alpha ?? 1; g.stroke();
      // windscreen + wheels (front wheels show the steering angle)
      g.fillStyle = css("--bg2"); g.globalAlpha = (o.alpha ?? 1) * 0.7;
      g.fillRect((L * 0.62) * s, -Wd * 0.38 * s, L * 0.16 * s, Wd * 0.76 * s);
      g.globalAlpha = o.alpha ?? 1; g.fillStyle = css("--text");
      const wl = 0.7 * s, ww = 0.25 * s;
      [[0, -Wd / 2], [0, Wd / 2]].forEach(([x, y]) => g.fillRect(x * s - wl / 2, y * s - ww / 2, wl, ww));
      [[L, -Wd / 2], [L, Wd / 2]].forEach(([x, y]) => { g.save(); g.translate(x * s, y * s); g.rotate(-(o.steer || 0)); g.fillRect(-wl / 2, -ww / 2, wl, ww); g.restore(); });
      g.restore(); return this;
    }
    robot(p, o) {
      o = o || {}; const r = o.r ?? 0.35;
      this.circle(p.x, p.y, r, { fill: o.color || "--c-be", alpha: o.alpha ?? 0.9 });
      this.line([[p.x, p.y], [p.x + r * 1.3 * Math.cos(p.th), p.y + r * 1.3 * Math.sin(p.th)]], { color: "--bg2", width: 2.5, alpha: o.alpha });
      return this;
    }
    drone(p, o) {
      o = o || {}; const s = o.size ?? 0.5, c = o.color || "--c-be", th = p.th || 0;
      const arm = a => [p.x + s * Math.cos(th + a), p.y + s * Math.sin(th + a)];
      [Math.PI / 4, 3 * Math.PI / 4, -Math.PI / 4, -3 * Math.PI / 4].forEach(a => {
        const q = arm(a); this.line([[p.x, p.y], q], { color: c, width: 3, alpha: o.alpha });
        this.circle(q[0], q[1], s * 0.42, { stroke: c, width: 1.5, alpha: (o.alpha ?? 1) * 0.8 });
      });
      this.circle(p.x, p.y, s * 0.28, { fill: c, alpha: o.alpha });
      return this;
    }
    obstacles(list, o) {
      o = o || {}; const c = o.color || "--muted";
      (list || []).forEach(ob => {
        if (ob.type === "circle") this.circle(ob.x, ob.y, ob.r, { fill: c, alpha: o.alpha ?? 0.55 });
        else if (ob.type === "poly") this.line(ob.pts, { fill: c, close: true, alpha: o.alpha ?? 0.55 });
        else this.rect(ob.x, ob.y, ob.w, ob.h, { fill: c, alpha: o.alpha ?? 0.55 });
      });
      return this;
    }
    rays(p, scan, o) {
      o = o || {};
      scan.forEach(b => {
        const e = [p.x + b.r * Math.cos(b.a), p.y + b.r * Math.sin(b.a)];
        this.line([[p.x, p.y], e], { color: o.color || "--c-se", width: 1, alpha: o.alpha ?? 0.25 });
        if (b.hit && o.hits !== false) this.dot(e, { color: o.color || "--c-se", r: 2.5 });
      });
      return this;
    }
    cells(gr, o) {
      o = o || {}; const g = this.ctx, full = col(o.color || "--text"), empty = o.empty ? col(o.empty) : null, unk = o.unknown ? col(o.unknown) : null;
      g.save();
      for (let j = 0; j < gr.ny; j++) for (let i = 0; i < gr.nx; i++) {
        const v = gr.data[i + gr.nx * j];
        const x = gr.x0 + i * gr.res, y = gr.y0 + j * gr.res;
        const X = this.X(x), Y = this.Y(y + gr.res), w = this.px(gr.res) + 0.5, h = gr.res * this.sy + 0.5;
        if (v == null) { if (unk) { g.globalAlpha = 0.35; g.fillStyle = unk; g.fillRect(X, Y, w, h); } continue; }
        if (empty && v < 0.5) { g.globalAlpha = (0.5 - v) * 0.6; g.fillStyle = empty; g.fillRect(X, Y, w, h); }
        g.globalAlpha = Math.max(0, Math.min(1, v)) * (o.alpha ?? 0.9); g.fillStyle = full; g.fillRect(X, Y, w, h);
      }
      g.restore(); return this;
    }

    // ---------- animation ----------
    run(step, o) {
      this.stop(); o = o || {};
      const dt = o.dt ?? 1 / 30; this.running = true;
      const frame = () => {
        if (!this.running) return;
        if (!this.c.isConnected) { this.running = false; return; }
        const keep = step(dt, (this.t += dt));
        this.render();
        if (keep === false) { this.running = false; if (o.onStop) o.onStop(); return; }
        this._raf = requestAnimationFrame(frame);
      };
      if (reduced() && o.instant) { while (this.running && step(dt, (this.t += dt)) !== false) { if (this.t > (o.maxT ?? 120)) break; } this.running = false; this.render(); return this; }
      this._raf = requestAnimationFrame(frame);
      return this;
    }
    stop() { this.running = false; cancelAnimationFrame(this._raf); return this; }
  }

  // ---------- static helpers ----------
  Sim.rng = seed => { let a = (seed ?? 1) >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  Sim.randn = rng => { const r = rng || Math.random; let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  Sim.dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  Sim.wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

  // distance along a ray (origin o, unit direction d) to a segment p→q, or Infinity
  function raySeg(o, d, p, q) {
    const ex = q[0] - p[0], ey = q[1] - p[1], den = d[0] * ey - d[1] * ex;
    if (Math.abs(den) < 1e-12) return Infinity;
    const wx = p[0] - o[0], wy = p[1] - o[1];
    const t = (wx * ey - wy * ex) / den, u = (wx * d[1] - wy * d[0]) / den;
    return t >= 0 && u >= 0 && u <= 1 ? t : Infinity;
  }
  function rayCircle(o, d, c, r) {
    const fx = o[0] - c[0], fy = o[1] - c[1], b = fx * d[0] + fy * d[1], cc = fx * fx + fy * fy - r * r, disc = b * b - cc;
    if (disc < 0) return Infinity;
    const s = Math.sqrt(disc), t1 = -b - s, t2 = -b + s;
    return t1 >= 0 ? t1 : t2 >= 0 ? (cc <= 0 ? 0 : t2) : Infinity;
  }
  const rectPts = ob => [[ob.x, ob.y], [ob.x + ob.w, ob.y], [ob.x + ob.w, ob.y + ob.h], [ob.x, ob.y + ob.h]];
  const polyHit = (o, d, pts) => { let m = Infinity; for (let i = 0; i < pts.length; i++) m = Math.min(m, raySeg(o, d, pts[i], pts[(i + 1) % pts.length])); return m; };

  Sim.raycast = (obs, o, ang, maxR, bounds) => {
    const d = [Math.cos(ang), Math.sin(ang)];
    let m = maxR ?? Infinity;
    (obs || []).forEach(ob => {
      const t = ob.type === "circle" ? rayCircle(o, d, [ob.x, ob.y], ob.r) : polyHit(o, d, ob.type === "poly" ? ob.pts : rectPts(ob));
      if (t < m) m = t;
    });
    if (bounds) { const [x0, y0, x1, y1] = bounds; m = Math.min(m, polyHit(o, d, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])); }
    return m;
  };
  Sim.lidar = (obs, p, o) => {
    o = Object.assign({ beams: 36, fov: 2 * Math.PI, range: 10, noise: 0, rng: Math.random }, o || {});
    const out = [], full = o.fov >= 2 * Math.PI - 1e-9;
    for (let i = 0; i < o.beams; i++) {
      const a = p.th + (full ? (i / o.beams) * 2 * Math.PI : -o.fov / 2 + (o.beams === 1 ? o.fov / 2 : (i / (o.beams - 1)) * o.fov));
      const t = Sim.raycast(obs, [p.x, p.y], a, o.range, o.bounds), hit = t < o.range;
      const r = hit ? Math.max(0, t + o.noise * Sim.randn(o.rng)) : o.range;
      out.push({ a, r, hit });
    }
    return out;
  };
  Sim.collides = (obs, p, rad) => (obs || []).some(ob => {
    const r = rad || 0;
    if (ob.type === "circle") return Math.hypot(p[0] - ob.x, p[1] - ob.y) <= ob.r + r;
    if (ob.type === "poly") { // point in polygon (radius ignored beyond the edges' distance check)
      let inside = false; const P = ob.pts;
      for (let i = 0, j = P.length - 1; i < P.length; j = i++) if ((P[i][1] > p[1]) !== (P[j][1] > p[1]) && p[0] < (P[j][0] - P[i][0]) * (p[1] - P[i][1]) / (P[j][1] - P[i][1]) + P[i][0]) inside = !inside;
      return inside;
    }
    const cx = Math.max(ob.x, Math.min(p[0], ob.x + ob.w)), cy = Math.max(ob.y, Math.min(p[1], ob.y + ob.h));
    return Math.hypot(p[0] - cx, p[1] - cy) <= r;
  });
  Sim.bicycle = (s, u) => { // kinematic bicycle model about the rear axle
    const L = u.L ?? 2.7, dt = u.dt ?? 0.05, ms = u.maxSteer ?? 0.6;
    const st = Math.max(-ms, Math.min(ms, u.steer || 0)), v = s.v ?? 0;
    return {
      x: s.x + v * Math.cos(s.th) * dt, y: s.y + v * Math.sin(s.th) * dt,
      th: Sim.wrap(s.th + (v / L) * Math.tan(st) * dt), v: v + (u.accel || 0) * dt, steer: st,
    };
  };
  Sim.unicycle = (s, u) => { const dt = u.dt ?? 0.05; return { x: s.x + u.v * Math.cos(s.th) * dt, y: s.y + u.v * Math.sin(s.th) * dt, th: Sim.wrap(s.th + u.w * dt) }; };
  Sim.covEllipse = (P, k) => { // eigen-decomposition of a symmetric 2×2 matrix
    const a = P[0][0], b = P[0][1], d = P[1][1], tr = (a + d) / 2, det = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
    const l1 = tr + det, l2 = Math.max(0, tr - det), ang = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
    return { a: (k ?? 2) * Math.sqrt(Math.max(0, l1)), b: (k ?? 2) * Math.sqrt(l2), angle: ang };
  };

  const redrawAll = () => sims.forEach(s => s.c.isConnected && s.redraw());
  window.addEventListener("resize", redrawAll);
  window.addEventListener("themechange", redrawAll);
  window.Sim = Sim;
})();

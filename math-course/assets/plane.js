/* ============ Plane: a 2D coordinate plane with draggable points (no dependencies) ============
 * const P = new Plane(canvas, { range: 5, onDrag(key, [x, y]) { ...; redraw(); }, snap: 0.5 });
 * P.draw([
 *   { type: "vec",  to: [2, 1], from: [0, 0], color, label, width, dash, drag: "a" },  // arrow
 *   { type: "pt",   at: [1, 2], color, r, label, drag: "p" },                          // dot
 *   { type: "seg",  from: [0, 0], to: [1, 1], color, width, dash },                    // plain line segment
 *   { type: "fn",   f: x => x * x, color, width, dash },                               // y = f(x) curve
 *   { type: "path", pts: [[x, y], ...], color, width, dash, fill, close },             // polyline / polygon
 *   { type: "text", at: [x, y], text, color, align },
 *   { type: "heat", f: (x, y) => value, lo, hi },                                      // background heatmap (e.g. a loss surface)
 * ]);
 * Items with `drag` get a grab handle; dragging calls onDrag(key, [x, y]) and the caller redraws.
 * Options: range (half-width shown, default 5), xRange/yRange ([lo, hi]) to override, grid (default true),
 * snap (round dragged coords), axes labels xLabel/yLabel. Re-draws on resize and theme change.
 * Colors may be CSS variable names ("--c-x") or plain colors.
 */
(function () {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const col = c => (c && c.startsWith("--") ? css(c) : c || css("--text"));
  const planes = new Set();

  class Plane {
    constructor(canvas, o) {
      this.c = canvas; this.o = Object.assign({ range: 5, grid: true }, o || {});
      this.items = []; this.handles = []; this.dragging = null;
      planes.add(this);
      const pos = e => {
        const r = canvas.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
      };
      canvas.addEventListener("pointerdown", e => {
        const [px, py] = pos(e);
        let best = null, bd = 22;
        this.handles.forEach(h => { const d = Math.hypot(h.px - px, h.py - py); if (d < bd) { bd = d; best = h; } });
        if (best) { this.dragging = best.key; canvas.setPointerCapture(e.pointerId); e.preventDefault(); }
      });
      canvas.addEventListener("pointermove", e => {
        const [px, py] = pos(e);
        if (!this.dragging) {
          const near = this.handles.some(h => Math.hypot(h.px - px, h.py - py) < 22);
          canvas.style.cursor = near ? "grab" : "crosshair";
          return;
        }
        let [x, y] = this.toWorld(px, py);
        const s = this.o.snap;
        if (s) { x = Math.round(x / s) * s; y = Math.round(y / s) * s; }
        const [x0, x1] = this.xr, [y0, y1] = this.yr;
        x = Math.max(x0, Math.min(x1, x)); y = Math.max(y0, Math.min(y1, y));
        if (this.o.onDrag) this.o.onDrag(this.dragging, [x, y]);
      });
      const end = () => { this.dragging = null; };
      canvas.addEventListener("pointerup", end);
      canvas.addEventListener("pointercancel", end);
    }
    get xr() { return this.o.xRange || [-this.o.range, this.o.range]; }
    get yr() { return this.o.yRange || [-this.o.range, this.o.range]; }
    toPx(x, y) {
      const [x0, x1] = this.xr, [y0, y1] = this.yr;
      return [((x - x0) / (x1 - x0)) * this.w, this.h - ((y - y0) / (y1 - y0)) * this.h];
    }
    toWorld(px, py) {
      const [x0, x1] = this.xr, [y0, y1] = this.yr;
      return [x0 + (px / this.w) * (x1 - x0), y0 + ((this.h - py) / this.h) * (y1 - y0)];
    }
    draw(items) {
      if (items) this.items = items;
      const c = this.c, dpr = window.devicePixelRatio || 1;
      this.w = c.clientWidth || 400; this.h = c.clientHeight || 400;
      c.width = this.w * dpr; c.height = this.h * dpr;
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      ctx.font = "13px " + css("--font");
      this.handles = [];
      const P = (x, y) => this.toPx(x, y);
      const [x0, x1] = this.xr, [y0, y1] = this.yr;

      // heatmaps first (backgrounds)
      this.items.filter(i => i.type === "heat").forEach(it => {
        const N = it.res || 60, cw = this.w / N, ch = this.h / N;
        let lo = it.lo, hi = it.hi;
        const vals = [];
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const v = it.f(x0 + (i + 0.5) / N * (x1 - x0), y1 - (j + 0.5) / N * (y1 - y0));
          vals.push(v);
        }
        if (lo == null) lo = Math.min(...vals);
        if (hi == null) hi = Math.max(...vals);
        const base = col(it.color || "--c-ls");
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const t = Math.max(0, Math.min(1, (vals[i * N + j] - lo) / (hi - lo || 1)));
          ctx.globalAlpha = 0.08 + 0.6 * (1 - t);
          ctx.fillStyle = base;
          ctx.fillRect(i * cw, j * ch, cw + 0.6, ch + 0.6);
        }
        ctx.globalAlpha = 1;
      });

      // grid + axes
      const step = this.o.step || niceStep(Math.max(x1 - x0, y1 - y0));
      if (this.o.grid) {
        ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
        for (let x = Math.ceil(x0 / step) * step; x <= x1 + 1e-9; x += step) { const [px] = P(x, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, this.h); ctx.stroke(); }
        for (let y = Math.ceil(y0 / step) * step; y <= y1 + 1e-9; y += step) { const [, py] = P(0, y); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(this.w, py); ctx.stroke(); }
      }
      ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.4;
      const [ox, oy] = P(0, 0);
      if (oy >= 0 && oy <= this.h) { ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(this.w, oy); ctx.stroke(); }
      if (ox >= 0 && ox <= this.w) { ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, this.h); ctx.stroke(); }
      ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--font");
      const lbl = v => (+v.toFixed(6)).toString();
      // tick labels (skipped right at the border, where they'd be cut off)
      for (let x = Math.ceil(x0 / step) * step; x <= x1 + 1e-9; x += step) { const [px] = P(x, 0); if (Math.abs(x) > 1e-9 && px > 10 && px < this.w - 10) { ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(lbl(x), px, Math.min(this.h - 14, Math.max(2, oy + 3))); } }
      for (let y = Math.ceil(y0 / step) * step; y <= y1 + 1e-9; y += step) { const [, py] = P(0, y); if (Math.abs(y) > 1e-9 && py > 10 && py < this.h - 10) { ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(lbl(y), Math.max(22, Math.min(this.w - 2, ox - 4)), py); } }
      if (this.o.xLabel) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(this.o.xLabel, this.w - 4, Math.min(this.h - 2, oy - 3)); }
      if (this.o.yLabel) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(this.o.yLabel, Math.max(2, ox + 5), 3); }
      ctx.font = "13px " + css("--font");

      this.items.forEach(it => {
        const color = col(it.color);
        ctx.save();
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = it.width || 2.5; ctx.lineCap = "round"; ctx.lineJoin = "round";
        if (it.dash) ctx.setLineDash(it.dash);
        if (it.alpha != null) ctx.globalAlpha = it.alpha;
        if (it.type === "fn") {
          ctx.beginPath();
          let pen = false;
          for (let i = 0; i <= 400; i++) {
            const x = x0 + (x1 - x0) * i / 400, y = it.f(x);
            if (!isFinite(y) || y < y0 - 10 * (y1 - y0) || y > y1 + 10 * (y1 - y0)) { pen = false; continue; }
            const [px, py] = P(x, y);
            pen ? ctx.lineTo(px, py) : ctx.moveTo(px, py); pen = true;
          }
          ctx.stroke();
        } else if (it.type === "seg") {
          const [a, b] = [P(...it.from), P(...it.to)];
          ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        } else if (it.type === "path") {
          ctx.beginPath();
          it.pts.forEach((p, i) => { const q = P(...p); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
          if (it.close || it.fill) ctx.closePath();
          if (it.fill) { ctx.globalAlpha = it.fillAlpha ?? 0.18; ctx.fillStyle = col(it.fill === true ? it.color : it.fill); ctx.fill(); ctx.globalAlpha = it.alpha ?? 1; }
          ctx.stroke();
        } else if (it.type === "vec") {
          const from = it.from || [0, 0];
          // `to` is the vector itself (drawn starting at `from`); with abs: true, `to` is the tip's position
          const tip = it.abs ? it.to : [from[0] + it.to[0], from[1] + it.to[1]];
          const [ax, ay] = P(...from), [bx, by] = P(...tip);
          const ang = Math.atan2(by - ay, bx - ax), len = Math.hypot(bx - ax, by - ay), hl = Math.min(13, len * 0.45);
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx - Math.cos(ang) * hl * 0.7, by - Math.sin(ang) * hl * 0.7); ctx.stroke();
          ctx.setLineDash([]);
          ctx.beginPath(); ctx.moveTo(bx, by);
          ctx.lineTo(bx - hl * Math.cos(ang - 0.42), by - hl * Math.sin(ang - 0.42));
          ctx.lineTo(bx - hl * Math.cos(ang + 0.42), by - hl * Math.sin(ang + 0.42));
          ctx.closePath(); ctx.fill();
          if (it.label) { ctx.font = "bold 14px " + css("--font"); ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(it.label, bx + 6, by - 4); }
          if (it.drag) { this.handles.push({ key: it.drag, px: bx, py: by }); ctx.globalAlpha = 0.22; ctx.beginPath(); ctx.arc(bx, by, 11, 0, 7); ctx.fill(); }
        } else if (it.type === "pt") {
          const [px, py] = P(...it.at);
          ctx.beginPath(); ctx.arc(px, py, it.r || 6, 0, 7); ctx.fill();
          if (it.label) { ctx.font = "bold 13px " + css("--font"); ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(it.label, px + 8, py - 4); }
          if (it.drag) { this.handles.push({ key: it.drag, px, py }); ctx.globalAlpha = 0.22; ctx.beginPath(); ctx.arc(px, py, 12, 0, 7); ctx.fill(); }
        } else if (it.type === "text") {
          const [px, py] = P(...it.at);
          ctx.font = (it.bold === false ? "" : "bold ") + (it.size || 13) + "px " + css("--font");
          ctx.textAlign = it.align || "left"; ctx.textBaseline = "middle"; ctx.fillText(it.text, px, py);
        }
        ctx.restore();
      });
      return this;
    }
  }
  function niceStep(span) {
    const raw = span / 10, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    return [1, 2, 5, 10].map(m => m * mag).find(s => span / s <= 12) || mag * 10;
  }
  const redrawAll = () => planes.forEach(p => p.c.isConnected && p.draw());
  window.addEventListener("resize", redrawAll);
  window.addEventListener("themechange", redrawAll);
  window.Plane = Plane;
})();

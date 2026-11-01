/* ============ Tiny canvas charts (no dependencies) ============
 * Plot.line(canvas, { series:[{label,color,data:[y]|[[x,y]],dash}], yMin, yMax, xLabel, yLabel, hlines:[{y,label,color}], vlines:[{x,label,color}],
 *                     xTicks:[values], xFmt: v => label, yFmt: v => label })
 *   Log axis: pass data as [Math.log10(x), y], xTicks: [0, 1, 2], xFmt: v => 10 ** v.
 * Plot.bars(canvas, { bars:[{label,value,color,marker,err}], yMin, yMax, yLabel, highlight })
 * Charts re-draw automatically on resize and theme change.
 */
(function () {
  const charts = new Set();
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600, h = canvas.clientHeight || 240;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "12px " + css("--font");
    return { ctx, w, h };
  }

  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmt = v => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : +v.toFixed(2) + "");

  function axes(ctx, w, h, m, x0, x1, y0, y1, o) {
    const muted = css("--muted"), line = css("--line");
    const X = x => m.l + ((x - x0) / (x1 - x0 || 1)) * (w - m.l - m.r);
    const Y = y => h - m.b - ((y - y0) / (y1 - y0 || 1)) * (h - m.t - m.b);
    ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.lineWidth = 1;
    niceTicks(y0, y1).forEach(t => {
      ctx.beginPath(); ctx.moveTo(m.l, Y(t)); ctx.lineTo(w - m.r, Y(t)); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(o.yFmt ? o.yFmt(t) : fmt(t), m.l - 6, Y(t));
    });
    if (!o.noXTicks) (o.xTicks || niceTicks(x0, x1, 6)).forEach(t => {
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(o.xFmt ? o.xFmt(t) : fmt(t), X(t), h - m.b + 5);
    });
    if (o.xLabel) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(o.xLabel, w - m.r, h - m.b - 4); }
    if (o.yLabel) { ctx.save(); ctx.translate(12, m.t + (h - m.t - m.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
    return { X, Y };
  }

  function legend(ctx, w, m, items, left) {
    let y = m.t + 4;
    ctx.textBaseline = "top";
    items.filter(i => i.label).forEach(i => {
      const tw = ctx.measureText(i.label).width;
      const x = left ? m.l + 26 : w - m.r - 6 - tw;
      ctx.fillStyle = color(css("--bg2"), 0.75); ctx.fillRect(x - 22, y - 1, tw + 26, 16);
      ctx.textAlign = "left"; ctx.fillStyle = css("--text"); ctx.fillText(i.label, x, y);
      ctx.fillStyle = i.color; ctx.fillRect(x - 18, y + 5, 12, 4);
      y += 16;
    });
  }
  const color = (c, a) => { ctx2.fillStyle = c; const v = ctx2.fillStyle; if (v[0] !== "#") return v; const n = parseInt(v.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
  const ctx2 = document.createElement("canvas").getContext("2d");

  function drawLine(canvas, o) {
    const { ctx, w, h } = setup(canvas);
    const m = { l: o.yLabel ? 52 : 42, r: 12, t: 12, b: 26 };
    const pts = o.series.map(s => s.data.map((d, i) => (Array.isArray(d) ? d : [i + (o.xStart ?? 1), d])));
    const all = pts.flat().filter(p => isFinite(p[1]));
    if (!all.length) { axes(ctx, w, h, m, 0, 1, o.yMin ?? 0, o.yMax ?? 1, o); return; }
    const x0 = o.xMin ?? Math.min(...all.map(p => p[0])), x1 = o.xMax ?? Math.max(...all.map(p => p[0]));
    let y0 = o.yMin ?? Math.min(...all.map(p => p[1])), y1 = o.yMax ?? Math.max(...all.map(p => p[1]));
    (o.hlines || []).forEach(l => { if (o.yMin == null) y0 = Math.min(y0, l.y); if (o.yMax == null) y1 = Math.max(y1, l.y); });
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    const { X, Y } = axes(ctx, w, h, m, x0, x1, y0, y1, o);
    (o.vlines || []).forEach(l => {
      ctx.save(); ctx.setLineDash([6, 5]); ctx.strokeStyle = l.color || css("--muted"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(l.x), m.t); ctx.lineTo(X(l.x), h - m.b); ctx.stroke(); ctx.restore();
      if (l.label) { ctx.fillStyle = l.color || css("--muted"); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(l.label, X(l.x) + 4, m.t + 2); }
    });
    (o.hlines || []).forEach(l => {
      ctx.save(); ctx.setLineDash([6, 5]); ctx.strokeStyle = l.color || css("--muted"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(m.l, Y(l.y)); ctx.lineTo(w - m.r, Y(l.y)); ctx.stroke(); ctx.restore();
      if (l.label) { ctx.fillStyle = l.color || css("--muted"); ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(l.label, m.l + 6, Y(l.y) - 3); }
    });
    o.series.forEach((s, si) => {
      const p = pts[si];
      ctx.save();
      ctx.beginPath(); ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b); ctx.clip();
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2.2; ctx.lineJoin = "round";
      if (s.dash) ctx.setLineDash(s.dash);
      if (s.fill) { ctx.fillStyle = s.color + "33"; ctx.beginPath(); ctx.moveTo(X(p[0][0]), Y(y0)); p.forEach(q => ctx.lineTo(X(q[0]), Y(q[1]))); ctx.lineTo(X(p[p.length - 1][0]), Y(y0)); ctx.fill(); }
      if (!s.dotsOnly) {
        ctx.beginPath();
        p.forEach((q, i) => (i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1]))));
        ctx.stroke();
      }
      if (s.dots || s.dotsOnly) {
        ctx.fillStyle = s.color; ctx.globalAlpha = s.dotsOnly ? 0.35 : 1;
        p.forEach(q => { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), s.dotsOnly ? 2.2 : 3, 0, 7); ctx.fill(); });
      }
      ctx.restore();
    });
    legend(ctx, w, m, o.series, o.legendLeft);
  }

  function drawBars(canvas, o) {
    const { ctx, w, h } = setup(canvas);
    const m = { l: o.yLabel ? 52 : 42, r: 12, t: 14, b: 30 };
    const n = o.bars.length;
    const vals = o.bars.flatMap(b => [b.value, b.marker, b.err != null ? b.value + b.err : undefined, b.err != null ? b.value - b.err : undefined]).filter(v => v != null && isFinite(v));
    const y0 = o.yMin ?? Math.min(0, ...vals), y1 = o.yMax ?? Math.max(1, ...vals);
    const { Y } = axes(ctx, w, h, m, 0, 1, y0, y1, { ...o, noXTicks: true });
    const slot = (w - m.l - m.r) / n, bw = Math.min(60, slot * 0.6);
    o.bars.forEach((b, i) => {
      const cx = m.l + slot * (i + 0.5), v = Math.max(y0, Math.min(y1, b.value));
      ctx.fillStyle = b.color || css("--accent");
      ctx.globalAlpha = o.highlight == null || o.highlight === i ? 1 : 0.55;
      ctx.fillRect(cx - bw / 2, Math.min(Y(v), Y(Math.max(0, y0))), bw, Math.abs(Y(v) - Y(Math.max(0, y0))) || 1);
      ctx.globalAlpha = 1;
      if (b.err != null) {
        const hi = Math.min(y1, b.value + b.err), lo = Math.max(y0, b.value - b.err);
        ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, Y(lo)); ctx.lineTo(cx, Y(hi)); ctx.moveTo(cx - 8, Y(hi)); ctx.lineTo(cx + 8, Y(hi)); ctx.moveTo(cx - 8, Y(lo)); ctx.lineTo(cx + 8, Y(lo)); ctx.stroke();
      }
      if (b.marker != null && b.marker >= y0 && b.marker <= y1) {
        ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = css("--text"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cx - bw / 2 - 6, Y(b.marker)); ctx.lineTo(cx + bw / 2 + 6, Y(b.marker)); ctx.stroke(); ctx.restore();
      }
      ctx.fillStyle = css("--muted"); ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(b.label ?? i + 1, cx, h - m.b + 6);
    });
  }

  function register(canvas, kind, o) {
    canvas._plot = { kind, o };
    charts.add(canvas);
    (kind === "line" ? drawLine : drawBars)(canvas, o);
  }
  const redrawAll = () => charts.forEach(c => c.isConnected && (c._plot.kind === "line" ? drawLine : drawBars)(c, c._plot.o));
  window.addEventListener("resize", redrawAll);
  window.addEventListener("themechange", redrawAll);

  window.Plot = {
    line: (c, o) => register(c, "line", o),
    bars: (c, o) => register(c, "bars", o),
    /* moving average helper for noisy curves */
    smooth(arr, k = 20) {
      const out = []; let s = 0;
      arr.forEach((v, i) => { s += v; if (i >= k) s -= arr[i - k]; out.push(s / Math.min(i + 1, k)); });
      return out;
    },
  };
})();

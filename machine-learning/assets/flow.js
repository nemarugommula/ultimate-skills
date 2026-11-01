/* ============ Flow: animated architecture diagrams (no dependencies) ============
 * const F = new Flow(svgElement, { width: 720, height: 300 });
 * F.node("lb", { x: 200, y: 150, w: 110, h: 46, label: "Features", sub: "12 columns", kind: "ft" });
 * F.edge("client", "lb", { dashed: false, label: "HTTPS" });
 * F.render();                                   // draw once after adding nodes/edges
 * F.set("api1", { sub: "120 rps", down: false, hot: true, load: 0.8, label: "API 1" });  // update a node
 * F.edgeStyle("lb", "api1", { color: "--c-lo", width: 3, dashed: true });                // update an edge
 * await F.packet(["client", "lb", "api1", "db"], { color: "--c-ft", dur: 900 });       // animate a request
 * F.packet([...], { back: true })  // travels the path, then returns to the start (a response)
 * kind picks the color: ft data/features, mo model, pd prediction, tg target/label, lo error/loss, mt metric, bd threshold/knob.
 * x, y are the node's CENTER. Coordinates are in the SVG's own units (viewBox), so it scales to any width.
 * Honors prefers-reduced-motion (packets jump instantly). Colors re-read on theme change.
 */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const col = c => (c && c.startsWith("--") ? css(c) : c || css("--text"));
  const KIND = { ft: "--c-ft", mo: "--c-mo", pd: "--c-pd", tg: "--c-tg", lo: "--c-lo", mt: "--c-mt", bd: "--c-bd" };
  const reduced = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const flows = new Set();
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (parent) parent.appendChild(e);
    return e;
  };

  class Flow {
    constructor(svg, o) {
      this.svg = svg; this.o = Object.assign({ width: 720, height: 300 }, o || {});
      svg.setAttribute("viewBox", `0 0 ${this.o.width} ${this.o.height}`);
      svg.classList.add("flow");
      this.nodes = {}; this.edges = [];
      flows.add(this);
    }
    node(id, o) { this.nodes[id] = Object.assign({ w: 110, h: 46, kind: "mo", label: id }, o); return this; }
    edge(a, b, o) { this.edges.push(Object.assign({ a, b }, o || {})); return this; }
    // point where the line from node a's center toward (tx, ty) leaves a's box
    exit(n, tx, ty) {
      const dx = tx - n.x, dy = ty - n.y;
      if (!dx && !dy) return [n.x, n.y];
      const sx = Math.abs(dx) / (n.w / 2), sy = Math.abs(dy) / (n.h / 2), s = Math.max(sx, sy);
      return [n.x + dx / s, n.y + dy / s];
    }
    ends(a, b) { const A = this.nodes[a], B = this.nodes[b]; return [this.exit(A, B.x, B.y), this.exit(B, A.x, A.y)]; }
    render() {
      const svg = this.svg;
      svg.innerHTML = "";
      const defs = el("defs", {}, svg);
      const mk = el("marker", { id: (this.mid = "ah" + Math.random().toString(36).slice(2, 8)), viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0L10,5L0,10z", fill: css("--muted") }, mk);
      this.gEdges = el("g", {}, svg); this.gNodes = el("g", {}, svg); this.gPk = el("g", {}, svg);
      this.edges.forEach(e => {
        const [[x1, y1], [x2, y2]] = this.ends(e.a, e.b);
        e.line = el("line", { x1, y1, x2, y2, class: "edge" }, this.gEdges);
        if (!e.noArrow) e.line.setAttribute("marker-end", `url(#${this.mid})`);
        if (e.label) e.text = el("text", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 6, "text-anchor": "middle", "font-size": 11, fill: css("--muted") }, this.gEdges);
        if (e.text) e.text.textContent = e.label;
        this.paintEdge(e);
      });
      Object.entries(this.nodes).forEach(([id, n]) => {
        const g = el("g", { class: "node", transform: `translate(${n.x - n.w / 2},${n.y - n.h / 2})` }, this.gNodes);
        n.g = g;
        n.rect = el("rect", { width: n.w, height: n.h, rx: 10 }, g);
        n.bar = el("rect", { x: 6, y: n.h - 7, height: 3, width: 0, rx: 1.5 }, g);
        n.t1 = el("text", { x: n.w / 2, y: n.sub ? n.h / 2 - 7 : n.h / 2 }, g);
        n.t2 = el("text", { x: n.w / 2, y: n.h / 2 + 9, class: "sub" }, g);
        this.paintNode(n);
      });
      return this;
    }
    paintNode(n) {
      const c = col(n.down ? "--c-lo" : KIND[n.kind] || n.kind);
      n.rect.setAttribute("stroke", c);
      n.g.classList.toggle("down", !!n.down);
      n.g.classList.toggle("hot", !!n.hot);
      n.t1.textContent = n.label;
      n.t1.setAttribute("y", n.sub ? n.h / 2 - 7 : n.h / 2);
      n.t2.textContent = n.sub || "";
      const load = Math.max(0, Math.min(1, n.load || 0));
      n.bar.setAttribute("width", (n.w - 12) * load);
      n.bar.setAttribute("fill", col(load > 0.85 ? "--c-lo" : load > 0.6 ? "--c-bd" : "--c-tg"));
    }
    paintEdge(e) { // inline styles, so they win over the .edge rule in style.css
      e.line.style.stroke = col(e.color || "--line");
      e.line.style.strokeWidth = e.width || 2;
      e.line.style.strokeDasharray = e.dashed ? "6 5" : "";
      e.line.style.opacity = e.hidden ? 0 : 1;
    }
    set(id, o) { const n = this.nodes[id]; if (!n) return this; Object.assign(n, o); if (n.rect) this.paintNode(n); return this; }
    edgeStyle(a, b, o) {
      const e = this.edges.find(x => (x.a === a && x.b === b) || (x.a === b && x.b === a));
      if (e) { Object.assign(e, o); if (e.line) this.paintEdge(e); if (e.text && o.label != null) e.text.textContent = o.label; }
      return this;
    }
    // animate a dot along node centers; resolves when it arrives (and comes back, with back: true)
    packet(path, o) {
      o = Object.assign({ color: "--c-ft", dur: 900, r: 6 }, o || {});
      const pts = [];
      for (let i = 0; i < path.length - 1; i++) {
        const [p, q] = this.ends(path[i], path[i + 1]);
        pts.push(p, q);
      }
      if (o.back) pts.push(...pts.slice().reverse());
      const dot = el("circle", { r: o.r, fill: col(o.color) }, this.gPk);
      const segs = []; let total = 0;
      for (let i = 0; i < pts.length - 1; i++) { const d = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]); segs.push(d); total += d; }
      const dur = reduced() ? 0 : o.dur * (o.back ? 2 : 1);
      return new Promise(res => {
        const t0 = performance.now();
        const step = now => {
          const f = dur ? Math.min(1, (now - t0) / dur) : 1;
          let d = f * total, i = 0;
          while (i < segs.length - 1 && d > segs[i]) { d -= segs[i]; i++; }
          const k = segs[i] ? d / segs[i] : 1, a = pts[i], b = pts[i + 1] || a;
          dot.setAttribute("cx", a[0] + (b[0] - a[0]) * k);
          dot.setAttribute("cy", a[1] + (b[1] - a[1]) * k);
          if (f < 1) requestAnimationFrame(step);
          else { if (!o.keep) dot.remove(); res(dot); }
        };
        requestAnimationFrame(step);
      });
    }
    clearPackets() { if (this.gPk) this.gPk.innerHTML = ""; }
  }
  window.addEventListener("themechange", () => flows.forEach(f => {
    if (!f.svg.isConnected || !f.gNodes) return;
    Object.values(f.nodes).forEach(n => f.paintNode(n));
    f.edges.forEach(e => f.paintEdge(e));
  }));
  window.Flow = Flow;
})();

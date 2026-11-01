/* ============ Space: 3D rotations, quaternions, transforms and a tiny orbit view (no dependencies) ============
 * Robotics convention: right-handed, x forward, y left, z up. Angles in radians. Matrices are arrays of rows.
 * SP.rotX(a) SP.rotY(a) SP.rotZ(a) → 3×3       SP.mul(A, B)   SP.apply(R, v)   SP.T(R) (transpose = inverse for rotations)
 * SP.euler(yaw, pitch, roll) → R = Rz(yaw)·Ry(pitch)·Rx(roll) (ZYX, as in aerospace / ROS RPY)
 * SP.toEuler(R) → {yaw, pitch, roll}            (ill-defined at pitch = ±90°: gimbal lock)
 * SP.q.*  quaternions as [w, x, y, z]: fromAxisAngle(axis, a), fromEuler(yaw, pitch, roll), mul(a, b), conj(q),
 *         normalize(q), toMatrix(q), fromMatrix(R), rotate(q, v), slerp(a, b, t), angle(a, b) (radians between them)
 * SP.hom(R, t) → 4×4   SP.hmul(A, B)   SP.happly(T, p)   SP.hinv(T)       (homogeneous transforms)
 * SP.project(K, p) → [u, v] pixel of a camera-frame point (camera z forward, x right, y down; K = 3×3 intrinsics)
 *
 * const V = new View3D(canvas, { yaw: 0.8, pitch: 0.45, dist: 8, target: [0, 0, 0], fov: 0.9, onChange });
 *   Drag to orbit (unless {fixed: true}); V.render(fn) runs fn(V) now and after every orbit / resize / theme change.
 *   Inside fn: V.clear(); V.grid(size, step); V.line([[x,y,z],...], {color, width, dash, alpha});
 *   V.arrow(a, b, {color, width, label}); V.point(p, {color, r, label}); V.poly(pts, {fill, stroke, alpha});
 *   V.text(p, "s", {color, size}); V.axes(T or {R, t}, {len, width, labels: ["x","y","z"], alpha});
 *   V.box(center, [lx, ly, lz], R, {color, alpha}); V.toScreen(p) → [px, py, depth] or null if behind the eye.
 * Use <canvas class="view3d">. Colors may be CSS variable names ("--c-se").
 */
(function () {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const col = c => (c && c.startsWith("--") ? css(c) : c || css("--text"));
  const c = Math.cos, s = Math.sin;

  const SP = {
    rotX: a => [[1, 0, 0], [0, c(a), -s(a)], [0, s(a), c(a)]],
    rotY: a => [[c(a), 0, s(a)], [0, 1, 0], [-s(a), 0, c(a)]],
    rotZ: a => [[c(a), -s(a), 0], [s(a), c(a), 0], [0, 0, 1]],
    mul: (A, B) => A.map(r => B[0].map((_, j) => r.reduce((t, v, k) => t + v * B[k][j], 0))),
    apply: (R, v) => R.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]),
    T: R => R[0].map((_, j) => R.map(r => r[j])),
    euler(yaw, pitch, roll) { return this.mul(this.rotZ(yaw), this.mul(this.rotY(pitch), this.rotX(roll))); },
    toEuler(R) {
      const sp = Math.max(-1, Math.min(1, -R[2][0])), pitch = Math.asin(sp);
      if (Math.abs(sp) > 0.999999) return { yaw: Math.atan2(-R[0][1], R[1][1]), pitch, roll: 0, locked: true };
      return { yaw: Math.atan2(R[1][0], R[0][0]), pitch, roll: Math.atan2(R[2][1], R[2][2]), locked: false };
    },
    hom: (R, t) => [[...R[0], t[0]], [...R[1], t[1]], [...R[2], t[2]], [0, 0, 0, 1]],
    hmul: (A, B) => A.map(r => B[0].map((_, j) => r.reduce((t, v, k) => t + v * B[k][j], 0))),
    happly: (T, p) => [0, 1, 2].map(i => T[i][0] * p[0] + T[i][1] * p[1] + T[i][2] * p[2] + T[i][3]),
    hinv(T) { const R = [T[0].slice(0, 3), T[1].slice(0, 3), T[2].slice(0, 3)], Rt = this.T(R), t = [T[0][3], T[1][3], T[2][3]], ti = this.apply(Rt, t).map(v => -v); return this.hom(Rt, ti); },
    project: (K, p) => (p[2] <= 1e-9 ? null : [K[0][0] * p[0] / p[2] + K[0][2], K[1][1] * p[1] / p[2] + K[1][2]]),
    q: {
      fromAxisAngle(ax, a) { const n = Math.hypot(...ax) || 1, h = a / 2; return [c(h), s(h) * ax[0] / n, s(h) * ax[1] / n, s(h) * ax[2] / n]; },
      fromEuler(yaw, pitch, roll) { return this.mul(this.fromAxisAngle([0, 0, 1], yaw), this.mul(this.fromAxisAngle([0, 1, 0], pitch), this.fromAxisAngle([1, 0, 0], roll))); },
      mul: (a, b) => [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]],
      conj: q => [q[0], -q[1], -q[2], -q[3]],
      normalize: q => { const n = Math.hypot(...q) || 1; return q.map(v => v / n); },
      toMatrix([w, x, y, z]) {
        return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
      },
      fromMatrix(R) {
        const tr = R[0][0] + R[1][1] + R[2][2]; let q;
        if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; q = [S / 4, (R[2][1] - R[1][2]) / S, (R[0][2] - R[2][0]) / S, (R[1][0] - R[0][1]) / S]; }
        else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) { const S = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2; q = [(R[2][1] - R[1][2]) / S, S / 4, (R[0][1] + R[1][0]) / S, (R[0][2] + R[2][0]) / S]; }
        else if (R[1][1] > R[2][2]) { const S = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2; q = [(R[0][2] - R[2][0]) / S, (R[0][1] + R[1][0]) / S, S / 4, (R[1][2] + R[2][1]) / S]; }
        else { const S = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2; q = [(R[1][0] - R[0][1]) / S, (R[0][2] + R[2][0]) / S, (R[1][2] + R[2][1]) / S, S / 4]; }
        return this.normalize(q);
      },
      rotate(q, v) { const r = this.mul(this.mul(q, [0, ...v]), this.conj(q)); return [r[1], r[2], r[3]]; },
      slerp(a, b, t) {
        let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        if (d < 0) { b = b.map(v => -v); d = -d; }              // take the short way round
        if (d > 0.9995) return this.normalize(a.map((v, i) => v + t * (b[i] - v)));
        const th = Math.acos(d), sa = s((1 - t) * th) / s(th), sb = s(t * th) / s(th);
        return a.map((v, i) => sa * v + sb * b[i]);
      },
      angle(a, b) { const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]); return 2 * Math.acos(Math.min(1, d)); },
    },
  };

  const views = new Set();
  class View3D {
    constructor(canvas, o) {
      this.c = canvas; this.g = canvas.getContext("2d");
      this.o = Object.assign({ yaw: 0.8, pitch: 0.45, dist: 8, target: [0, 0, 0], fov: 0.9 }, o || {});
      this.fn = null; views.add(this); this.resize();
      if (!this.o.fixed) {
        let last = null;
        canvas.addEventListener("pointerdown", e => { last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; });
        canvas.addEventListener("pointermove", e => {
          if (!last) return;
          this.o.yaw -= (e.clientX - last[0]) * 0.01;
          this.o.pitch = Math.max(-1.4, Math.min(1.4, this.o.pitch + (e.clientY - last[1]) * 0.01));
          last = [e.clientX, e.clientY]; this.render(); if (this.o.onChange) this.o.onChange(this);
        });
        const up = () => { last = null; canvas.style.cursor = ""; };
        canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
      }
    }
    resize() {
      const dpr = window.devicePixelRatio || 1;
      this.w = this.c.clientWidth || 480; this.h = this.c.clientHeight || 360;
      this.c.width = this.w * dpr; this.c.height = this.h * dpr; this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    render(fn) { if (fn) this.fn = fn; if (this.fn) this.fn(this); return this; }
    // world point → [px, py, depth]
    toScreen(p) {
      const { yaw, pitch, dist, target, fov } = this.o;
      const eye = [target[0] + dist * c(pitch) * c(yaw), target[1] + dist * c(pitch) * s(yaw), target[2] + dist * s(pitch)];
      const f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]].map(v => v / dist);
      let r = [f[1], -f[0], 0]; const rn = Math.hypot(r[0], r[1]) || 1; r = r.map(v => v / rn);
      const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
      const d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
      const z = d[0] * f[0] + d[1] * f[1] + d[2] * f[2]; if (z < 0.05) return null;
      const k = (Math.min(this.w, this.h) / 2) / Math.tan(fov / 2);
      return [this.w / 2 + k * (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / z, this.h / 2 - k * (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) / z, z];
    }
    clear() { this.g.clearRect(0, 0, this.w, this.h); return this; }
    line(pts, o) {
      o = o || {}; const g = this.g, P = pts.map(p => this.toScreen(p));
      g.save(); g.strokeStyle = col(o.color || "--muted"); g.lineWidth = o.width ?? 2; g.globalAlpha = o.alpha ?? 1; g.setLineDash(o.dash || []);
      g.lineJoin = "round"; g.beginPath(); let pen = false;
      P.forEach(q => { if (!q) { pen = false; return; } pen ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); pen = true; });
      g.stroke(); g.restore(); return this;
    }
    arrow(a, b, o) {
      o = o || {}; this.line([a, b], o);
      const A = this.toScreen(a), B = this.toScreen(b); if (!A || !B) return this;
      const g = this.g, ang = Math.atan2(B[1] - A[1], B[0] - A[0]), hd = o.head ?? 9;
      g.save(); g.fillStyle = col(o.color || "--muted"); g.globalAlpha = o.alpha ?? 1;
      g.beginPath(); g.moveTo(B[0], B[1]); g.lineTo(B[0] - hd * c(ang - 0.4), B[1] - hd * s(ang - 0.4)); g.lineTo(B[0] - hd * c(ang + 0.4), B[1] - hd * s(ang + 0.4)); g.fill();
      if (o.label) { g.font = `700 13px ${css("--font")}`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(o.label, B[0] + 12 * c(ang), B[1] + 12 * s(ang)); }
      g.restore(); return this;
    }
    point(p, o) {
      o = o || {}; const q = this.toScreen(p); if (!q) return this; const g = this.g;
      g.save(); g.fillStyle = col(o.color || "--text"); g.globalAlpha = o.alpha ?? 1;
      g.beginPath(); g.arc(q[0], q[1], o.r ?? 3.5, 0, 2 * Math.PI); g.fill();
      if (o.label) { g.font = `600 12px ${css("--font")}`; g.textAlign = "left"; g.textBaseline = "bottom"; g.fillText(o.label, q[0] + 6, q[1] - 4); }
      g.restore(); return this;
    }
    poly(pts, o) {
      o = o || {}; const P = pts.map(p => this.toScreen(p)); if (P.some(q => !q)) return this; const g = this.g;
      g.save(); g.globalAlpha = o.alpha ?? 0.25; g.fillStyle = col(o.fill || o.color || "--accent");
      g.beginPath(); P.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.closePath(); g.fill();
      if (o.stroke) { g.globalAlpha = 1; g.strokeStyle = col(o.stroke); g.lineWidth = o.width ?? 1.5; g.stroke(); }
      g.restore(); return this;
    }
    text(p, str, o) {
      o = o || {}; const q = this.toScreen(p); if (!q) return this; const g = this.g;
      g.save(); g.fillStyle = col(o.color || "--text"); g.font = `${o.bold ? 700 : 500} ${o.size || 12}px ${css("--font")}`;
      g.textAlign = o.align || "center"; g.textBaseline = "middle"; g.fillText(str, q[0], q[1]); g.restore(); return this;
    }
    grid(size, step) {
      const n = size ?? 5, st = step ?? 1;
      for (let v = -n; v <= n + 1e-9; v += st) { this.line([[v, -n, 0], [v, n, 0]], { color: "--line", width: 1 }); this.line([[-n, v, 0], [n, v, 0]], { color: "--line", width: 1 }); }
      return this;
    }
    axes(T, o) {
      o = o || {}; const len = o.len ?? 1, R = T.R || [T[0].slice(0, 3), T[1].slice(0, 3), T[2].slice(0, 3)], t = T.t || [T[0][3], T[1][3], T[2][3]];
      const labels = o.labels || ["x", "y", "z"], cols = o.colors || ["#e0474c", "#27ae60", "#2f80ed"];
      [0, 1, 2].forEach(i => this.arrow(t, [t[0] + len * R[0][i], t[1] + len * R[1][i], t[2] + len * R[2][i]], { color: cols[i], width: o.width ?? 3, label: labels[i], alpha: o.alpha }));
      if (o.name) this.text([t[0], t[1], t[2] - 0.25 * len], o.name, { color: o.nameColor || "--text", bold: true });
      return this;
    }
    box(ctr, L, R, o) {
      o = o || {}; R = R || SP.rotZ(0);
      const corners = [];
      [-1, 1].forEach(a => [-1, 1].forEach(b => [-1, 1].forEach(d => {
        const v = SP.apply(R, [a * L[0] / 2, b * L[1] / 2, d * L[2] / 2]); corners.push([ctr[0] + v[0], ctr[1] + v[1], ctr[2] + v[2]]);
      })));
      [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([i, j]) => this.line([corners[i], corners[j]], { color: o.color || "--c-ag", width: o.width ?? 2, alpha: o.alpha }));
      return this;
    }
  }
  const redrawAll = () => views.forEach(v => { if (v.c.isConnected) { v.resize(); v.render(); } });
  window.addEventListener("resize", redrawAll);
  window.addEventListener("themechange", redrawAll);
  window.SP = SP; window.View3D = View3D;
})();

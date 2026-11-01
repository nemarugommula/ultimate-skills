/* ============ Math for AI — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index) get theme + quizzes only.
 * Also exposes window.MX: small math/random helpers used by playgrounds,
 * and window.MathCourse: progress export/import + small helpers.
 * Storage keys start with "mathcourse." so progress never mixes with the RL course.
 */
const RL_COURSE = "../../rl-course"; // relative to a lesson page; links into the sibling RL course

const COURSE = [
  {
    id: "m0", num: 0, emoji: "🌱", title: "The language of math", status: "ready",
    why: "Warm up: why AI needs math at all, functions as machines, exponents and logs, and how to read Σ and Π like a for-loop.",
    lessons: [
      { id: "m0-1", title: "Why math for AI? A map of the journey", href: "phase0/1-why-math.html", mins: 10 },
      { id: "m0-2", title: "Functions: machines that turn inputs into outputs", href: "phase0/2-functions.html", mins: 15 },
      { id: "m0-3", title: "Exponents & logarithms: the multiply ↔ add trick", href: "phase0/3-exp-log.html", mins: 15 },
      { id: "m0-4", title: "e, exp() and growth: the most useful number in AI", href: "phase0/4-e-and-exp.html", mins: 15 },
      { id: "m0-5", title: "Σ and Π: reading math like code", href: "phase0/5-sigma-pi.html", mins: 15 },
      { id: "m0-r", title: "Recap: the language of math", href: "phase0/recap.html", mins: 15 },
    ],
  },
  {
    id: "m1", num: 1, emoji: "➡️", title: "Vectors & embeddings", status: "ready",
    why: "Linear algebra part 1. A vector is a list of numbers and an arrow at the same time. Learn the dot product, the one operation inside attention, search and every neuron.",
    lessons: [
      { id: "m1-1", title: "Vectors: arrows and lists of numbers", href: "phase1/1-vectors.html", mins: 15 },
      { id: "m1-2", title: "Adding & scaling vectors (king − man + woman)", href: "phase1/2-vector-arithmetic.html", mins: 15 },
      { id: "m1-3", title: "The dot product & cosine similarity", href: "phase1/3-dot-product.html", mins: 20 },
      { id: "m1-4", title: "Length, distance & norms (L1, L2)", href: "phase1/4-norms.html", mins: 15 },
      { id: "m1-5", title: "Embeddings: meaning as geometry", href: "phase1/5-embeddings.html", mins: 15 },
      { id: "m1-r", title: "Recap: vectors", href: "phase1/recap.html", mins: 15 },
    ],
  },
  {
    id: "m2", num: 2, emoji: "🧮", title: "Matrices & transformations", status: "ready",
    why: "Linear algebra part 2. A matrix is a machine that moves space. Every neural-network layer is one, and SVD, eigenvectors and low rank explain PCA, PageRank and LoRA.",
    lessons: [
      { id: "m2-1", title: "Matrices: machines that move space", href: "phase2/1-matrices.html", mins: 15 },
      { id: "m2-2", title: "Matrix multiplication = a neural-network layer", href: "phase2/2-matmul.html", mins: 20 },
      { id: "m2-3", title: "Tensors, shapes & broadcasting", href: "phase2/3-tensors.html", mins: 15 },
      { id: "m2-4", title: "Inverse, rank & independence (why LoRA works)", href: "phase2/4-rank.html", mins: 20 },
      { id: "m2-5", title: "Eigenvectors: the directions a matrix can't turn", href: "phase2/5-eigen.html", mins: 20 },
      { id: "m2-6", title: "SVD & PCA: finding what matters in data", href: "phase2/6-svd-pca.html", mins: 20 },
      { id: "m2-r", title: "Recap: matrices", href: "phase2/recap.html", mins: 15 },
    ],
  },
  {
    id: "m3", num: 3, emoji: "📈", title: "Calculus: how things change", status: "ready",
    why: "Derivatives tell you which way to nudge a knob. The chain rule becomes backpropagation, gradients become gradient descent: this is how every neural network learns.",
    lessons: [
      { id: "m3-1", title: "Derivatives: slope & sensitivity", href: "phase3/1-derivatives.html", mins: 15 },
      { id: "m3-2", title: "Derivative rules & the chain rule", href: "phase3/2-chain-rule.html", mins: 20 },
      { id: "m3-3", title: "Partial derivatives & the gradient", href: "phase3/3-gradients.html", mins: 20 },
      { id: "m3-4", title: "Gradient descent: rolling downhill", href: "phase3/4-gradient-descent.html", mins: 20 },
      { id: "m3-4b", title: "Taylor approximation & curvature: why descent works", href: "phase3/4b-taylor-curvature.html", mins: 20 },
      { id: "m3-5", title: "Backpropagation: the chain rule on a graph", href: "phase3/5-backprop.html", mins: 20 },
      { id: "m3-6", title: "Integrals: area & adding up tiny pieces", href: "phase3/6-integrals.html", mins: 15 },
      { id: "m3-r", title: "Recap: calculus", href: "phase3/recap.html", mins: 15 },
    ],
  },
  {
    id: "m4", num: 4, emoji: "🎲", title: "Probability: reasoning with uncertainty", status: "ready",
    why: "An LLM is a probability machine: every token is a draw from a distribution. Learn chance, Bayes, distributions, expectation, sampling and temperature.",
    lessons: [
      { id: "m4-1", title: "Probability: counting chances", href: "phase4/1-probability.html", mins: 15 },
      { id: "m4-2", title: "Conditional probability & Bayes' rule", href: "phase4/2-bayes.html", mins: 20 },
      { id: "m4-3", title: "Random variables & distributions", href: "phase4/3-distributions.html", mins: 20 },
      { id: "m4-4", title: "Expectation & variance", href: "phase4/4-expectation.html", mins: 15 },
      { id: "m4-5", title: "Joint probability & the chain rule (how LLMs write)", href: "phase4/5-joint-chain.html", mins: 15 },
      { id: "m4-6", title: "Sampling, temperature & Monte Carlo", href: "phase4/6-sampling.html", mins: 20 },
      { id: "m4-6b", title: "Importance sampling: reusing someone else's samples", href: "phase4/6b-importance-sampling.html", mins: 20 },
      { id: "m4-7", title: "Law of large numbers & the bell curve (CLT)", href: "phase4/7-lln-clt.html", mins: 15 },
      { id: "m4-r", title: "Recap: probability", href: "phase4/recap.html", mins: 15 },
    ],
  },
  {
    id: "m5", num: 5, emoji: "📊", title: "Statistics & data science", status: "ready",
    why: "From data to decisions: summaries, correlation vs causation, maximum likelihood (what training really is), Bayesian thinking, A/B tests and overfitting.",
    lessons: [
      { id: "m5-1", title: "Describing data: mean, median, spread", href: "phase5/1-describing-data.html", mins: 15 },
      { id: "m5-2", title: "Covariance, correlation & causation", href: "phase5/2-correlation.html", mins: 15 },
      { id: "m5-2b", title: "Clustering with k-means: finding groups", href: "phase5/2b-kmeans.html", mins: 20 },
      { id: "m5-3", title: "Maximum likelihood: training = making data probable", href: "phase5/3-mle.html", mins: 20 },
      { id: "m5-4", title: "Bayesian thinking: priors & posteriors", href: "phase5/4-bayesian.html", mins: 20 },
      { id: "m5-5", title: "Confidence intervals, p-values & A/B tests", href: "phase5/5-ab-tests.html", mins: 20 },
      { id: "m5-6", title: "Overfitting, bias–variance & train/test splits", href: "phase5/6-overfitting.html", mins: 20 },
      { id: "m5-6b", title: "Classification metrics: precision, recall, F1 & calibration", href: "phase5/6b-metrics.html", mins: 20 },
      { id: "m5-r", title: "Recap: statistics", href: "phase5/recap.html", mins: 15 },
    ],
  },
  {
    id: "m6", num: 6, emoji: "📡", title: "Information theory", status: "ready",
    why: "Surprise, entropy, cross-entropy and KL divergence: the loss every LLM is trained on, the leash in RLHF, and the math of compression.",
    lessons: [
      { id: "m6-1", title: "Information = surprise (bits and −log p)", href: "phase6/1-surprise.html", mins: 15 },
      { id: "m6-2", title: "Entropy: average surprise", href: "phase6/2-entropy.html", mins: 15 },
      { id: "m6-3", title: "Cross-entropy & perplexity: the LLM loss", href: "phase6/3-cross-entropy.html", mins: 20 },
      { id: "m6-4", title: "KL divergence: distance between beliefs", href: "phase6/4-kl.html", mins: 20 },
      { id: "m6-5", title: "Mutual information & contrastive learning", href: "phase6/5-mutual-info.html", mins: 15 },
      { id: "m6-r", title: "Recap: information theory", href: "phase6/recap.html", mins: 15 },
    ],
  },
  {
    id: "m7", num: 7, emoji: "⛰️", title: "Optimization: finding the best knobs", status: "ready",
    why: "Training is optimization. Loss landscapes, SGD and mini-batches, momentum and Adam, learning-rate schedules, regularization and constraints.",
    lessons: [
      { id: "m7-1", title: "Loss functions & landscapes", href: "phase7/1-loss-landscapes.html", mins: 15 },
      { id: "m7-2", title: "Stochastic gradient descent & mini-batches", href: "phase7/2-sgd.html", mins: 20 },
      { id: "m7-3", title: "Momentum, Adam & learning-rate schedules", href: "phase7/3-adam.html", mins: 20 },
      { id: "m7-4", title: "Regularization: weight decay, dropout, early stopping", href: "phase7/4-regularization.html", mins: 20 },
      { id: "m7-5", title: "Constraints & Lagrange multipliers", href: "phase7/5-constraints.html", mins: 20 },
      { id: "m7-r", title: "Recap: optimization", href: "phase7/recap.html", mins: 15 },
    ],
  },
  {
    id: "m8", num: 8, emoji: "🧠", title: "Putting it together: the math inside real models", status: "ready",
    why: "Every earlier phase meets here: regression, softmax classifiers, a neural network, attention and transformers, floating point, and the math of RL.",
    lessons: [
      { id: "m8-1", title: "Linear & logistic regression from the math up", href: "phase8/1-regression.html", mins: 20 },
      { id: "m8-1b", title: "Decision trees & gradient boosting", href: "phase8/1b-trees-boosting.html", mins: 25 },
      { id: "m8-2", title: "Softmax classifiers & the cross-entropy gradient", href: "phase8/2-softmax.html", mins: 20 },
      { id: "m8-3", title: "A neural network, all the math", href: "phase8/3-neural-net.html", mins: 25 },
      { id: "m8-3b", title: "Convolution: how networks see images", href: "phase8/3b-convolution.html", mins: 20 },
      { id: "m8-4", title: "Attention & transformers: softmax(QKᵀ/√d)V", href: "phase8/4-attention.html", mins: 25 },
      { id: "m8-4b", title: "Multivariate Gaussians & diffusion models", href: "phase8/4b-diffusion.html", mins: 25 },
      { id: "m8-5", title: "Numbers on computers: floats, log-sum-exp, bf16", href: "phase8/5-numerics.html", mins: 20 },
      { id: "m8-6", title: "The math of RL: expected return, Bellman, policy gradients", href: "phase8/6-rl-math.html", mins: 25 },
      { id: "m8-r", title: "Graduation: the whole map", href: "phase8/recap.html", mins: 20 },
    ],
  },
];

/* ---------- safe localStorage (can throw in private mode / blocked storage) ---------- */
const Store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } },
  json(k, fallback) { try { const v = JSON.parse(this.get(k)); return v == null ? fallback : v; } catch (e) { return fallback; } },
};

/* ---------- progress ---------- */
const Progress = {
  key: "mathcourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "m1-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "mathcourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "mathcourse.theme";
const darkMQ = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
function applyTheme(t, save) {
  document.documentElement.setAttribute("data-theme", t);
  if (save !== false) Store.set(THEME_KEY, t);
  window.dispatchEvent(new Event("themechange"));
}
function toggleTheme() {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
}
applyTheme(Store.get(THEME_KEY) || (darkMQ && darkMQ.matches ? "dark" : "light"), false);
if (darkMQ) {
  const follow = e => { if (!Store.get(THEME_KEY)) applyTheme(e.matches ? "dark" : "light", false); };
  if (darkMQ.addEventListener) darkMQ.addEventListener("change", follow);
  else if (darkMQ.addListener) darkMQ.addListener(follow);
}
function bindThemeButtons() {
  document.querySelectorAll(".theme-btn").forEach(b => {
    if (!b.onclick) b.onclick = toggleTheme;
    if (!b.hasAttribute("aria-label")) b.setAttribute("aria-label", "Toggle light / dark theme");
    if (!b.title) b.title = "Toggle theme";
  });
}

const reducedMotion = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

/* ---------- shell ---------- */
function flatLessons() {
  return COURSE.filter(p => p.status === "ready").flatMap(p => p.lessons.map(l => ({ ...l, phase: p })));
}

function centerActive(side) {
  const a = side.querySelector("a.active");
  if (!a) return;
  // scroll only the sidebar (scrollIntoView would also scroll the page)
  const top = a.getBoundingClientRect().top - side.getBoundingClientRect().top + side.scrollTop;
  side.scrollTop = Math.max(0, top - side.clientHeight / 2 + a.offsetHeight / 2);
}

let refreshQuizScore = () => {};

function buildShell() {
  const body = document.body;
  const root = body.dataset.root || ".";
  const lessonId = body.dataset.lesson;
  const flat = flatLessons();
  const idx = flat.findIndex(l => l.id === lessonId);
  const cur = flat[idx];
  const done = Progress.all();

  const top = document.createElement("header");
  top.className = "topbar";
  top.innerHTML = `
    <button class="menu-btn" aria-label="Open lesson menu" aria-expanded="false">☰</button>
    <a class="logo" href="${root}/index.html">Math <span>for AI</span></a>
    <div class="crumb">${cur ? `Phase ${cur.phase.num} · ${cur.phase.title}` : ""}</div>
    <button class="theme-btn" title="Toggle theme" aria-label="Toggle light / dark theme">🌓</button>`;

  const side = document.createElement("nav");
  side.className = "sidebar";
  side.setAttribute("aria-label", "Course lessons");
  side.innerHTML = COURSE.map(p => {
    const open = cur && cur.phase.id === p.id;
    const head = extra => `<button type="button" class="phase-title" aria-expanded="${open}" aria-controls="side-${p.id}">` +
      `<span class="chev" aria-hidden="true">▸</span><span class="pt">${p.emoji} ${p.num}. ${p.title}</span>${extra}</button>`;
    if (p.status !== "ready")
      return `<div class="side-phase${open ? "" : " collapsed"}">${head("")}<div class="phase-lessons" id="side-${p.id}"><div class="locked"><span class="tick">🔒</span>Coming soon</div></div></div>`;
    const n = p.lessons.filter(l => done.has(l.id)).length;
    return `<div class="side-phase${open ? "" : " collapsed"}">` +
      head(`<span class="pcount">${n}/${p.lessons.length}</span>`) +
      `<div class="phase-lessons" id="side-${p.id}">` + p.lessons.map(l =>
        `<a href="${root}/${l.href}" class="${l.id === lessonId ? "active" : ""}"${l.id === lessonId ? ` aria-current="page"` : ""}><span class="tick">${done.has(l.id) ? "✅" : "○"}</span>${l.title}</a>`).join("") +
      `</div></div>`;
  }).join("");
  side.querySelectorAll(".phase-title").forEach(t => (t.onclick = () => {
    const g = t.parentNode, collapsed = g.classList.toggle("collapsed");
    t.setAttribute("aria-expanded", String(!collapsed));
  }));

  const main = document.querySelector("main");
  const shell = document.createElement("div");
  shell.className = "shell";
  const content = document.createElement("div");
  content.className = "content";
  main.parentNode.insertBefore(shell, main);
  content.appendChild(main);
  shell.append(side, content);
  body.insertBefore(top, shell);

  top.querySelector(".theme-btn").onclick = toggleTheme;
  const menu = top.querySelector(".menu-btn");
  menu.onclick = () => {
    const open = side.classList.toggle("open");
    menu.setAttribute("aria-expanded", String(open));
    if (open) centerActive(side);
  };
  centerActive(side);

  if (!cur) return;
  const phaseLessons = cur.phase.lessons;
  const k = phaseLessons.findIndex(l => l.id === cur.id) + 1;

  // meta line under the title
  const h1 = main.querySelector("h1");
  if (h1 && !main.querySelector(".meta")) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<span>⏱ ~${cur.mins} min</span><span>📍 Lesson ${idx + 1} of ${flat.length}</span>` +
      `<span>🧭 Phase ${cur.phase.num} · ${k} of ${phaseLessons.length}</span>`;
    (main.querySelector(".subtitle") || h1).after(meta);
  }

  // complete button + quiz score + prev/next
  const wrap = document.createElement("div");
  wrap.className = "complete-wrap";
  const btn = document.createElement("button");
  btn.className = "btn primary";
  const paint = () => (btn.textContent = Progress.has(cur.id) ? "✅ Completed (click to undo)" : "Mark lesson complete 🎉");
  paint();
  btn.onclick = () => {
    const on = !Progress.has(cur.id);
    Progress.set(cur.id, on);
    paint();
    if (on) confetti(btn);
    if (on && /-r$/.test(cur.id)) {
      const last = COURSE[COURSE.length - 1] === cur.phase;
      toast(`🏁 Phase ${cur.phase.num} complete!` + (last ? " You finished the course 🎓" : ""));
    }
    const link = side.querySelector("a.active .tick");
    if (link) link.textContent = on ? "✅" : "○";
    const cnt = side.querySelector(`#side-${cur.phase.id}`);
    const pc = cnt && cnt.parentNode.querySelector(".pcount");
    if (pc) pc.textContent = `${phaseLessons.filter(l => Progress.has(l.id)).length}/${phaseLessons.length}`;
  };
  wrap.appendChild(btn);

  if (main.querySelector(".quiz")) {
    const chip = document.createElement("span");
    chip.className = "quiz-score";
    chip.setAttribute("aria-live", "polite");
    wrap.appendChild(chip);
    refreshQuizScore = () => {
      const qs = [...main.querySelectorAll(".quiz")];
      const answered = qs.filter(q => q.classList.contains("done")).length;
      const right = qs.filter(q => q.classList.contains("done") && q.dataset.ok === "1").length;
      chip.textContent = `Quizzes: ${right}/${qs.length} ✓`;
      chip.title = `${answered} answered, ${right} correct`;
      chip.classList.toggle("all", right === qs.length);
    };
  }

  const nav = document.createElement("div");
  nav.className = "lesson-nav";
  const prev = flat[idx - 1], next = flat[idx + 1];
  nav.innerHTML =
    (prev ? `<a class="navcard prev" href="${root}/${prev.href}"><small>← Previous</small>${prev.title}</a>` : `<span></span>`) +
    (next ? `<a class="navcard next" href="${root}/${next.href}"><small>Next →</small>${next.title}</a>`
          : `<a class="navcard next" href="${root}/index.html"><small>Next →</small>Back to the roadmap</a>`);
  content.append(wrap, nav);

  // ← / → keys: previous / next lesson
  window.addEventListener("keydown", e => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const to = e.key === "ArrowLeft" ? prev : next;
    if (to) location.href = `${root}/${to.href}`;
  });
}

/* ---------- quizzes ---------- */
function initQuizzes() {
  const page = document.body.dataset.lesson || location.pathname.split("/").slice(-2).join("/");
  document.querySelectorAll(".quiz").forEach((q, qi) => {
    const id = `${page}#${qi}`;
    const opts = [...q.querySelectorAll("button.opt")];
    if (!opts.length) return;

    const answer = (b, save) => {
      if (q.classList.contains("done")) return;
      const ok = b.hasAttribute("data-ok");
      q.classList.add("done");
      q.dataset.ok = ok ? "1" : "0";
      opts.forEach(o => { o.disabled = true; if (o.hasAttribute("data-ok")) o.classList.add("right"); });
      if (!ok) b.classList.add("wrong");
      b.insertAdjacentHTML("afterbegin", `<span class="mark" aria-hidden="true">${ok ? "✅ " : "❌ "}</span>`);
      if (save) QuizStore.set(id, { pick: opts.indexOf(b), ok });
      refreshQuizScore();
    };
    const reset = () => {
      q.classList.remove("done");
      delete q.dataset.ok;
      opts.forEach(o => {
        o.disabled = false; o.classList.remove("right", "wrong");
        o.querySelectorAll(".mark").forEach(m => m.remove());
      });
      QuizStore.set(id, null);
      refreshQuizScore();
      opts[0].focus();
    };

    opts.forEach(b => b.addEventListener("click", () => answer(b, true)));
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "retry";
    retry.textContent = "↻ Try again";
    retry.onclick = reset;
    q.appendChild(retry);

    const saved = QuizStore.get(id);
    if (saved && opts[saved.pick]) answer(opts[saved.pick], false);
  });
  refreshQuizScore();
  document.querySelectorAll(".timeline .ev").forEach(e => e.addEventListener("click", () => e.classList.toggle("open")));
}

/* ---------- toast (non-blocking banner) ---------- */
function toast(msg, ms) {
  const t = document.createElement("div");
  t.className = "toast";
  t.setAttribute("role", "status");
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => { t.classList.remove("show"); setTimeout(() => t.remove(), 500); }, ms || 4500);
}

/* ---------- tiny confetti ---------- */
function confetti(anchor) {
  if (reducedMotion()) return;
  const r = anchor.getBoundingClientRect();
  const colors = ["#6c5ce7", "#00b894", "#fdcb6e", "#e17055", "#0984e3", "#e84393"];
  for (let i = 0; i < 60; i++) {
    const d = document.createElement("div");
    const ang = Math.random() * Math.PI * 2, sp = 120 + Math.random() * 220;
    Object.assign(d.style, {
      position: "fixed", left: r.left + r.width / 2 + "px", top: r.top + "px", width: "8px", height: "12px",
      background: colors[i % colors.length], borderRadius: "2px", zIndex: 999, pointerEvents: "none",
      transition: "transform 1.1s cubic-bezier(.2,.8,.3,1), opacity 1.1s",
    });
    document.body.appendChild(d);
    requestAnimationFrame(() => {
      d.style.transform = `translate(${Math.cos(ang) * sp}px, ${Math.sin(ang) * sp - 120}px) rotate(${Math.random() * 720}deg)`;
      d.style.opacity = "0";
    });
    setTimeout(() => d.remove(), 1200);
  }
}

/* ---------- progress export / import (progress is per-origin: file:// and localhost don't share it) ---------- */
window.MathCourse = {
  COURSE,
  RL_COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("mathcourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "math-for-ai", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like a Math for AI progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("mathcourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
      if (k === Progress.key) {
        const mine = Progress.all();
        JSON.parse(v).forEach(id => mine.add(id));
        Store.set(k, JSON.stringify([...mine]));
      } else if (k === QuizStore.key) {
        Store.set(k, JSON.stringify(Object.assign(QuizStore.all(), JSON.parse(v))));
      } else Store.set(k, v);
      n++;
    });
    return n;
  },
};

/* ---------- math helpers for playgrounds ---------- */
window.MX = {
  randn() { // standard normal (Box–Muller)
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  },
  bernoulli(p) { return Math.random() < p ? 1 : 0; },
  randint(n) { return Math.floor(Math.random() * n); },
  seeded(seed) { // mulberry32: repeatable random numbers, MX.seeded(42)() -> [0, 1)
    let a = seed >>> 0;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  },
  linspace(a, b, n) { return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1)); },
  sum(arr) { return arr.reduce((s, v) => s + v, 0); },
  mean(arr) { return arr.length ? this.sum(arr) / arr.length : NaN; },
  std(arr) { const m = this.mean(arr); return Math.sqrt(this.mean(arr.map(v => (v - m) ** 2))); },
  dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); },
  norm(a) { return Math.sqrt(this.dot(a, a)); },
  cosine(a, b) { return this.dot(a, b) / (this.norm(a) * this.norm(b) || 1); },
  matvec(M, v) { return M.map(row => this.dot(row, v)); },
  matmul(A, B) { return A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0))); },
  transpose(M) { return M[0].map((_, j) => M.map(row => row[j])); },
  sigmoid(z) { return 1 / (1 + Math.exp(-z)); },
  softmax(z, T) { // temperature T (default 1); subtract the max for numerical safety
    const t = T || 1, m = Math.max(...z), e = z.map(v => Math.exp((v - m) / t)), s = this.sum(e);
    return e.map(v => v / s);
  },
  entropy(p) { return -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0); }, // in bits
  sample(p) { // index drawn from a probability list
    let u = Math.random(), i = 0;
    for (; i < p.length - 1; i++) { u -= p[i]; if (u < 0) return i; }
    return i;
  },
  argmax(arr) { let b = 0; arr.forEach((v, i) => { if (v > arr[b]) b = i; }); return b; },
  clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); },
  fmt(v, d) { return (+v).toFixed(d == null ? 2 : d); },
  css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); },
};

document.addEventListener("DOMContentLoaded", () => {
  if (document.body.dataset.lesson) buildShell();
  bindThemeButtons();
  initQuizzes();
});

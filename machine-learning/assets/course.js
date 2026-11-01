/* ============ Machine Learning — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index) get theme + quizzes only.
 * Also exposes window.MX: small math/random helpers used by playgrounds,
 * and window.MLCourse: progress export/import + small helpers.
 * Storage keys start with "mlcourse." so progress never mixes with the sibling courses.
 * The COURSE manifest below is the source of truth for lesson ids, titles, files and minutes.
 */

const COURSE = [
  {
    id: "ml0", num: 0, emoji: "🌱", title: "Foundations: what machine learning is", status: "ready",
    why: "Learning a function from examples, the end-to-end workflow, the kinds of learning, when ML beats rules (and when deep learning beats trees), the scikit-learn API, and the splits that keep you honest.",
    lessons: [
      { id: "ml0-1", title: "What machine learning is: learning a function from examples", href: "phase0/1-what-is-ml.html", mins: 20 },
      { id: "ml0-2", title: "The ML workflow: data → features → model → evaluation → deployment", href: "phase0/2-workflow.html", mins: 20 },
      { id: "ml0-3", title: "Kinds of learning: supervised, unsupervised, self-supervised, RL", href: "phase0/3-kinds-of-learning.html", mins: 20 },
      { id: "ml0-4", title: "When to use ML: rules vs classical ML vs deep learning", href: "phase0/4-when-ml.html", mins: 20 },
      { id: "ml0-5", title: "The scikit-learn API: fit, predict, transform", href: "phase0/5-sklearn-api.html", mins: 20 },
      { id: "ml0-6", title: "Train, validation & test splits, and data leakage", href: "phase0/6-splits-leakage.html", mins: 25 },
      { id: "ml0-r", title: "Recap: Foundations", href: "phase0/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml1", num: 1, emoji: "🧹", title: "Data & features", status: "ready",
    why: "Most of the accuracy is decided before the model: looking at the data, fixing missing values and outliers, encoding categories, scaling, inventing better features, and wiring it all into a leak-proof Pipeline.",
    lessons: [
      { id: "ml1-1", title: "Exploratory data analysis: look before you model", href: "phase1/1-eda.html", mins: 20 },
      { id: "ml1-2", title: "Missing values & outliers", href: "phase1/2-missing-outliers.html", mins: 25 },
      { id: "ml1-3", title: "Encoding categoricals: one-hot, ordinal, target encoding", href: "phase1/3-encoding.html", mins: 25 },
      { id: "ml1-4", title: "Scaling: why distance-based models need it", href: "phase1/4-scaling.html", mins: 20 },
      { id: "ml1-5", title: "Feature engineering: dates, ratios, interactions, text counts", href: "phase1/5-feature-engineering.html", mins: 25 },
      { id: "ml1-6", title: "Pipelines & ColumnTransformer", href: "phase1/6-pipelines.html", mins: 25 },
      { id: "ml1-r", title: "Recap: Data & features", href: "phase1/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml2", num: 2, emoji: "📈", title: "Linear models", status: "ready",
    why: "The workhorses: least squares, regularization (and why L1 makes zeros), logistic regression and thresholds, multiclass, and how stacking linear models with bends becomes a neural network.",
    lessons: [
      { id: "ml2-1", title: "Linear regression: least squares, intuitively", href: "phase2/1-linear-regression.html", mins: 25 },
      { id: "ml2-2", title: "Regularization: ridge, lasso, elastic net", href: "phase2/2-regularization.html", mins: 25 },
      { id: "ml2-3", title: "Logistic regression: log-odds, boundaries, thresholds", href: "phase2/3-logistic-regression.html", mins: 25 },
      { id: "ml2-4", title: "Multiclass & beyond: softmax, one-vs-rest, Poisson", href: "phase2/4-multiclass-glm.html", mins: 20 },
      { id: "ml2-5", title: "Polynomial features & the step to neural networks", href: "phase2/5-polynomial-to-nn.html", mins: 20 },
      { id: "ml2-r", title: "Recap: Linear models", href: "phase2/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml3", num: 3, emoji: "📏", title: "Distance & probability models", status: "ready",
    why: "Predict from your neighbours (and why that breaks in 100 dimensions), count words with naive Bayes, and draw the widest possible street with support vector machines and kernels.",
    lessons: [
      { id: "ml3-1", title: "k-nearest neighbours & the curse of dimensionality", href: "phase3/1-knn.html", mins: 25 },
      { id: "ml3-2", title: "Naive Bayes: why “naive” still catches spam", href: "phase3/2-naive-bayes.html", mins: 25 },
      { id: "ml3-3", title: "Support vector machines: margins & hinge loss", href: "phase3/3-svm.html", mins: 25 },
      { id: "ml3-4", title: "Kernels & the kernel trick", href: "phase3/4-kernels.html", mins: 25 },
      { id: "ml3-r", title: "Recap: Distance & probability models", href: "phase3/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml4", num: 4, emoji: "🌳", title: "Trees & ensembles", status: "ready",
    why: "Decision trees, then the two big ways to combine them: averaging (bagging, random forests) and correcting (boosting, XGBoost, LightGBM, CatBoost). Plus stacking, and why boosted trees still rule tabular data.",
    lessons: [
      { id: "ml4-1", title: "Decision trees: questions, Gini & entropy", href: "phase4/1-decision-trees.html", mins: 25 },
      { id: "ml4-2", title: "Bagging & random forests", href: "phase4/2-random-forests.html", mins: 30 },
      { id: "ml4-3", title: "Boosting: from AdaBoost to gradient boosting", href: "phase4/3-boosting.html", mins: 30 },
      { id: "ml4-4", title: "XGBoost, LightGBM & CatBoost: what each changed", href: "phase4/4-xgboost-lightgbm-catboost.html", mins: 25 },
      { id: "ml4-5", title: "Stacking & blending", href: "phase4/5-stacking.html", mins: 20 },
      { id: "ml4-6", title: "Why boosted trees still win on tabular data (and when they don't)", href: "phase4/6-trees-tabular.html", mins: 20 },
      { id: "ml4-r", title: "Recap: Trees & ensembles", href: "phase4/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml5", num: 5, emoji: "🎯", title: "Evaluation & tuning", status: "ready",
    why: "How to know a model is good before your users find out: cross-validation done right, the right metric for the job, ROC and PR curves, imbalanced data, calibrated probabilities, hyperparameter search, and reading learning curves.",
    lessons: [
      { id: "ml5-1", title: "Cross-validation: k-fold, stratified, group, time series", href: "phase5/1-cross-validation.html", mins: 25 },
      { id: "ml5-2", title: "Regression metrics: MAE, RMSE, R², MAPE", href: "phase5/2-regression-metrics.html", mins: 20 },
      { id: "ml5-3", title: "Classification metrics & the confusion matrix", href: "phase5/3-classification-metrics.html", mins: 25 },
      { id: "ml5-4", title: "ROC & precision–recall curves", href: "phase5/4-roc-pr.html", mins: 25 },
      { id: "ml5-5", title: "Imbalanced data: class weights, SMOTE, threshold moving", href: "phase5/5-imbalanced.html", mins: 25 },
      { id: "ml5-6", title: "Calibration: when 70% should mean 70%", href: "phase5/6-calibration.html", mins: 20 },
      { id: "ml5-7", title: "Hyperparameter search: grid, random, Bayesian (Optuna)", href: "phase5/7-hyperparameter-search.html", mins: 25 },
      { id: "ml5-8", title: "Learning curves & bias–variance in practice", href: "phase5/8-bias-variance.html", mins: 25 },
      { id: "ml5-r", title: "Recap: Evaluation & tuning", href: "phase5/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml6", num: 6, emoji: "🔮", title: "Unsupervised learning", status: "ready",
    why: "Finding structure with no labels: k-means, hierarchical, DBSCAN/HDBSCAN and Gaussian mixtures; squeezing many columns into few with PCA, t-SNE and UMAP; and catching the weird ones with anomaly detection.",
    lessons: [
      { id: "ml6-1", title: "k-means & choosing k", href: "phase6/1-kmeans.html", mins: 25 },
      { id: "ml6-2", title: "Hierarchical clustering & dendrograms", href: "phase6/2-hierarchical.html", mins: 20 },
      { id: "ml6-3", title: "DBSCAN & HDBSCAN: clusters of any shape", href: "phase6/3-dbscan.html", mins: 20 },
      { id: "ml6-4", title: "Gaussian mixture models: soft clusters", href: "phase6/4-gmm.html", mins: 20 },
      { id: "ml6-5", title: "PCA: the directions that matter", href: "phase6/5-pca.html", mins: 25 },
      { id: "ml6-6", title: "t-SNE & UMAP: what each map preserves", href: "phase6/6-tsne-umap.html", mins: 20 },
      { id: "ml6-7", title: "Anomaly detection: isolation forest, one-class SVM, LOF", href: "phase6/7-anomaly-detection.html", mins: 25 },
      { id: "ml6-r", title: "Recap: Unsupervised learning", href: "phase6/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml7", num: 7, emoji: "🧭", title: "Special domains", status: "ready",
    why: "Where classical ML earns its living: forecasting time series, recommending items, ranking search results, classifying and clustering text, and getting more from fewer labels.",
    lessons: [
      { id: "ml7-1", title: "Time series: lags, seasonality & leakage-safe validation", href: "phase7/1-time-series.html", mins: 25 },
      { id: "ml7-2", title: "Forecasting models: ARIMA & ETS vs gradient boosting", href: "phase7/2-forecasting-models.html", mins: 25 },
      { id: "ml7-3", title: "Recommender systems: collaborative filtering & matrix factorization", href: "phase7/3-recommenders.html", mins: 25 },
      { id: "ml7-4", title: "Learning to rank", href: "phase7/4-learning-to-rank.html", mins: 20 },
      { id: "ml7-5", title: "Classical NLP: bag-of-words, TF-IDF, text classifiers", href: "phase7/5-classical-nlp.html", mins: 25 },
      { id: "ml7-6", title: "Topic models: LDA & friends", href: "phase7/6-topic-models.html", mins: 20 },
      { id: "ml7-7", title: "Semi-supervised & active learning", href: "phase7/7-semi-active.html", mins: 20 },
      { id: "ml7-r", title: "Recap: Special domains", href: "phase7/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml8", num: 8, emoji: "🏭", title: "Production & responsible ML", status: "ready",
    why: "Explaining models (permutation importance, SHAP, partial dependence), measuring fairness, telling correlation from causation, and running models for real: tracking, registries, feature stores, serving, drift and AutoML.",
    lessons: [
      { id: "ml8-1", title: "Interpretability: feature importance, permutation, partial dependence", href: "phase8/1-interpretability.html", mins: 25 },
      { id: "ml8-2", title: "SHAP: fair credit for every feature", href: "phase8/2-shap.html", mins: 25 },
      { id: "ml8-3", title: "Fairness: metrics, trade-offs & mitigation", href: "phase8/3-fairness.html", mins: 25 },
      { id: "ml8-4", title: "Causal inference basics: A/B tests & uplift modelling", href: "phase8/4-causal.html", mins: 25 },
      { id: "ml8-5", title: "MLOps: tracking, registry, feature stores, serving", href: "phase8/5-mlops.html", mins: 25 },
      { id: "ml8-6", title: "Drift monitoring & retraining", href: "phase8/6-drift.html", mins: 20 },
      { id: "ml8-7", title: "AutoML: what it automates and what it can't", href: "phase8/7-automl.html", mins: 20 },
      { id: "ml8-r", title: "Recap: Production & responsible ML", href: "phase8/recap.html", mins: 25 },
    ],
  },
  {
    id: "ml9", num: 9, emoji: "🎓", title: "Becoming senior: design, projects & interviews", status: "ready",
    why: "Put it all together: a framework for ML system design, three classic designs (fraud, churn, ads click-through), an end-to-end Kaggle-style project, interview prep, and where to go next.",
    lessons: [
      { id: "ml9-1", title: "ML system design: a framework", href: "phase9/1-ml-system-design.html", mins: 25 },
      { id: "ml9-2", title: "Design cases: fraud detection, churn, ads click-through", href: "phase9/2-design-cases.html", mins: 30 },
      { id: "ml9-3", title: "An end-to-end project, Kaggle style", href: "phase9/3-end-to-end-project.html", mins: 30 },
      { id: "ml9-4", title: "Interview prep: the questions and the answers that land", href: "phase9/4-interview-prep.html", mins: 25 },
      { id: "ml9-5", title: "Where to go next: deep learning, books & courses", href: "phase9/5-where-next.html", mins: 15 },
      { id: "ml9-r", title: "Graduation: the whole course on one page", href: "phase9/recap.html", mins: 30 },
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
  key: "mlcourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "a3-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "mlcourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "mlcourse.theme";
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
    <a class="logo" href="${root}/index.html">Machine <span>Learning</span></a>
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
window.MLCourse = {
  COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("mlcourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "machine-learning", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like a Machine Learning course progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("mlcourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
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
  initPredicts();
});

/* ---------- 🔮 predict-then-reveal: <div class="predict" data-answer="index of the right choice"> ----------
 * <div class="predict" data-answer="1"><span class="label">🔮 Predict first</span><p>question</p>
 *   <div class="choices"><button>A</button><button>B</button><button>C</button></div>
 *   <div class="reveal">what really happens, and why</div></div>
 * Not scored and not saved (quizzes are). Any button reveals; the right one turns green. */
function initPredicts() {
  document.querySelectorAll(".predict").forEach(pr => {
    const bs = [...pr.querySelectorAll(".choices button")], ok = +pr.dataset.answer;
    bs.forEach((b, i) => { b.type = "button"; b.onclick = () => {
      if (pr.classList.contains("done")) return;
      pr.classList.add("done");
      bs.forEach((x, j) => { x.disabled = true; if (j === ok) x.classList.add("right"); });
      if (i !== ok) b.classList.add("wrong");
      pr.dispatchEvent(new CustomEvent("revealed", { detail: { pick: i, ok: i === ok } }));
    }; });
  });
}

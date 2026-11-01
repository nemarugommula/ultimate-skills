# AGENTS.md: Machine Learning course

Static, dependency-free HTML course (no build step; must work from `file://`). Sibling of `~/rl-course`, `~/math-course`,
`~/system-design-course` and `~/ai-engineer`, built on the same design system (and soon `~/deep-learning`, `~/autonomous-systems`).
Goal: a beginner (a working software engineer) learns to **understand, use and choose classical ML methods**, with intuition, ready
for real work and interviews. **Not a proof course**: formulas link to `~/math-course` lessons and are not re-derived.

## The learner's #1 request: intuition, motivation, fun
- **Never just say “we use X”. Always explain WHY X**: what problem existed before, what people tried, why X fixes it (plain words,
  numbers, a picture or playground), what it costs, when it fails, what refined it. That is the 🤔 box below, and the tone of every page.
- **Same format as the existing courses.** Before writing, open `phase4/2-random-forests.html` (the reference lesson) and skim
  `~/ai-engineer/phase3/3-chunking.html` and one `~/math-course` lesson. Same CSS components, box order, voice, quizzes, recaps.
- **Curiosity first.** Open with a puzzle, a surprising real result or a “what would you do?” question that the lesson answers.
- **Predict, then reveal.** Before a playground shows the answer, ask the learner to guess with a `.predict` widget (see below).
  Aim for 2–3 per lesson.
- **Interactive throughout**, not one demo at the end: a playground, prediction or small challenge **near every big idea**
  (≥ 2 real playgrounds, usually 3–4). Let the learner **break it and fix it** (overfit, leak, pick the wrong k), add **score targets**
  (`.goal`: “🎯 get validation error under 0.10”), and show **live numbers** that update as they drag.
- **Motivation.** Show the real-world payoff early (who uses this, what it powers). Light, friendly, conversational voice; short
  sections, small wins, fun facts. **Never a wall of text**: if a section has more than ~3 paragraphs in a row, add a table, a
  prediction, a diagram or a playground.

## Facts must be current (today is 2026-11-01)
Verify tool versions, APIs, stories and numbers on the web; **never invent numbers, scores or quotes**. If you can't verify a claim, drop it.
Versions checked on PyPI on 2026-11-01 (re-check anything else): scikit-learn **1.9.1**, pandas **3.1.0**, NumPy **2.5.4**,
XGBoost **3.4.1**, LightGBM **4.7.0**, CatBoost **1.2.10**, Optuna **5.0.0**, SHAP **0.54.1**, MLflow **3.17.2**,
imbalanced-learn **0.14.2**, umap-learn **0.5.12**, hdbscan **0.8.44**, statsmodels **0.15.0**, Polars **2.1.0**, Feast **0.67.0**,
Evidently **0.7.23**. Write current APIs (e.g. scikit-learn: `max_features="sqrt"` default for forests, `HistGradientBoostingClassifier`
has native categorical support, `TargetEncoder` exists since 1.3 and cross-fits internally, `sklearn.cluster.HDBSCAN` since 1.3,
`set_output(transform="pandas")`). Check the current docs page before quoting a default or a parameter name.
Label typical numbers as typical; anything time-sensitive is “as of 2026” and sourced.

## Files
- `assets/course.js`: `COURSE` manifest (phases `ml0`–`ml9`, lesson `id`, `title`, `href`, `mins`), sidebar, prev/next, progress,
  quizzes, `.predict` widgets, theme, `window.MX` helpers (randn, seeded, linspace, sum, mean, std, dot, norm, sigmoid, softmax,
  argmax, clamp, fmt, css(var)…). Storage keys start with `mlcourse.`. **The manifest is the source of truth** (ids, files, titles).
- `assets/ml.js`: **the offline ML toy kit. Read its header first.** scikit-learn-like models you can refit on every slider move:
  `ML.KNN`, `ML.Logistic`, `ML.Linear` (OLS / ridge / lasso / elastic net), `ML.Tree` (CART; `.root`, `.regions(box)`, `.apply`),
  `ML.Forest` (`.oobScore`, `.use(m)`), `ML.GBM` (`.stage(k)`, `.history`), `ML.AdaBoost`, `ML.GaussianNB`, `ML.MultinomialNB`,
  `ML.SVM` (simplified SMO, linear/rbf/poly, `.supportIdx`, `.decision`), `ML.KMeans` (`.init/.step`), `ML.pca`, `ML.eigSym`;
  datasets `ML.blobs/moons/circles/xor/spiral/linear1d`, `ML.split`, `ML.kfold`, `ML.bootstrap`, `ML.standardize`, `ML.poly1d`;
  metrics `ML.accuracy/mse/rmse/mae/r2/logloss/brier/confusion/roc/prCurve/auc`, `ML.tokenize`, `ML.rng(seed)`.
  Everything is seeded: same seed, same data, same model. It's also a Node module (`require(".../ml.js")`) so you can **prototype a
  playground and compute every number you quote in node first**.
  Not in the kit (write them in your page's IIFE): GMM/EM, DBSCAN, hierarchical clustering, t-SNE, isolation forest, LOF, matrix
  factorization, ARIMA/ETS, SHAP, calibration, SMOTE, etc. Keep them small and seeded.
- `assets/plane.js`: `new Plane(canvas, {xRange, yRange, range, grid, snap, onDrag(key,[x,y]), onClick([x,y], event)})` then `.draw(items)`.
  Items: `field` (decision regions: `f(x,y)` → P(class 1) or an array of class probabilities; shaded by class, gold boundary line;
  pass `cacheKey` so unchanged fields aren't recomputed), `pts` (`[[x,y,cls]]`, shapes ● ■ ▲ ◆ by class, `hollow` for test points,
  `ring: [i]`, `drag: "p"` → keys `"p:<i>"`), `circle` (world-unit radius), `rect`, `vec`, `pt`, `seg`, `fn`, `path`, `text`, `heat`.
  Use `<canvas class="plane">` (square) or `class="plane wide"` (16:9). Redraw with `requestAnimationFrame` when sliders move.
- `assets/plot.js`: `Plot.line(canvas, {series:[{label,color,data:[[x,y]],dash,dots,dotsOnly,fill,width}], xMin,xMax,yMin,yMax,xLabel,yLabel,
  hlines:[{y,label,color}], vlines:[{x,label,color}], xTicks, xFmt, yFmt})` (log axis: data `[Math.log10(x), y]`, `xFmt: v => 10 ** v`),
  `Plot.bars(canvas, {bars:[{label,value,color,err,marker}], yMin,yMax,yLabel,highlight})`. Use `<canvas class="chart">`.
- `assets/flow.js`: animated pipeline diagrams: `new Flow(svg, {width, height})`, `.node(id, {x, y, w, h, label, sub, kind})`
  (kind: `ft` data, `mo` model, `pd` prediction, `tg` label, `lo` loss, `mt` metric, `bd` threshold/knob), `.edge(a, b, {label, dashed})`,
  `.render()`, `.set(id, {sub, hot, down})`, `await .packet([ids…], {color})`.
- `assets/style.css`: the design system. Extra components for this course: `.box.why`, `.predict`, `.goal`, `table.data`
  (dataset rows; `td.miss` for missing values), `table.cm` (confusion matrix: `td.tp/.tn/.fp/.fn`), `.seg` (segmented buttons),
  `.verdict`, `.doc`, `details.mathbox`, `.cards/.case`.
- **Do not edit shared files** (`assets/*`, `index.html`, `review.html`, `glossary.html`, `tools/*`, `AGENTS.md`, `README.md`, other phases).
  If you need a shared helper or find a bug in one, work around it in your page and **say so in your final report**.
- `tools/test-pages.mjs` (headless harness), `tools/build-review-bank.py`, `tools/check-links.py`, `tools/glossary/` (see Glossary).

## Lesson anatomy (keep this order)
`<body data-lesson="ml<P>-<k>" data-root="..">`, title `P.k Title · Machine Learning`, same head tags (favicon 🌲) as the reference.
1. `.kicker` “Phase P · Lesson k” + `h1` + `.subtitle` (one or two sentences: what + why it matters).
2. `.box.prereq` “🧰 You'll need (quick reminders)”: 1–3 one-line reminders with lesson numbers (skip only in 0.1).
3. `.box.hook` “🎯 Hook”: a **real, verified** story, result or puzzle this lesson explains. Then usually a `.predict`.
4. Content. **Numbers first → plain sentence → picture/formula.** Never a formula before its words; every `.eq` has a `.say` line that
   reads it out loud. Include `.box.bigidea`, `.box.analogy` (everyday picture), **`.box.why` “🤔 Why this?”** (below),
   `.box.walk` “✏️ Worked example” (small numbers by hand), comparison tables, and **≥ 2 `.playground`s** that are genuinely
   interactive with live readouts; let the learner cause the failure and see the fix. Add `.predict` widgets and `.goal` targets.
5. `.box.tools` “🧰 Tools you'll actually use”: real libraries with a **short, correct, current** Python snippet (scikit-learn,
   pandas, XGBoost/LightGBM/CatBoost, statsmodels, Optuna, SHAP, MLflow… whatever fits). Use built-in datasets (`load_*`, `make_*`).
   Syntax-highlight with `<span class="k|s|n|c">` like the reference. **Run every snippet** in the venv (see Verify) unless it needs a
   package we can't install (XGBoost/LightGBM need OpenMP here); then check it against the official docs instead.
6. `.box.tradeoff` “⚖️ Trade-offs” (table). 7. `.box.startup` “🚀 Startup vs big tech” (`.cols`). 8. `.box.interview` “🎤 Interview angle”.
9. `.box.used` “🏭 In the real world”: 3–5 `.use` rows with **How:** / **Why this method:** and a linked **primary source** (paper,
   official docs, company engineering blog). Tags: `tag co` 🏢 Company · year, `tag res` 🧪 Research · year, `tag oss` 🧰 Open source,
   `tag hist` 📜 History · year.
10. `.box.fact` “🤯 Fun fact” (verified). 11. `.box.bridge` “🌉 How it connects” (lessons by number + sibling-course links).
12. `.box.explore` “📚 Explore more” (2–3 verified links, `.kind` Paper/Book/Docs/Course/Video/Blog).
13. `h2` “Check yourself” + **exactly 3 `.quiz`** (one `button.opt[data-ok]`, 3 options, a teaching `.explain`; one calculation or
    estimation, one realistic scenario). 14. `.box.tldr` “📝 TL;DR” (3–5 bullets).
Scripts at the end: `../assets/course.js`, then `plot.js` / `plane.js` / `flow.js` / `ml.js` as needed, then **one IIFE**.
Lessons are typically 35–60 KB. Concept-only lessons (0.1–0.4, 8.5, 9.x) still need ≥ 2 interactive playgrounds (sorters, simulators,
cost calculators, “pick the method” games) and a 🤔 box about the main idea (e.g. 0.6: “why a separate test set?”).

### 🤔 Why this? (required in every lesson)
`<div class="box why"><span class="label">🤔 Why this? Why &lt;method&gt;?</span><ol>` with five `<li>` that start with these bold leads:
1. **The problem before.** What was broken or impossible without this method.
2. **What people tried.** And why that wasn't enough.
3. **Why this works.** The intuition, **with a tiny numeric example**.
4. **What it costs, when it fails.**
5. **What came next.** What refined or replaced it (link the lesson).
Example in spirit (random forests): a single deep tree memorizes noise (high variance). Averaging many trees trained on different
bootstrap samples and feature subsets cancels independent mistakes. With 100 trees whose errors correlate at 0.3, the variance drops
to roughly 0.3 + 0.7/100 = 0.307 of one tree's. The cost is less interpretability and more memory. Put it after the analogy, before the
main mechanics, and keep it to 5 tight items (one to three sentences each).

### 🔮 Predict-then-reveal widget (handled by course.js; not scored, not saved)
```html
<div class="predict" data-answer="1">            <!-- 0-based index of the right choice -->
  <span class="label">🔮 Predict first</span>
  <p>Question…</p>
  <div class="choices"><button>A</button><button>B</button><button>C</button></div>
  <div class="reveal">What really happens, and why (point at the playground below).</div>
</div>
```
`🎯` targets: `<span class="goal" id="goal">🎯 Goal: …</span>`; add class `hit` (and change the text) when the learner reaches it.

### Playground rules
- Plain JS on tiny synthetic data, seeded, instant (refit < 100 ms on input; heavier work behind a button). No network, no workers.
- Every control works and changes a live readout. Use `.controls`, `.stat-row` / `.stat`, `.hint`, a message line that **explains what
  the learner is seeing** in words (“the tree built an island around one noisy point…”).
- Colors: use the color language and class colors via CSS vars (`MX.css("--c-c0")`); shapes as well as colors for classes.
- Prototype in node with `ml.js`, and **compute every number you quote in prose with node or python**; the default state of a
  playground should reproduce the numbers in the text (check them with the harness `read`).
- Prose inside JS strings: no raw double quotes; use ‘ ’ or “ ”. Escape any user-visible HTML you build.

## Recap page (`phaseP/recap.html`, id `mlP-r`)
Kicker “Phase P · Recap”, h1, subtitle. Then, in order:
🧾 **Cheat sheet** (`.grid` of `.card`s, one per lesson: `.rule` line with the key rule/number, one line of meaning, `.tools-line`),
🗣️ **Teach-back** (`.box.analogy`, phrased as an interviewer's question, plus a muted “a strong answer covers…”),
🎮 **an interactive game** in a `.playground` (pick the right model for a dataset, diagnose overfitting from curves, match symptom ↔ fix,
spot the leak…; instant feedback, score, reset),
🏗️ **Build it yourself** (`.box.walk` problem with real numbers + `<details class="mathbox">` reveal: requirements, data, features,
model choice with *why*, validation plan, metrics, pitfalls),
💻 **Lab** section (`h2` “💻 Lab P: …”, a `.box.bigidea` “What you'll build”, the file path, setup line, an `<ol>` of the TODOs, and
“Solutions are in `labs/solutions/`”),
🏆 **Boss quiz** of **exactly 5 `.quiz`** mixing all lessons, and a 🌉 **What's next** `.box.bridge`.
Phase 9's recap is the graduation page: an interactive skills map (pick a goal → highlighted lessons), a portfolio/project plan and
where to go next (verified resources, including the sibling `~/deep-learning` course in words until it exists).

## Color language (CSS classes; same meaning in prose, formulas, tables and canvases)
`.ft` data / features x (blue) · `.mo` model / parameters / weights (purple) · `.pd` prediction / score ŷ (orange) ·
`.tg` target / label y (green) · `.lo` error / loss / residual (red) · `.mt` metric / validation score (teal) ·
`.bd` decision boundary / threshold / knob (hyperparameter) (gold). CSS vars `--c-ft --c-mo --c-pd --c-tg --c-lo --c-mt --c-bd`.
Class colors (scatter plots, regions): `--c-c0` blue ●, `--c-c1` pink ■, `--c-c2` green ▲, `--c-c3` violet ◆ (classes `.c0`–`.c3`),
plus `--text --muted --line --accent --good --bad --bg2 --panel`. Uppercase labels capitalize Greek: wrap as `<span class="nt">λ</span>`.

## Clarity rules
- Every acronym and term explained at first use **in each lesson** (CV, AUC, ROC, SMOTE, PCA, GBM, OOB, RMSE, MAPE, TF-IDF, SHAP…).
- No term before it's taught; mention later ideas in plain words + “(Lesson X.Y)”. Short sentences, concrete numbers, small datasets.
- The math stays light: show the formula only when it earns its place, say it in words, and link the math course for the derivation.

## Cross-links (exact files; link only where it truly helps)
From a lesson page: `../../math-course/phaseN/<file>.html`, `../../ai-engineer/phaseN/<file>.html`, `../../rl-course/phaseN/<file>.html`,
`../../system-design-course/...`. List files first (`ls ~/math-course/phase*/`). Math-course lessons you'll likely want:
0/2-functions, 0/3-exp-log · 1/3-dot-product, 1/4-norms, 1/5-embeddings · 2/6-svd-pca, 2/5-eigen · 3/3-gradients, 3/4-gradient-descent ·
4/1-probability, 4/2-bayes, 4/3-distributions, 4/4-expectation, 4/7-lln-clt · 5/1-describing-data, 5/2-correlation, 5/2b-kmeans,
5/3-mle, 5/5-ab-tests, 5/6-overfitting, 5/6b-metrics · 6/2-entropy, 6/3-cross-entropy, 6/5-mutual-info · 7/1-loss-landscapes,
7/2-sgd, 7/4-regularization · 8/1-regression, 8/1b-trees-boosting, 8/2-softmax, 8/3-neural-net.
`~/deep-learning` and `~/autonomous-systems` are being written in parallel: **mention them in words only** (“the sibling Deep
Learning course”), no links yet. Cross-links into this course from siblings will be added later.

## Facts and links
- Check every story, number, quote and API detail against its source (fetch it; confirm the claim is on the page). If you can't, drop it.
- Links: `curl -s -o /dev/null -w '%{http_code}' -L -A 'Mozilla/5.0' URL` → 200. Some sites (microsoft.com) need a full browser UA:
  `-A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'`.
  arXiv: `https://export.arxiv.org/api/query?id_list=…` (check the title). PDFs: download to the scratchpad and `pdftotext`.
- Prefer primary sources: papers, official docs, company engineering blogs, Kaggle competition write-ups by the winners.

## Labs (`labs/phaseN_<topic>.py`, one per phase)
- `# %%` cells (VS Code / Jupyter), a markdown header cell (what you'll build, how to run, lessons to reread), small TODOs that
  `raise NotImplementedError("TODO k: …")`, and `check(cond, msg)` right after each (copy the helpers from `~/math-course/labs/phase8_models.py`:
  `check`, `show`, `close`). Answers in `labs/solutions/` (same filename, header says SOLUTION). 5–8 TODOs.
- **CPU only, seconds to run, offline**: scikit-learn + NumPy + pandas + matplotlib only. Datasets: `sklearn.datasets.make_*` and the
  bundled `load_iris / load_wine / load_breast_cancer / load_diabetes / load_digits`; no `fetch_*` (they download). Small synthetic
  DataFrames built in the file are great. XGBoost/LightGBM only in an optional cell guarded by `try: import xgboost except ImportError:`.
- Must work on Python 3.9 + scikit-learn 1.6 and on Python 3.12 + scikit-learn 1.9 (avoid `match`, `X | Y` type hints, brand-new params).
- Verify both: `cd ~/machine-learning/labs && MPLBACKEND=Agg $VENV/bin/python solutions/<file>.py` with
  `$VENV` = `/private/tmp/claude-501/-Users-vishnu-nemarugommula/885e97f0-4565-423e-89da-97fe4e43ec47/scratchpad/ml312` (3.12, sklearn 1.9.1)
  and `.../scratchpad/mlvenv` (3.9, sklearn 1.6.1). All ✅, < 30 s. Also run the TODO file and confirm it stops at TODO 1.
- `labs/requirements.txt` exists already; don't edit it.

## Glossary
Don't edit `glossary.html`. Write your phase's entries to `tools/glossary/phaseN.json`:
`{"tools": [[name, kind, category, what, when, alternatives, "4.2 4.3"]], "acronyms": [[acr, expansion, meaning, tools, where-used, "lessons"]],
"terms": [[term, meaning, tools, where-used, "lessons"]]}` (kind = one of `ft mo pd tg lo mt bd` or ""). Every term and acronym your
lessons teach, plain English, HTML allowed (`<code>`), lesson numbers like `4.2`.

## Verify (done means)
- Server: `http://127.0.0.1:8771/` serves a folder of symlinks to all sibling courses (check
  `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8771/machine-learning/index.html` → 200; if it's down, tell the lead).
- Harness (puppeteer in `/tmp/rltest`): `cd /tmp/rltest && ~/.nvm/versions/node/v20.9.0/bin/node ml-test.mjs phase4` tests every
  page of a phase: 0 errors, 3 quizzes per lesson / 5 per recap, a `.box.why`, ≥ 2 playgrounds, and auto-clicks every button, slider
  and select in every playground. For specific checks: `node ml-test.mjs '[{"page":"phase4/2-random-forests.html","clicks":[{"id":"reseed"}],"read":["teA"],"shot":"ml42","scroll":"#board"}]'`
  (`clicks` also take `"value"` for sliders or `"select"`), screenshots land in `/tmp/rltest/<shot>.png`: **look at them**.
  Canvas clicks/drags (plane `onClick`, dragging points) aren't covered by the auto-clicker: write a tiny puppeteer script (see
  `/tmp/rltest/ml-canvas2.mjs`); first set `document.documentElement.style.scrollBehavior = "auto"` and `scrollIntoView({block:"center"})`,
  or the smooth scroll makes your click coordinates stale.
- Links: `python3 tools/check-links.py phase4/*.html` (internal) and `--external` (curl every external link).
- Review bank: `python3 tools/build-review-bank.py` (fails loudly on malformed quizzes). Quiz answers are saved by position: don't
  reorder quizzes in published lessons. Insert lessons without renumbering with a letter suffix (`phase4/2b-…`, id `ml4-2b`).
- No heavy jobs on this laptop: no model downloads, no long training, no GPU.

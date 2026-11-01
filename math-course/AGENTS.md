# AGENTS.md: Math for AI course

Static, dependency-free HTML course (no build step; must work from `file://`). Sibling of `~/rl-course`
and built on the same design system. Goal: the math foundations for ML, deep learning, LLMs, RL and data
science, taught for **intuition and use**, not proofs. **The learner is a beginner**: assume school
arithmetic and a little Python, nothing else. Every concept is tied to exactly where and how it's used in AI.

## Files
- `assets/course.js`: `COURSE` manifest (phases, lesson `id`, `title`, `href`, `mins`), sidebar, prev/next,
  progress, quizzes, theme. Also `window.MX` math helpers (below). Storage keys start with `mathcourse.`.
  The manifest is the source of truth for lesson numbers, titles and file names. Don't rename files.
- `assets/style.css`: design system (forked from the RL course + math color language + `.box.code`, `.calc`,
  `.two-col`, `canvas.plane`, `.vec`/`.matrix` inline brackets, domain tags).
- `assets/plot.js`: `Plot.line(canvas, {series:[{label,color,data:[y]|[[x,y]],dash,dots,dotsOnly,fill,width}], xMin,xMax,yMin,yMax,xLabel,yLabel,hlines})`,
  `Plot.bars(canvas, {bars:[{label,value,color,err,marker}], yMin,yMax,yLabel,highlight})`, `Plot.smooth(arr,k)`.
  Use `<canvas class="chart">`. Charts redraw on resize/theme change.
- `assets/plane.js`: `new Plane(canvas, {range, xRange, yRange, snap, step, xLabel, yLabel, onDrag(key,[x,y])})`
  then `.draw(items)` with items `vec` (arrow; `to` is the vector, drawn from `from`; `abs:true` = `to` is the tip),
  `pt`, `seg`, `fn` (y=f(x) curve), `path` (polyline/polygon, `fill`), `text`, `heat` (background f(x,y), e.g. a loss surface).
  Items with `drag: "key"` get a grab handle; `onDrag` gets world coords, update your state and redraw.
  Use `<canvas class="plane">` (square) or `class="plane wide"` (16:9). Colors can be CSS vars: `"--c-x"`.
- `phase1/3-dot-product.html` is the **reference lesson**. Copy its structure, tone and level of interactivity.
- `tools/test-pages.mjs`: headless Chrome harness (see Verify).

## Lesson anatomy (keep this order)
`<body data-lesson="m<P>-<k>" data-root="..">`, title `P.k Title · Math for AI`, same favicon/link tags as the reference.
1. `.kicker` (Phase P · Lesson k) + `h1` + `.subtitle` (one sentence: what + where it's used).
2. `.box.prereq` “🧰 You'll need”: 1–3 one-line reminders with lesson numbers (skip only in lesson 0.1).
3. `.box.hook` “🎯 Hook”: a puzzle or surprise from real AI that this lesson's math explains.
4. Content. **Numbers/example first → plain sentence → equation.** Never an equation before its words.
   Include `.box.bigidea`, `.box.analogy` (everyday picture), `.eq` blocks each with a `.say` line that reads the
   equation out loud in plain English, an optional `details.mathbox` refresher, a `.box.walk` “✏️ Worked example”
   with small numbers done by hand, and **≥ 1 `.playground`** (aim for 2: one to build intuition, one that shows the
   AI use). Playgrounds must be genuinely interactive (drag, sliders, buttons, step/animate) with a live readout
   that shows the arithmetic, not just a picture.
5. `.box.code` “🐍 In code”: 3–10 lines of Python/NumPy (or PyTorch where natural) doing the same thing, with comments.
6. Optional `.box.watch` “📺 Watch” (only verified videos, say what to watch for and how long).
7. `.box.used` “🌍 Where it's used”: 3–5 `.use` rows, one per main concept, diverse domains (LLMs, vision, recsys,
   RL, data science, classic ML, science/industry). Each: bold concept name, a tag, a linked source, then
   **How:** (map the system onto the math) and **Why this method:** (vs the obvious simpler alternative).
   Tags: `<span class="tag llm">🧠 LLMs · 2017</span>`, `tag dl` 🖼️ Deep learning, `tag ml` 🤖 Machine learning,
   `tag rl` 🎮 RL, `tag ds` 📊 Data science, `tag prod` 🏭 In production, `tag hist` 📜 History. Always with a year.
8. `.box.fact` “🤯 Fun fact” (verified).
9. `.box.bridge` “🌉 AI bridge”: how the idea appears in LLMs / deep learning / RL. Link to the RL course lesson that
   uses it when there is one: `href="../../rl-course/phaseN/<file>.html"` (list files with `ls ~/rl-course/phase*/`;
   read the target lesson's intro so the link is accurate). Also point forward to later lessons in this course.
10. `.box.explore` “📚 Explore more”: 2–3 verified links tagged Paper/Blog/Course/Book/Docs/Video with a one-line why.
11. `h2` Check yourself + exactly **3 `.quiz`** (one `button.opt[data-ok]`, 3 options, `.explain` that teaches).
    Make at least one quiz a small calculation and one a “where/why is this used” question.
12. `.box.tldr` “📝 TL;DR”: 3–5 bullets.
Scripts at the end: `../assets/course.js`, then `plot.js` / `plane.js` if used, then one IIFE.
Keep each lesson roughly 12–25 KB. Fun, visual, short paragraphs; not verbose.

## Recap page (`phaseP/recap.html`, id `mP-r`)
Kicker “Phase P · Recap”, h1, subtitle. Then: 🧾 cheat sheet (`.grid` of `.card`s, one per lesson: the formula
in color + one line of meaning + where it's used), 🗣️ teach-back prompt (`.box.analogy`), 🔎 “Spot it in the wild”
(an interactive matching or sorting game: real AI systems ↔ this phase's concepts, with instant feedback),
🧩 one “Use it yourself” problem with a `<details class="mathbox">` reveal (a small realistic task solved with only
this phase's math, numbers worked out), a 🏆 boss quiz of **5 `.quiz`** mixing all lessons, and a 🌉 “What's next”
box bridging to the next phase. The last phase's recap is the graduation page: a map of how every phase feeds
ML / DL / LLMs / RL / data science, plus where to go next (including the RL course).

## Color language (CSS classes; use consistently in prose and equations)
`.x` input/data/vector (blue) · `.w` weights/parameters/matrix (orange) · `.y` output/prediction/target (green) ·
`.pr` probability/distribution (purple) · `.gr` gradient/derivative/change (pink) · `.ls` loss/error (teal) ·
`.lr` learning rate/step size (gold). CSS vars for canvases: `--c-x --c-w --c-y --c-pr --c-gr --c-ls --c-lr`,
plus `--text --muted --line --accent --good --bad`. The RL classes (`.s .a .r .p .q .g .al`) exist for the RL lesson 8.6.
Uppercase `.label` text capitalizes Greek letters: wrap them as `<span class="nt">σ</span>`.

## MX helpers (`window.MX`)
`randn bernoulli randint seeded(seed)→rng linspace(a,b,n) sum mean std dot norm cosine matvec matmul transpose
sigmoid softmax(z,T) entropy(p) [bits] sample(p)→index argmax clamp fmt(v,d) css(varName)`.

## Clarity rules (beginner audience)
- Every symbol gets a plain-English reading at first use in each lesson (Σ, Π, ∂, ∇, ‖x‖, ∈, ≈, ∝, 𝔼, ~, θ, hats, subscripts).
- No term before it's taught; if a later idea is mentioned, say it in plain words + “(Lesson X.Y)”.
- Re-anchor reused concepts with a 5–10 word reminder, not just a lesson number.
- Short sentences. One idea per paragraph. Prefer concrete numbers over letters. No proofs; at most a 2–3 line
  “why it works” with numbers checked alongside.
- Prose inside JS string literals: don't use raw double quotes; use ‘ ’ or “ ”.
- Verify every number shown in a lesson (worked examples, quoted results) with node or python3 before writing it.

## Facts and links
- Every “Where it's used”, fun fact, date and quoted result must be checked against its source. Prefer primary sources.
- arXiv: verify IDs/titles in bulk with `curl -s "https://export.arxiv.org/api/query?id_list=ID1,ID2"`.
- Other links: `curl -s -o /dev/null -w '%{http_code}' -L URL` must give 200 (some sites 403 scripts; confirm by search).
- YouTube: `curl -s "https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v=ID"` returns the real
  title (401 = exists but embedding disabled, 404 = bad ID). Good sources: 3Blue1Brown, StatQuest, Andrej Karpathy.
- If you can't verify a claim, leave it out.

## Verify
- A server for both courses usually runs at `http://127.0.0.1:8766/` (serves `math-course/` and `rl-course/` side by side,
  so `../../rl-course/` links resolve). Check with `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/math-course/index.html`.
- Harness (puppeteer is installed in `/tmp/rltest`):
  `cp ~/math-course/tools/test-pages.mjs /tmp/rltest/math-test.mjs && cd /tmp/rltest && ~/.nvm/versions/node/v20.9.0/bin/node math-test.mjs '[{"page":"phase1/3-dot-product.html","clicks":[{"id":"pPerp"}],"read":["dcalc"]}]'`
  It reports JS errors, failed requests, quiz count, blank canvases and element text. Add `"shot":"name","scroll":".playground"`
  to save `/tmp/rltest/name.png` and look at it. Every page must load with **0 errors**, the right quiz count, and every
  playground button must work.
- Light checks only: no model training or heavy jobs on this laptop.

## Tooling gotcha (seen in the RL repo)
After a file is changed by a shell command (sed/python), later in-editor edits to that same file were sometimes lost.
Prefer one method per file within a session, and confirm on disk (`grep -c "new text" file`).

## Notation conventions (settled while writing phases 0–8; keep new lessons consistent)
- x_i counts from 1 in math, 0 in Python. x̄ mean, s std (÷n; n−1 only as a gotcha), ŷ prediction, θ all knobs, ℝ^d a list of d reals, ln natural log.
- ‖x‖ = L2 norm; ‖x‖₁ “taxi”, ‖x‖∞ “chess king”; distance d(a,b) = ‖a − b‖. Vectors are columns (Wx); PyTorch's xWᵀ is explained in 2.2.
- Update rule w ← w − η·g (η “eta” = `.lr`; the RL course writes α). ∂ “partial”, ∇ “grad”. Momentum m ← βm + g carries into Adam's m, v.
- P(A | B) “given”, X ~ D “drawn from”, 𝔼, Var, SE = s/√n, 95% CI = ± 1.96 SE, H₀ null. L(θ) likelihood, ℓ log-likelihood, NLL loss.
- H(p), H(p, q), D_KL(p‖q), I(X ; Y); LLM losses in nats; perplexity = e^(avg loss); cross-entropy gradient p − y (y one-hot).
- Matrix rank vs tensor “number of axes”; LoRA = W + BA (B d×r, A r×d); A = UΣVᵀ with σ₁ ≥ σ₂ ≥ …; forward ≈ 2N FLOPs/token, training ≈ 6N.
- λ = weight-decay/penalty strength (7.4) and also the constraint multiplier (7.5); RLHF's β is described as that price. Dropout p = drop probability (PyTorch).
- RL return G = Σ γ^(t−1) r_t (rewards start at r₁, as in the RL course). Running example sentence: “The cat sat on the … mat”.

## Inserting lessons
Insert without renumbering by using a letter suffix (file `phase4/6b-importance-sampling.html`, id `m4-6b`, kicker “Lesson 6b”)
and add it to `COURSE` right after its parent. Then add its cheat-sheet card to the phase recap (match that recap's card markup;
don't change recap quiz counts/order) and a one-line pointer from the lesson(s) that lead into it. Added Oct 2026: 3.4b, 4.6b, 5.2b,
5.6b, 8.1b, 8.3b, 8.4b.

## Review bank, glossary, labs
- After adding, editing or moving any quiz, rerun `python3 tools/build-review-bank.py` (regenerates `assets/review-bank.js`,
  the question bank behind `review.html`; it fails loudly on malformed quizzes). Saved review state is `mathcourse.review`.
- New terms or symbols go in `glossary.html` with a link to the lesson that teaches them.
- Labs: `labs/phaseN_*.py` with `# %%` cells, small TODOs raising `NotImplementedError`, `check(cond, msg)` right after each;
  answers in `labs/solutions/` (same filename). Python 3.9, numpy + matplotlib only, seconds to run.
  Verify: `cd labs && MPLBACKEND=Agg python3 solutions/<file>.py` (all ✅). Each recap has a “💻 Lab N” section.

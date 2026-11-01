# AGENTS.md: Autonomous Systems course

Static, dependency-free HTML course (no build step; must work from `file://`). Sibling of `~/rl-course`, `~/math-course`,
`~/system-design-course`, `~/ai-engineer`, `~/machine-learning` and `~/deep-learning`, built on the same design system.
Goal: the **foundations of autonomous vehicles, drones and mobile robots** for a beginner: how a machine perceives 3D space,
knows where it is, predicts others, plans and controls itself, and how **Tesla** (camera-only, end-to-end neural), **Waymo**
(lidar + radar + cameras, HD maps, modular plus learned) and **autonomous drones** (Skydio, PX4/ArduPilot) actually apply each
idea, and why they chose differently. Not a proof course: formulas link to `~/math-course`; deep-learning internals are referenced
in words (see Cross-links). The learner is a working software engineer, new to robotics: **no jargon before it's taught**.

## The two rules that matter most
1. **Beginner-easy, and always WHY.** Never just say “we use X”. Every method answers: what problem existed before, what people
   tried, why X fixes it (plain words + a tiny numeric example + a picture or playground), what it costs, when it fails, what
   refined it. That's the required **🤔 Why this?** box (below).
2. **Curious, interactive, fun.** Same look and box anatomy as the sibling courses, and while reading the learner should feel
   curiosity and momentum:
   - **Open with a puzzle**, surprising result or “what would you do?” question that the lesson then answers.
   - **Predict, then reveal**: ask for a guess before a playground or a `.predict` box shows the answer.
   - **Interactive throughout**, not two playgrounds at the end: a playground or small challenge near **every big idea**
     (aim for 3; never fewer than 2). Let the learner break things and fix them; add mini-games with **score targets**
     (“park with fewer than 3 corrections”, “keep the drone within 10 cm”, “beat raw GPS by 2×”) and live numbers that
     update while they drag.
   - **Show the payoff early** (who uses this, what it powers). Light, friendly, conversational voice; short sections, small
     wins, fun facts. Never a wall of text: if a paragraph runs past ~5 lines, split it or turn it into a table, list or picture.

## The field moves monthly: verify everything
Today is late 2026. Company approaches, fleet sizes, cities, software versions and papers change constantly. Before writing any
company fact, **search the web and confirm it on a primary source** (waymo.com/blog, waymo.com/safety, waymo.com/research,
tesla.com support/AI pages or Tesla's own talks, docs.px4.io, ardupilot.org, skydio.com, arXiv, official docs). Date volatile facts
(“as of Oct 2026”, “in its Feb 2026 post”). Never invent a number, score, quote, version or date. Tesla publishes little about its
internals: say what Tesla has **publicly said** (AI Day 2021/2022, CVPR talks, release notes) and label it as such; when something
isn't public, say so. Shared, already-verified facts are in `tools/facts.md`; reuse them and add to them.

## Files
- `assets/course.js`: `COURSE` manifest (phases `s0`–`s9`, lesson `id`, `title`, `href`, `mins`), sidebar, prev/next, progress,
  quizzes, theme, and `window.MX` helpers: `randn bernoulli randint seeded(seed)→rng linspace sum mean std dot norm cosine matvec
  matmul transpose sigmoid softmax entropy sample argmax clamp fmt(v,d) css(var)` plus, for filters and geometry,
  `eye(n) madd msub mscale inv(M) gauss(x,mu,sigma) wrap(angle) deg rad lerp`. Storage keys start with `ascourse.`.
  **The manifest is the source of truth** for titles, ids and file names.
- `assets/sim.js`: **2D world for playgrounds** (meters, y up): `new Sim(canvas, {xRange, yRange, grid, onPointer})`, then
  `W.render(w => { w.clear(); ... })` with `rect circle line arrow dot text ellipse(cov) car robot drone obstacles rays cells`,
  `W.run((dt,t)=>…)`/`W.stop()`. Static: `Sim.rng(seed) Sim.randn(rng) Sim.raycast Sim.lidar Sim.collides Sim.bicycle Sim.unicycle
  Sim.covEllipse Sim.dist Sim.wrap`. Read its header. `<canvas class="sim">` (16:9), `sim square`, `sim tall` (4:3).
  A render fn may run before your state exists (it runs on construction and on resize): guard with `if (!state) return;`.
- `assets/space.js`: **3D math + orbit view**: `SP.rotX/rotY/rotZ mul apply T euler(yaw,pitch,roll) toEuler`, quaternions
  `SP.q.fromAxisAngle fromEuler mul conj normalize toMatrix fromMatrix rotate slerp angle`, transforms `SP.hom hmul happly hinv`,
  `SP.project(K, p)`. `new View3D(canvas, {yaw, pitch, dist, target, fov})` with `clear grid line arrow point poly text axes box`;
  drag to orbit. Convention: x forward, y left, z up (ROS REP 103); camera frames are z forward, x right, y down.
- `assets/plane.js`: 2D math plane with draggable vectors/points (`new Plane(canvas, {range, onDrag})`, items `vec pt seg fn path
  text heat`), from the Math course. `<canvas class="plane">`.
- `assets/plot.js`: `Plot.line(canvas, {series:[{label,color,data:[y]|[[x,y]],dash,dots,dotsOnly,fill,width}], xMin,xMax,yMin,yMax,
  xLabel,yLabel,hlines})`, `Plot.bars(canvas, {bars:[{label,value,color}], yMin,yMax})`. `<canvas class="chart">`.
- `assets/flow.js`: animated pipeline diagrams (`new Flow(svg, {width,height}).node(id,{x,y,w,h,label,sub,kind}).edge(a,b).render()`,
  `.set`, `.edgeStyle`, `await .packet([...])`). Autonomy kinds: `se be un mp ag pl ct` (colors below). Great for the stack diagram.
- `phase3/3-kalman-filter.html` is the **reference lesson**. Copy its structure, tone, box order and level of interactivity
  (predict-then-reveal, three playgrounds, score challenges, `.predict` boxes, the 🤔 and 🚗 boxes).
- `tools/test-pages.mjs` (headless harness), `tools/build-review-bank.py` (quiz bank for `review.html`), `tools/facts.md`
  (verified, dated facts with sources).

## Lesson anatomy (keep this order)
`<body data-lesson="s<P>-<k>" data-root="..">`, title `P.k Title · Autonomous Systems`, same head tags as the reference.
1. `.kicker` + `h1` + `.subtitle`.
2. `.box.prereq` “🧰 You'll need (quick reminders)” (skip only in 0.1).
3. `.box.hook` “🎯 Hook”: a **puzzle or surprising question first**, then a real, verified story or number that this lesson explains.
4. An early playground or `.predict` box that lets the learner commit to a guess (predict, then reveal).
5. Content: **numbers first → plain sentence → picture/formula**. `.box.walk` “✏️ Worked example” by hand (meters, seconds,
   pixels, variances…), `.box.bigidea`, `.box.analogy`, tables, `.eq` blocks with the color classes.
6. `.box.why` “🤔 Why this?” as an `<ol>` with exactly these five items, in order, each starting with a bold lead:
   **The problem before.** · **What people tried.** · **Why this works.** (intuition + a tiny numeric example) ·
   **What it costs, when it fails.** · **What came next.** Put it right after the core idea is shown (as in the reference).
   Example (Kalman): GPS alone jumps ±5 m and an IMU alone drifts; blend them weighted by trust: variances 25 and 4 → gain
   25/(25+4) ≈ 0.86 → fused variance ≈ 3.4 m²; fails with nonlinear motion → EKF, UKF, particle filters.
7. **≥ 2 (aim for 3) `.playground`s**, each next to the idea it teaches, genuinely interactive: sliders, drag, draw, click; live
   stats; a 🏆 challenge with a target and a win badge; a way to cause the failure and see the fix. Plus `.predict` boxes
   (`<div class="predict" id="…">…<button data-open="…">Show the answer</button><div class="ans">…</div></div>`, CSS in the reference).
8. `.box.tools` “🧰 Tools you'll actually use”: real libraries with a short, correct, current snippet (Python/NumPy, OpenCV,
   Open3D, ROS 2 `rclpy`/`tf2`, PX4/MAVSDK, FilterPy, GTSAM, CARLA…). Check the API against the docs.
9. `.box.howco` “🚗 How Tesla / Waymo / drones do it” wherever it applies (almost always): `.cols` with three `<div>`s
   (🚘 Tesla, 🚕 Waymo, 🛸 Drones; add 🤖 robots/others if useful), each ending in `<p class="src">Sources: …</p>` with primary
   links, then `<p class="why-diff"><b>Why they differ:</b> …</p>`. If a company hasn't published how it does something, say so.
10. `.box.tradeoff` “⚖️ Trade-offs”. (This course has **no** “Startup vs big tech” box; the 🚗 box replaces it.)
11. `.box.interview` “🎤 Interview angle” (what robotics / AV interviewers listen for).
12. `.box.used` “🏭 In the real world”: 3–5 `.use` rows with **How:** / **Why this method:** and a linked primary source.
    Tags: `tag co` 🏢 Company · year, `tag res` 🧪 Research · year, `tag oss` 🧰 Open source, `tag hist` 📜 History, `tag av` 🚗 AV.
13. `.box.fact` “🤯 Fun fact” (verified). 14. `.box.bridge` “🌉 How it connects” (lessons by number + sibling courses).
15. `.box.explore` “📚 Explore more” (2–3 verified links). 16. **Exactly 3 `.quiz`** (one `data-ok`, 3 options, a teaching
    `.explain`; one calculation, one scenario). 17. `.box.tldr` “📝 TL;DR”.
Scripts at the end: `../assets/course.js`, then `plot.js` / `sim.js` / `space.js` / `plane.js` / `flow.js` as needed, then one IIFE.
Lessons are typically 35–55 KB.

## Recap page (`phaseP/recap.html`, id `sP-r`)
In this order: cheat-sheet cards (one per lesson: key rule/number + one line + tools); a **teach-back** prompt phrased as an
interviewer's question with a model answer in `.muted`; an **interactive game** (e.g. tune a PID to stop oscillation, place sensors
to cover blind spots, fix a diverging filter, match symptom ↔ cause) with a score; a 🏗️ **Build it yourself** problem with a
`<details class="mathbox">` reveal (requirements, numbers, design, pitfalls); a **💻 Lab N** section pointing to that phase's
Python lab (what you'll build, file name, how to run, one line per TODO, “Solutions are in `labs/solutions/`”); a boss quiz of
**5 `.quiz`**; a 🌉 “What's next” box. Phase 9's recap is the graduation page: career paths (perception, localization/mapping,
planning, controls, simulation, safety), a skills map, a portfolio plan, and verified resources (ROS 2, CARLA, PX4, courses).

## Color language (CSS classes and vars)
`.se` sensor / raw measurement (blue) · `.be` estimate / belief / state (purple) · `.un` uncertainty / noise / covariance (gold) ·
`.mp` map / static world / free space (green) · `.ag` other agents / obstacles / risk (red) · `.pl` plan / trajectory / goal (teal) ·
`.ct` control / actuation / command (orange). Vars `--c-se --c-be --c-un --c-mp --c-ag --c-pl --c-ct` (+ `--text --muted --line
--accent --good --bad`). Use them in prose, tables, `.eq` blocks, Sim drawings and Flow nodes so a picture reads at a glance.
3D axes: x red, y green, z blue (the ROS/RViz convention; `View3D.axes` does this).

## Clarity rules
- Every acronym and term is explained at first use **in each lesson** (IMU, GNSS, RTK, SLAM, EKF, BEV, MPC, LQR, ODD, VIO, ICP…).
- No term before it's taught; mention later ideas in plain words + “(Lesson X.Y)”. Short sentences, concrete numbers, one idea per
  paragraph. Re-anchor reused ideas with a 5–10 word reminder, not just a lesson number.
- Uppercase `.label` text capitalizes Greek letters: wrap them as `<span class="nt">σ</span>`.
- Label typical numbers as typical (“a typical automotive radar…”); sensor prices as illustrative unless sourced with a date.
- Prose inside JS string literals: no raw double quotes; use ‘ ’ or “ ”.
- **Verify every number you compute** with node or python3, and **prototype playground behaviour in node** before claiming it in
  prose (the reference lesson's tunnel challenge was retuned twice after simulation). Make sure every 🏆 target is reachable and
  that the default settings don't already win.

## Cross-links (list files to get exact names; only link files that exist)
- Math course (`../../math-course/…`): phase1 `1-vectors 2-vector-arithmetic 3-dot-product 4-norms`, phase2 `1-matrices 2-matmul
  4-rank 5-eigen 6-svd-pca`, phase3 `1-derivatives 2-chain-rule 3-gradients 4-gradient-descent 4b-taylor-curvature 6-integrals`,
  phase4 `1-probability 2-bayes 3-distributions 4-expectation 5-joint-chain 6-sampling 7-lln-clt`, phase5 `3-mle 4-bayesian`,
  phase7 `1-loss-landscapes 5-constraints`, phase8 `3b-convolution 4-attention 4b-diffusion`. (All `.html`.)
- RL course (`../../rl-course/…`): `phase2/2-mdps.html`, `phase6/3-model-based.html`, `phase6/3b-mcts.html`,
  `phase6/7-imitation.html`, `phase6/4-offline-rl.html`, `phase6/6-self-play.html`.
- Deep Learning course: **reference in words with a marker**, which the course owner converts to a link at the end if the file
  exists: `<span class="dlref" data-href="phase4/8-3d-vision.html">the Deep Learning course, Phase 4 (3D vision basics)</span>`.
  Useful lessons: d3 CNNs (`phase3/2-convolution.html`), d4-1 detection basics (`phase4/1-detection-basics.html`), d4-2 detectors
  (`phase4/2-detectors.html`), d4-3 segmentation (`phase4/3-segmentation.html`), d4-5 ViT (`phase4/5-vit.html`), d4-7 pose, video
  & tracking (`phase4/7-pose-video.html`), d4-8 3D vision (`phase4/8-3d-vision.html`), d6-1 self-attention
  (`phase6/1-self-attention.html`), d6-4 transformer block (`phase6/4-transformer-block.html`), d7-4 diffusion
  (`phase7/4-diffusion.html`), d7-8 autoregressive video (`phase7/8-autoregressive-visual.html`), d9-1 self-supervised
  (`phase9/1-self-supervised.html`), d9-6 uncertainty & OOD (`phase9/6-uncertainty-ood.html`), d10-4 quantization
  (`phase10/4-efficiency.html`), d10-5 TensorRT/deploy (`phase10/5-deploy.html`), d1-1 losses (`phase1/1-losses.html`).
- Machine Learning course: mention in words (“the Machine Learning course”) without links.

## Facts and links
- Check every story, number, quote and API detail against its source (fetch it; confirm the claim is on the page). If you can't, drop it.
- Links: `curl -s -o /dev/null -w '%{http_code}' -L -A 'Mozilla/5.0' URL` → 200. arXiv: `https://export.arxiv.org/api/query?id_list=…`
  (check the title). YouTube: `https://www.youtube.com/oembed?format=json&url=…`.

## Labs (`labs/`)
`labs/phaseN_<topic>.py` with `# %%` cells, small TODOs raising `NotImplementedError`, `check(cond, msg)` right after each, a
`# %% [markdown]` header (what you'll build, setup, how to run, lessons to reread) and a closing “🔬 Try it yourself”. Answers in
`labs/solutions/` (same filename). Python 3.9, **NumPy only** (matplotlib optional, guarded with the `show()` helper so it runs
headless); CPU-only; **the whole file runs in seconds**. Verify: `cd labs && MPLBACKEND=Agg python3 solutions/<file>.py` → all ✅,
and the unsolved file stops at TODO 1 with `NotImplementedError`. Each recap has a “💻 Lab N” section. Copy the pattern of
`~/math-course/labs/phase4_probability.py`.

## Verify
- Server: `http://127.0.0.1:8773/` serves a scratch `www` dir with symlinks to every course, so check
  `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8773/autonomous-systems/index.html`.
- Harness (puppeteer in `/tmp/rltest`): `cp ~/autonomous-systems/tools/test-pages.mjs /tmp/rltest/as-test.mjs && cd /tmp/rltest &&
  ~/.nvm/versions/node/v20.9.0/bin/node as-test.mjs '[{"page":"phase3/3-kalman-filter.html","clicks":[{"id":"lock"},{"id":"go","wait":3000}],"read":["eK","t2"]}]'`.
  Add `"shot":"name","scroll":"#id"` for `/tmp/rltest/name.png` (then look at it). Every page: **0 errors**, the right quiz count,
  every button working. Click-test every playground and read its stats.
- No heavy jobs on this laptop: no GPU, no model downloads, no long training, no big simulators.

## Review bank
After adding/editing/moving quizzes, rerun `python3 tools/build-review-bank.py`. Quiz answers are saved by position; don't reorder
quizzes in published lessons. Insert lessons without renumbering with a letter suffix (`phase4/2b-…`, id `s4-2b`).

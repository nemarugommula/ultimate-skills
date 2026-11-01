# AGENTS.md: System Design at Scale course

Static, dependency-free HTML course (no build step; must work from `file://`). Sibling of `~/rl-course` and `~/math-course`,
built on the same design system. Goal: take a developer who can implement features but has never designed systems, and make
them able to design, deploy and scale systems for millions of users: as a senior/lead engineer at big tech, or a founding
engineer at a startup. Covers architecture, data, async systems, microservices, Docker/Kubernetes/cloud, CI/CD, observability
(Prometheus/Grafana/OpenTelemetry), incidents, the classic design problems, and senior-level leadership/interviews.
**Teach for intuition and real use**: numbers first, simulators you can break, the tools companies actually run, trade-offs,
and what interviewers listen for. The learner is a working developer but new to system design: no jargon before it's taught.

## Files
- `assets/course.js`: `COURSE` manifest (phases `s0`–`s8`, lesson `id`, `title`, `href`, `mins`), sidebar, prev/next, progress,
  quizzes, theme, and `window.MX` helpers (randn, seeded(seed)→rng, linspace, sum, mean, std, clamp, fmt(v,d), css(var), sample(p)…).
  Storage keys start with `sdcourse.`. The manifest is the source of truth for lesson numbers, titles and file names.
- `assets/style.css`: design system + system-design color language and boxes (below).
- `assets/flow.js`: `new Flow(svg, {width, height})` → `.node(id, {x, y, w, h, label, sub, kind})` (x, y = center; kind
  `cl sv db nw ca er lt`) → `.edge(a, b, {dashed, label, noArrow})` → `.render()`; then `.set(id, {sub, label, down, hot, load 0..1})`,
  `.edgeStyle(a, b, {color, width, dashed, hidden, label})`, `await .packet([ids…], {color, dur, back, r})` animates a request
  dot along the path (back: true = it returns, like a response). Use `<svg id="…"></svg>`; it scales to width.
- `assets/plot.js`: `Plot.line(canvas, {series:[{label,color,data:[y]|[[x,y]],dash,dots,fill}], xMin,xMax,yMin,yMax,xLabel,yLabel,hlines})`,
  `Plot.bars(canvas, {bars:[{label,value,color}], yMin,yMax})`. Use `<canvas class="chart">`.
- `phase1/3-caching.html` is the **reference lesson**. Copy its structure, tone and level of interactivity.
- `tools/test-pages.mjs`: headless Chrome harness (see Verify). `tools/build-review-bank.py`: quiz bank for `review.html`.

## Lesson anatomy (keep this order)
`<body data-lesson="s<P>-<k>" data-root="..">`, title `P.k Title · System Design at Scale`, same head tags as the reference.
1. `.kicker` (Phase P · Lesson k) + `h1` + `.subtitle` (one sentence: what it is + why it matters at scale).
2. `.box.prereq` “🧰 You'll need”: 1–3 one-line reminders with lesson numbers (skip only in lesson 0.1).
3. `.box.hook` “🎯 Hook”: a real, verified story/number/outage from a known company.
4. Content. **Numbers first → plain sentence → diagram/formula.** Back-of-the-envelope math with realistic, clearly-labelled
   “typical” numbers. Include `.box.bigidea`, `.box.analogy` (everyday picture), `.box.walk` “✏️ Worked example” with numbers done
   by hand, tables comparing options, and **≥ 2 `.playground`s**: genuine simulators (traffic, failures, timing) with live
   numbers, built with Flow (animated architecture) and/or Plot. Each should let the learner *cause* a failure and see the fix.
   Optional `.box.incident` “🔥 Real outage” (verified postmortem).
5. `.box.tools` “🧰 Tools you'll actually use”: the real tools for this topic (open source + managed cloud versions) with a short
   config/code/CLI snippet in `<pre><code>` (Dockerfile, k8s YAML, nginx.conf, redis-cli, SQL, Terraform, PromQL…) and a
   “try it locally” line when it's cheap (`docker run …`).
6. `.box.tradeoff` “⚖️ Trade-offs”: when to use / when not, costs, alternatives.
7. `.box.startup` “🚀 Startup vs big tech” with `.cols` (Founding engineer 0 → 1 / Big tech 1 → 100).
8. `.box.interview` “🎤 Interview angle”: what a senior answer covers; common follow-up questions.
9. `.box.used` “🏭 In the real world”: 3–5 `.use` rows (bold concept, tag, linked primary source, **How:** and **Why this method:**).
   Tags: `<span class="tag co">🏢 Company · 2019</span>`, `tag oss` 🧰 Open source, `tag hist` 📜 History, `tag prod` 🏭 In production.
   Prefer companies' own engineering blogs, papers, official docs and public postmortems. Diverse companies.
10. `.box.fact` “🤯 Fun fact” (verified). 11. `.box.bridge` “🌉 How it connects” (earlier/later lessons by number; AI systems where natural).
12. `.box.explore` “📚 Explore more”: 2–3 verified links tagged Paper/Blog/Docs/Book/Video/Course with a one-line why.
13. `h2` Check yourself + exactly **3 `.quiz`** (one `button.opt[data-ok]`, 3 options, an `.explain` that teaches). At least one
    estimation question and one “which design/why” scenario.
14. `.box.tldr` “📝 TL;DR”: 3–5 bullets.
Scripts at the end: `../assets/course.js`, then `plot.js` / `flow.js` if used, then one IIFE. Lessons are typically 25–40 KB.

## Recap page (`phaseP/recap.html`, id `sP-r`)
Kicker “Phase P · Recap”, h1, subtitle. Then: 🧾 cheat sheet (`.grid` of `.card`s, one per lesson: the key rule/number + one line
+ the tools), 🗣️ teach-back prompt (`.box.analogy`, phrased like an interviewer's question), 🔎 an interactive game (e.g. “Spot the
bottleneck”, match symptoms ↔ fixes, or order the steps of a request) with instant feedback, 🏗️ “Design it yourself”: one realistic
design problem using only this phase's ideas, with a `<details class="mathbox">` reveal (requirements, numbers, architecture,
trade-offs, classic mistakes), a 🏆 boss quiz of **5 `.quiz`**, and a 🌉 “What's next” box. Phase 8's recap is the graduation page:
an interactive map of the whole course, a 30/60/90-day plan for a new senior/founding role, and where to go next (verified books,
courses, engineering blogs).

## Color language (CSS classes; use consistently in prose, tables, diagrams)
`.cl` client/user/request (blue) · `.sv` service/compute/server (orange) · `.db` database/storage (green) ·
`.nw` network/queue/message (purple) · `.ca` cache/CDN/fast path (teal) · `.er` failure/error/overload (red) · `.lt` latency/time/cost (gold).
CSS vars: `--c-cl --c-sv --c-db --c-nw --c-ca --c-er --c-lt` plus `--text --muted --line --accent --good --bad`. Flow `kind`s use the same letters.
Uppercase `.label` text: wrap lowercase-sensitive bits like `<span class="nt">k8s</span>` if needed.

## Clarity rules
- Every acronym and term is spelled out and explained at first use in each lesson (QPS, p99, TTL, L7, RPC, SLO, ACID, HPA, CRD…).
- No term before it's taught; mention later ideas in plain words + “(Lesson X.Y)”. Re-anchor reused concepts with a 5–10 word reminder.
- Short sentences, one idea per paragraph, concrete numbers. Label typical numbers as typical (“about”, “typically”); never invent
  precise vendor benchmarks.
- Prose inside JS string literals: no raw double quotes; use ‘ ’ or “ ”.
- Verify every number you compute (worked examples, simulator claims) with node or python3.

## Facts and links
- Every company story, number, date and quote must be checked against its source (engineering blog, paper, official docs, postmortem).
  Fetch it with curl and confirm the claim is actually there. If you can't verify it, leave it out.
- Links: `curl -s -o /dev/null -w '%{http_code}' -L -A 'Mozilla/5.0' URL` should give 200 (some sites block scripts; confirm by search).
  arXiv: `https://export.arxiv.org/api/query?id_list=…`. YouTube: `https://www.youtube.com/oembed?format=json&url=…`.

## Verify
- Server for the sibling courses: `http://127.0.0.1:8766/` serves `system-design-course/` (check with
  `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/system-design-course/index.html`).
- Harness (puppeteer in `/tmp/rltest`): `cp ~/system-design-course/tools/test-pages.mjs /tmp/rltest/sd-test.mjs && cd /tmp/rltest &&
  ~/.nvm/versions/node/v20.9.0/bin/node sd-test.mjs '[{"page":"phase1/3-caching.html","clicks":[{"id":"run"}],"read":["hr"]}]'`.
  Reports JS errors, failed requests, quiz count, blank canvases, element text; add `"shot":"name","scroll":".playground"` to save
  `/tmp/rltest/name.png`. Every page: **0 errors**, right quiz count, every playground button working.
- Light checks only: no heavy jobs (don't run Docker/Kubernetes clusters) on this laptop.

## Review bank
After adding, editing or moving any quiz, rerun `python3 tools/build-review-bank.py` (regenerates `assets/review-bank.js`).
Quiz answers are saved by position (`lessonId#index`); don't reorder quizzes in published lessons.
To insert a lesson without renumbering, use a letter suffix (`phase3/2b-…html`, id `s3-2b`) and add it to `COURSE`.

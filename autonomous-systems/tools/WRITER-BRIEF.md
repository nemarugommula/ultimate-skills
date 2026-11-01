# Writer brief (common to every phase writer)

You are writing one phase of the **Autonomous Systems** course at `~/autonomous-systems` (static HTML, no build step).

## Read first (in this order, fully)
1. `~/autonomous-systems/AGENTS.md`: the rules. Follow every one: lesson anatomy and box order, the two required boxes
   (🤔 Why this? with its five items, 🚗 How Tesla / Waymo / drones do it), the curiosity/interactivity rules, color language,
   clarity rules, cross-link rules, facts policy, labs, verification.
2. `~/autonomous-systems/phase3/3-kalman-filter.html`: the **reference lesson**. Match its look, tone, box order, CSS components,
   `.predict` boxes, three playgrounds with 🏆 challenges and win badges, live stats, and the quality of its 🤔 and 🚗 boxes.
   (It defines `.predict`, `.badge-win`, `.badge-lose` and `canvas.strip` in its own `<style>`; copy what you use.)
3. The headers of `assets/sim.js`, `assets/space.js`, `assets/plot.js`, `assets/plane.js`, `assets/flow.js` and the `MX` helpers
   at the bottom of `assets/course.js`; the manifest in `assets/course.js` (ids, titles, hrefs are fixed: use them exactly).
4. `tools/facts.md` (already-verified, dated facts) and, for the feel of the sibling courses, skim one lesson each of
   `~/ai-engineer/phase3/3-chunking.html`, `~/math-course/phase4/2-bayes.html` and `~/rl-course/phase6/7-imitation.html`, and one
   recap (`~/ai-engineer/phase3/recap.html`). For the lab, copy the pattern of `~/math-course/labs/phase4_probability.py` and its
   solution file.

## The user's voice (why this course exists)
A working software engineer teaching themself AI from the ground up asked for the fundamentals behind spatial awareness and what
Tesla FSD, Waymo and autonomous drones do. Their strongest request: **very easy and intuitive for a beginner**. Never “we use X”:
always the problem before, what people tried, why X fixes it (plain words, a tiny numeric example, a picture or playground), what it
costs, when it fails and what refined it. And: “it should be as interactive and in the same format as our existing courses, so that
while reading we have curiosity, learning motivation and fun.” Open each lesson with a puzzle, use predict-then-reveal, put a
playground or challenge near every big idea, add score targets, show the real-world payoff early, keep a light friendly voice.

## What you deliver for your phase
- Every lesson file of your phase listed in the manifest (exact href), each a full lesson per AGENTS.md (typically 35–55 KB),
  with ≥ 2 (aim for 3) genuinely interactive playgrounds built on `sim.js` / `space.js` / `plot.js` / `plane.js` / `flow.js`,
  exactly 3 quizzes, and the 🤔 and 🚗 boxes.
- `phaseN/recap.html` (id `sN-r`) per the recap anatomy: cheat sheet, teach-back, interactive game with a score, 🏗️ build-it-yourself
  with a reveal, 💻 Lab N section, exactly 5 boss quizzes, 🌉 what's next.
- `labs/phaseN_<topic>.py` and `labs/solutions/phaseN_<topic>.py` (NumPy only, runs in seconds; all ✅ in the solution; the unsolved
  file stops at TODO 1 with NotImplementedError).
- `tools/glossary/phaseN.json`: a JSON array of the terms, acronyms and tools your phase teaches, each
  `{"term": "EKF", "kind": "acronym"|"term"|"tool", "expansion": "Extended Kalman filter" (acronyms only, else ""),
    "meaning": "plain-English meaning, 1–2 sentences", "tools": "libraries/products that implement it, or \"\"",
    "used": "where it matters (e.g. PX4 EKF2, Waymo, Tesla)", "lessons": "3.4 3.6"}`. Aim for 25–50 entries.
- Append newly verified, reusable facts to `tools/facts.md` under a `## Phase N additions` heading, with dates and source URLs
  (append only, e.g. with `cat >> tools/facts.md <<'EOF'`; other writers append too, so never rewrite the file).

## Hard rules
- **Only write inside your own phase folder, your own lab files and your own glossary JSON** (plus appending to facts.md).
  Other writers are editing the other phases in parallel. Never edit `assets/`, `AGENTS.md`, other phases, or any sibling course.
  If you need a helper that's missing from `assets/`, write it inside your lesson's script.
- **Verify every fact on a primary source with the web tools** (WebSearch, WebFetch) and every link with curl (200). Date volatile
  facts. Tesla: only what Tesla has publicly said or shipped, labelled as such. If you can't verify it, leave it out.
- **Verify every number** you compute with node or python3, and **prototype playground behaviour in node first** so every 🏆
  target is reachable and the defaults don't already win. Keep playgrounds light (no heavy loops on page load).
- Deep Learning course references use the `dlref` marker from AGENTS.md (words + data-href); Math and RL course links must point to
  files that exist (list the folders).
- No heavy jobs: no GPU, no model downloads, no installs beyond `pip install --user filterpy`-sized packages (prefer none), no long
  training, no big simulators.

## Verify before you report (done means)
1. Harness every page of your phase (server already running on 127.0.0.1:8773):
   `cp ~/autonomous-systems/tools/test-pages.mjs /tmp/rltest/as-test-<phase>.mjs && cd /tmp/rltest && ~/.nvm/versions/node/v20.9.0/bin/node as-test-<phase>.mjs '<json>'`
   with clicks on every button and reads of the live stats. 0 errors on every page; 3 quizzes per lesson, 5 per recap.
   Take at least one screenshot per lesson (`"shot":"as-pN-k","scroll":"#someCanvas"`) and look at it (Read the png); fix anything
   that looks broken, empty, overlapping or unreadable.
2. curl every external link (200, or a documented reason such as a site that blocks bots but works in a browser).
3. `cd ~/autonomous-systems/labs && MPLBACKEND=Agg python3 solutions/<your lab>.py` → all ✅ in seconds.
4. `python3 -c "import json;json.load(open('tools/glossary/phaseN.json'))"` parses.

## Your report (final message, short)
Files written, page sizes, harness result per page, lab result, anything you could not verify or left out, and any helper you wish
existed in `assets/`. If you hit an API spend/rate limit (HTTP 429) mid-way, stop cleanly and say exactly which files are finished.

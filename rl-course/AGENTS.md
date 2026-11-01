# AGENTS.md: RL from Zero course

Static, dependency-free HTML course on RL (no build step; must work from `file://`).
The learner is an experienced software dev who has forgotten most school math. Teach
intuition first, then a little math, then code. Keep it fun and visual, not verbose.

## Adding a lesson / phase
1. Add or modify the phase in `COURSE` in `assets/course.js` (`status: "ready"`, lessons with
   `id`, `title`, `href`, `mins`). The sidebar, home roadmap, prev/next and progress all derive from it.
2. Create `phaseN/<k>-<slug>.html` by copying an existing lesson. `<body data-lesson="pN-k" data-root="..">`,
   load `../assets/course.js` (and `../assets/plot.js` if charts are used).
3. Lesson anatomy (keep this order): kicker + h1 + subtitle → `.box.hook` → content with
   `.box.bigidea`, `.box.analogy`, `details.mathbox` (Math refresher, plain-English), `.eq` equations
   with a `.say` line, ≥1 `.playground` → `.box.used` (🌍 Where it's used) → `.box.fact` → `.box.bridge` (LLM connection) →
   3 `.quiz` (one `button.opt[data-ok]` + `.explain`) → `.box.tldr`.
   `.box.used` = one `.use` per main concept: a tag (`.tag.prod` 🏭 in production / `.tag.res` 🧪 research or
   “inside” / `.tag.hist` 📜 history, with a year) + linked source, **How** (map the system onto state/action/reward)
   and **Why this method** (vs the obvious simpler option). Every row verified against its source; diverse domains.
4. Color language (CSS classes): `.s` state, `.a` action, `.r` reward, `.p` policy, `.q` value,
   `.g` gamma, `.al` alpha. Use them consistently in equations and prose.
5. Each phase ends with `recap.html`: cheat sheet cards, teach-back prompt, 🗂️ Case files, 🧩 Frame it yourself,
   lab instructions, boss quiz, and a "what's next" bridge. Case files = the same five problems in every recap
   (cooling plant, feed, robot hand, board game, coding agent), introduced on `cases.html`; keep each recap's
   cards consistent with the per-phase arc on that page. Frame it yourself = one new real problem with a
   `<details>` reveal (state/action/reward/γ, method, classic mistakes), using only methods taught so far.
   Root reference pages: `cases.html`, `chooser.html` (method decision tree), `connections.html` (neighbouring fields).
6. To insert a lesson without renumbering, use a letter suffix (e.g. `phase6/3b-mcts.html`, id `p6-3b`).
   Quiz answers are saved by position (`lessonId#index`), so reordering or moving quizzes remaps saved answers:
   add an entry to `QUIZ_MOVES` in `assets/course.js` (next `v`) so learners' answers follow the questions.
   A phase can be split without renumbering lessons: `num` may be a string (Phase 8 = `"8A"` 8.1–8.9 + `recap.html`,
   `"8B"` 8.10–8.15 + `recap-agents.html`, both in `phase8/`); recap ids must end in `-r` (phase-complete toast).
8. After adding, editing or moving any quiz, rerun `python3 tools/build-review-bank.py` (regenerates
   `assets/review-bank.js`, the question bank behind `review.html`). It fails loudly on malformed quizzes.
7. New terms or symbols go in the array in `glossary.html`. Phase 7–8 lessons carry `verified: "YYYY-MM"`
   in `COURSE`; bump it when you re-check a lesson's claims.

## Shared gridworld (Phase 2+)
`assets/gridworld.js` (`window.GW`) is the Phase 0 world as an MDP: `GW.model({slip, step})`,
`evalSweep`/`viSweep`/`greedy`/`evaluate`/`valueIteration`/`rollout`, and `new GW.Grid(el).draw({V, pi})`
for heatmap + policy arrows. Numbers must match `labs/solutions/phase2_dynamic_programming.py`
(e.g. random policy γ=0.9 V(start)=−9.86; optimal γ=0.9 V(start)=−0.43; slip 0.2 γ=1 V*(start)=+0.03).
Verify any number quoted in a lesson with node or the lab solution before writing it.

## Labs
- `labs/phaseN_*.py` uses `# %%` cells, TODOs that raise `NotImplementedError`, and `check(cond, msg)`
  self-checks right after each TODO. Full answers in `labs/solutions/` (same filename).
- Python 3.9 compatible (no `match`, no `X | Y` types). Deps: numpy, matplotlib only (until deep RL phases).
- Mac, no NVIDIA GPU: keep labs CPU-fast; LLM labs (Phases 7–8) target Google Colab.

## Verify
- Pages (automated): `tools/test-pages.mjs` loads pages in headless Chrome, clicks playground buttons by id,
  and reports JS errors / HTTP failures / quiz counts / element text. Setup once:
  `mkdir -p /tmp/rltest && cd /tmp/rltest && npm init -y && npm i puppeteer-core@25.12.0`, then
  `cp ~/rl-course/tools/test-pages.mjs /tmp/rltest/ && node /tmp/rltest/test-pages.mjs '[{"page":"phase5/6-ppo.html","clicks":[{"id":"run"}],"read":["msg"]}]'`.
  Serve the course with a *threaded* server (plain `python3 -m http.server` can hang on stalled connections).
- `node` may not be on PATH in non-interactive shells: use `~/.nvm/versions/node/v20.9.0/bin/node`.
  The harness uses a 180 s Chrome launch timeout because the Mac is often heavily loaded (VMs, Gradle).
- Deep-RL labs (Phase 4+) need torch + gymnasium; verify in a scratch venv (e.g. `/tmp/rlvenv`).
- LLM labs (7–8) need transformers and are slow on CPU (Lab 7 ≈ 5–30 min, Lab 8 ≈ 15–50 min depending on load).
  Run them detached (`nohup … > /tmp/labN.txt &`) and poll. Lab 8 checks *sampled* (T=1) accuracy: GRPO on
  SmolLM2-135M mostly sharpens the distribution, so greedy accuracy barely moves (0.64 → 0.69 in one run).
- Phase 8–9 lessons are all ready; Lab 9 (`phase9_statistics.py`) is numpy-only.

## Clarity rules (from the learner's review, Oct 2026)
- Every symbol gets a plain-English reading at first use in each lesson (𝔼, ∝, ∇, ‖, ≻, ~, argmax, C(n,k), ★, hats).
- Order inside explanations: numbers/example → sentence → equation. Never an equation before its words.
- No method names before they're taught; tables of methods need "what it changes, in plain English" + "why".
- Re-anchor reused concepts with a 5–10 word reminder, not just "(see 7.4)".
- Every lesson keeps a `.box.analogy`.
- Core lessons start with a `.box.prereq` ("🧰 You'll need", 2–3 one-line reminders with lesson refs) right
  before the hook, and have a `.box.walk` ("✏️ Worked example") with small numbers computed by hand.
- Review loop that worked: parallel read-only subagents per 2 phases rating intuitiveness/ease/fun and quoting
  the worst passages with rewrites; plus a research subagent for missing topics (verify its sources yourself).
- Prose inside JS string literals (e.g. arrays of cards): don't use raw double quotes; use ‘ ’ instead.
- `.box.watch` ("📺 Watch/Read") boxes link only to videos/posts verified by web search; say what to look for
  and roughly how long. Verify YouTube IDs with `https://www.youtube.com/oembed?format=json&url=...`
  (returns real title + channel; 401 = embedding disabled but the video exists, 404 = bad ID). Phase 9.8 is the curated "Go deeper" resource page.
- Every lesson has a `.box.explore` ("📚 Explore more") before the quizzes: 2–4 verified links tagged
  Paper/Blog/Course/Book/Docs/Code with a one-line "why". Verify arXiv IDs in bulk via
  `http://export.arxiv.org/api/query?id_list=...`; check other links with curl (OpenAI and DOI sites return
  403 to scripts, so confirm those by web search).
- Animations ("🎬 Animation" playgrounds) are in-page DOM/canvas with Play/Pause/Step; prototype any claimed
  behavior in node first (e.g. PPO clip demo: 34% vs 2% collapse over 200 seeded runs; noisy-TV: TV bonus
  1–1.2, ε = 0.03). Prefer these over rendered video (rendering is CPU-heavy on this laptop).

## User preference: no heavy jobs on this laptop
Do NOT run models, training, or other CPU/GPU-heavy jobs (Labs 4–8, torch/transformers) on the user's Mac;
it slows their machine. Leave them as TODOs below (or for the user to run on Colab). Light checks (numpy
labs 0–3 and 9, the page harness) are fine but keep them short.

## TODO (resource-heavy, not run)
- Lab 8: confirmation run of `labs/solutions/phase8_grpo.py` after its final checks were changed to
  "sampled accuracy +20 points" and "sampled/greedy gap shrinks". The previous run (same seeds) printed
  BEFORE greedy 0.64 / sampled 0.28 and AFTER greedy 0.58 / sampled 0.59, so both checks should pass.
- Lab 8: smoke-test the exercise file `labs/phase8_grpo.py` (it loads the model, then should stop at TODO 1).
- Oct 2026 review edits, py_compile-only (run each once on Colab or a scratch venv):
  Lab 4 pass mark ≥ 80 + debug checklist; Lab 5 PPO diagnostics (approx KL, clip frac, entropy);
  Lab 6 torch imports moved to Part B (Part A verified torch-free) + instant SAC-loss checks;
  Lab 7 brute-force `seq_logprob` check + per-epoch chosen log-prob print; Lab 8 clip-frac/length prints.
  DQN is seed-sensitive: sweep configs over ≥3 seeds before setting defaults/thresholds.
- `labs/phase0_env_and_returns.py` holds the user's own in-progress answers (TODO 1–2). Before sharing,
  ship a clean exercise copy (blank the TODO bodies from the solution); keep the user's file.
- Uppercase `.label` text turns Greek letters into capitals; wrap them as `<span class="nt">γ</span>`.
- Labs: `cd labs && MPLBACKEND=Agg python3 solutions/<file>.py` (all ✅, no errors).
- Pages: `python3 -m http.server 8765` in repo root, load every page, click playground buttons,
  check the browser console for errors.

## Content rules
- Phases 7–8 (LLM RL): re-research the latest methods right before writing them; the field moves monthly.
- Verify fun facts/dates before including them. Prefer primary sources.

## Tooling gotcha (seen in this repo)
After a file is changed by a shell command (sed/python), later in-editor edits to that same file
were silently lost: the tool reported success but the disk was unchanged. After editing, confirm on
disk (`grep -c "new text" file`), and prefer one method per file within a session.

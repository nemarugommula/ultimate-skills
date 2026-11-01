# AGENTS.md: Deep Learning course

Static, dependency-free HTML course (no build step; must work from `file://`). Sibling of `~/math-course`, `~/rl-course`,
`~/system-design-course`, `~/ai-engineer`, `~/machine-learning` and `~/autonomous-systems`, on the **same design system**.
Goal: a beginner gets a **deep, intuitive understanding of how neural networks work and why every technique exists**, across
training, vision, sequences, transformers, generative models, audio and multimodal, robustness and training at scale.
It is not a proof course: formulas link to `~/math-course` lessons instead of being re-derived.
The learner is a working software engineer and a deep-learning beginner. **No jargon before it's taught.**

## The two rules that make this course
1. **Never just say “we use X”.** Every activation, normalization, optimizer, architecture block and trick gets a
   `🤔 Why this?` box (below). The learner must be able to say what broke before X, why X fixes it, and what X costs.
2. **Curiosity, interactivity and fun.** The user's words: *“as interactive and in same format of our existing courses, so
   while reading we have that curiosity and learning motivation and fun.”* In practice:
   - Same look, structure and box anatomy as the siblings. Before writing, open `phase0/4-activations.html` (this course's
     reference lesson), `~/ai-engineer/phase3/3-chunking.html`, and one lesson each from `~/rl-course` and `~/math-course`.
   - **Open with a puzzle**, surprising result or “what would you do?” question that the lesson then answers.
   - **Predict, then reveal**: before a playground shows the answer, ask the learner to guess (a `.predict` line:
     “🔮 Predict: …”) and say what they should see after.
   - **A playground or small challenge near every big idea**, not just two at the end. Let the learner break things and fix
     them. Add **score targets / mini-games** (“get the net to 95% on the spiral”, “find the learning rate that kills it”) and
     live numbers that update as they drag.
   - **Payoff early**: who uses this and what it powers, in the first screen. Light, friendly, conversational voice; short
     sections; small wins; fun facts. Never a wall of text (no paragraph over ~5 lines).

## The field moves monthly
Today is late 2026. Before writing about models, libraries, APIs or benchmarks, **search the web for the current state**
(official docs, release notes, papers) and use current names and APIs (PyTorch 2.x: `torch.compile`, `torch.amp`,
`torchvision.transforms.v2`, `nn.RMSNorm`, `F.scaled_dot_product_attention`…). Keep volatile details (model names, versions,
scores) few, dated (“as of 2026”) and sourced. **Never invent a model name, paper, number, score or quote.**

## Files
- `assets/course.js`: `COURSE` manifest (phases `d0`–`d10`, lesson `id` `d<P>-<k>`, recap `d<P>-r`, `title`, `href`, `mins`),
  sidebar, prev/next, progress, quizzes, theme, `window.MX` helpers (randn, seeded, linspace, sum, mean, std, dot, norm,
  cosine, matvec, matmul, transpose, sigmoid, softmax(z, T), entropy, sample, argmax, clamp, fmt, css). Storage prefix
  `dlcourse.`. The manifest is the source of truth: don't rename lessons or files without updating it.
- `assets/nn.js`: **the deep-learning toy kit for playgrounds**. Read its header comment. Seeded RNG, every activation with
  its slope, 2D datasets (xor, circle, spiral, moons, gauss, linear), a real trainable `NN.MLP` (any depth, activation,
  init, optimizer, loss, clipping, per-layer gradient norms, dead/saturated unit stats), optimizers, loss surfaces,
  `conv2d`/`pool`, tiny 8×8 pictures and digits, `iou`/`nms`, `attention`, `fft`/`stft`, and drawing helpers
  (`drawBoundary`, `heat` for any matrix (attention maps, kernels, images), `onCells` for painting, `histogram`, `loop`).
  Its gradients are checked against finite differences. Extend it if you need to (keep the header up to date, add a short
  comment, don't change existing behaviour); put one-off code in the lesson's IIFE.
- `assets/plot.js` (`Plot.line`, `Plot.bars`, `Plot.smooth`; `<canvas class="chart">`), `assets/plane.js`
  (`new Plane(canvas, {range, xRange, yRange, onDrag})` + `.draw(items)` with `vec pt seg fn path text heat`;
  `<canvas class="plane">`), `assets/flow.js` (animated diagrams: `new Flow(svg,{width,height})`, `.node(id,{x,y,w,h,label,sub,kind})`,
  `.edge(a,b)`, `.render()`, `.set`, `.edgeStyle`, `await .packet([...])`; kinds `x w ac gr ls at y` + `er`).
- `phase0/4-activations.html` is the **reference lesson**. Copy its structure, tone and level of interactivity.
- `tools/test-pages.mjs` (headless harness), `tools/build-review-bank.py` (quiz bank for `review.html`),
  `tools/build-glossary.py` (merges `tools/glossary/phase*.json` into `glossary.html`).

## Lesson anatomy (keep this order)
`<body data-lesson="d<P>-<k>" data-root="..">`, title `P.k Title · Deep Learning`, same head tags as the reference.
1. `.kicker` + `h1` + `.subtitle`. 2. `.box.prereq` “🧰 You'll need” (1–3 one-line reminders with lesson numbers; skip in 0.1).
3. `.box.hook` “🎯 Hook”: a real, verified story/result, or a puzzle, plus who uses this. End with a question the lesson answers.
4. Content: **numbers first → plain sentence → picture/playground → formula** (a formula always has its words first and a
   link to the math-course lesson that derives it). Include `.box.bigidea`, `.box.analogy`, `.box.walk` “✏️ Worked example”
   (small numbers by hand: parameters, gradients, shapes, FLOPs, memory), comparison tables, **one `🤔 Why this?` box per
   technique**, `.predict` lines before reveals, and **≥ 2 `.playground`s** (aim for 3) that are genuinely interactive
   (sliders, drag, paint, train live, step) with live readouts, where the learner can **cause the failure and see the fix**.
5. `.box.tools` “🧰 Tools you'll actually use”: current PyTorch (and torchvision / Hugging Face / timm / ultralytics…
   where natural) code, short and correct. **Run every snippet you can on CPU in seconds** (`python3`, torch 2.8 is
   installed; no downloads of weights or datasets) and only quote outputs you saw.
6. `.box.tradeoff` “⚖️ Trade-offs”. 7. `.box.interview` “🎤 Interview angle”.
8. `.box.used` “🏭 In the real world”: 3–5 `.use` rows with **How:** / **Why this method:** and a linked primary source
   (paper, official docs, company engineering blog). Tags: `tag co` 🏢 Company · year, `tag res` 🧪 Research · year,
   `tag oss` 🧰 Open source, `tag hist` 📜 History.
9. `.box.fact` “🤯 Fun fact” (verified). 10. `.box.bridge` “🌉 How it connects” (lessons by number; sibling links where they
   truly help). 11. `.box.explore` “📚 Explore more” (2–3 verified links). 12. `h2` Check yourself + **exactly 3 `.quiz`**
   (one `button.opt[data-ok]`, 3 options, teaching `.explain`; one calculation, one scenario/diagnosis). 13. `.box.tldr` “📝 TL;DR”.
Scripts at the end: `../assets/course.js`, then `nn.js` / `plot.js` / `plane.js` / `flow.js` as needed, then one IIFE.
Lessons are typically 30–50 KB. No external scripts, fonts or images (inline SVG/canvas only).

### 🤔 Why this? (required for every technique)
```html
<div class="box why">
  <span class="label">🤔 Why this? <span class="what">ReLU</span></span>
  <ol>
    <li><b>The problem before</b> … (with a number)</li>
    <li><b>What people tried</b> …</li>
    <li><b>Why this works</b> intuition + a tiny numeric example …</li>
    <li><b>What it costs, when it fails</b> …</li>
    <li><b>What refined it</b> … (lesson links)</li>
  </ol>
</div>
```
Example (ReLU): sigmoid's slope is at most 0.25, so 10 layers shrink the gradient by up to 0.25¹⁰ ≈ 1e-6 → people tried tanh
and careful init → ReLU has slope exactly 1 for z > 0, is one comparison, and gives sparse activations → units stuck below 0
get zero gradient forever (“dying ReLU”) → Leaky ReLU, then GELU/SiLU, then SwiGLU.

### Predict-then-reveal
`<p class="predict">🔮 <b>Predict:</b> with 6 sigmoid layers, will the first layer learn faster or slower than the last?</p>`
then the playground, then a short “What you should see:” paragraph. Prototype the playground in node first
(`assets/nn.js` loads in node with tiny stubs: see `tools/nnload.js`) so the promise is true for the seeded defaults.

## Recap page (`phaseP/recap.html`, id `dP-r`)
Kicker “Phase P · Recap”, h1, subtitle. Then: 🧾 cheat sheet (`.grid` of `.card`s, one per lesson: the rule/number in color +
one line + tools), 🗣️ teach-back prompt (`.box.analogy`, phrased as an interviewer's question, with what a strong answer
covers), 🎮 an interactive game (e.g. diagnose a training run from its loss curve, match architecture ↔ task, symptom ↔ fix;
score + instant feedback + lesson to revisit), 🏗️ “Build it yourself” problem (`.box.walk`) with a
`<details class="mathbox">` reveal (architecture, shapes, parameter count, loss, training recipe, pitfalls), 🏆 boss quiz of
**5 `.quiz`**, a 💻 **Lab P** section (what you'll build, the file, how to run, one line per TODO), and a 🌉 “What's next” box.
Phase 10's recap is the graduation page: an interactive skills map, a portfolio plan, where to go next (verified resources).

## Labs (`labs/phaseN_<topic>.py`, solutions in `labs/solutions/` with the same filename)
`# %%` cells (VS Code “Run Cell”), a markdown header cell (what you'll build, setup, lessons to reread), small TODOs that
`raise NotImplementedError("TODO k: …")`, and `check(cond, msg)` printing ✅ right after each. PyTorch on **CPU only**, tiny
synthetic data (or sklearn-free generators you write), `torch.manual_seed`, **whole file runs in under a minute** (aim < 20 s).
No downloads (no pretrained weights, no datasets). Requirements: `labs/requirements.txt` (torch, numpy, matplotlib).
Plots must not block: use the `show()` helper (closes the figure under `MPLBACKEND=Agg`).
Verify: `cd ~/deep-learning/labs && MPLBACKEND=Agg python3 solutions/<file>.py` → all ✅, exit 0. The TODO file must fail at
TODO 1 with NotImplementedError (not a syntax error).

## Color language (CSS classes; same meaning everywhere)
`.x` input / data (blue) · `.w` weights / parameters (orange) · `.ac` activations / hidden features (purple) ·
`.gr` gradients (pink) · `.ls` loss (teal) · `.at` attention (gold) · `.y` output / prediction / target (green) ·
`.dead` dead / failure (red). `.lr` learning rate (gold, from the math course). Canvas vars: `--c-x --c-w --c-ac --c-gr
--c-ls --c-at --c-y --c-dead` (+ `--text --muted --line --accent --good --bad --bg2 --panel`). Two-class toy data uses
coral (class 0) and blue (class 1) via `NN.drawBoundary`. Uppercase `.label` text capitalizes Greek: wrap as `<span class="nt">σ</span>`.

## Clarity rules
- Explain every acronym and term at first use in each lesson (MLP, SGD, BN, CNN, IoU, NMS, mAP, ViT, RNN, LSTM, KV cache, MoE…).
- No term before it's taught; mention later ideas in plain words + “(Lesson X.Y)”. Short sentences, concrete numbers.
- Prose inside JS string literals: no raw double quotes; use ‘ ’ or “ ”.
- Verify every number you compute with node or python3.

## Cross-links (list files first; exact names)
- Math: `../../math-course/phase3/2-chain-rule.html`, `3-gradients.html`, `4-gradient-descent.html`, `5-backprop.html`,
  `phase6/3-cross-entropy.html`, `phase7/2-sgd.html`, `3-adam.html`, `4-regularization.html`, `phase8/2-softmax.html`,
  `3-neural-net.html`, `3b-convolution.html`, `4-attention.html`, `4b-diffusion.html`, `5-numerics.html` (and the rest of `ls ~/math-course/phase*/`).
- RL course for RLHF/DPO/RL fine-tuning (`../../rl-course/phase7/…`, `phase8/…`); AI Engineer for applied LLMs
  (fine-tuning/LoRA `../../ai-engineer/phase7/1-fine-tuning.html`, serving `phase7/4-serving.html`, RAG, evals, multimodal
  `phase8/4-multimodal.html`); Machine Learning course (`../../machine-learning/…`, classical ML) and Autonomous Systems
  (`../../autonomous-systems/…`, sensors/SLAM/BEV perception/planning) only to files that exist when you write.
  Don't teach what those courses own; link to them.

## Facts and links
- Check every story, number, quote and API detail against its source (curl it; confirm the claim is in the text; `pdftotext`
  is installed for PDFs). If you can't, drop it.
- Links: `curl -s -o /dev/null -w '%{http_code}' -L -A 'Mozilla/5.0' URL` → 200. arXiv: `https://export.arxiv.org/api/query?id_list=…`
  (check the title matches).

## Verify
- Server: `http://127.0.0.1:8772/deep-learning/` (a scratch `www` dir with symlinks to every sibling course). If it is down:
  `cd <dir with symlinks> && python3 -m http.server 8772 --bind 127.0.0.1`.
- Harness (puppeteer in `/tmp/rltest`): `cp ~/deep-learning/tools/test-pages.mjs /tmp/rltest/dl-test.mjs && cd /tmp/rltest &&
  ~/.nvm/versions/node/v20.9.0/bin/node dl-test.mjs '[{"page":"phase0/4-activations.html","clicks":[{"id":"train"}],"read":["loss"]}]'`.
  Clicks take `{"id"}` or `{"sel"}` (CSS selector) plus `"all":true` (click every match), `"value":v` (set a slider/input and
  fire input+change) or `"select":"v"`. Add `"shots":[{"sel":".playground","name":"x"}]` for `/tmp/rltest/x.png` (look at it). Every page: **0 errors**, 3 quizzes
  (5 on recaps), every button clicked once in the harness.
- No heavy jobs on this laptop: no GPU, no model or dataset downloads, no training beyond seconds.

## Review bank and glossary
After adding/editing quizzes: `python3 tools/build-review-bank.py`. Quiz answers are saved by position; don't reorder quizzes
in published lessons. Insert lessons without renumbering with a letter suffix (`phase4/2b-…`, id `d4-2b`).
Glossary: each phase writes `tools/glossary/phaseN.json` = `[{"term", "kind" (x w ac gr ls at y or ""), "meaning",
"tools", "lessons": "3.2 3.4"}]` (plain-English meaning, 1–3 sentences); `python3 tools/build-glossary.py` rebuilds the page.

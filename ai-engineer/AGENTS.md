# AGENTS.md: AI Engineer course

Static, dependency-free HTML course (no build step; must work from `file://`). Sibling of `~/rl-course`, `~/math-course` and
`~/system-design-course`, built on the same design system. Goal: take a software engineer to **senior AI engineer** level:
building LLM products, RAG (every major strategy), vector search, evals, agents/MCP, fine-tuning and serving, LLMOps, safety and
AI system design. Based on the roadmap.sh AI Engineer roadmap, expanded to senior depth. **Teach from fundamentals, intuitively**:
numbers first, playgrounds the learner can break, the real tools and code, trade-offs, startup vs big tech, the interview angle.
The learner is a working developer, new to AI engineering: no jargon before it's taught.

## The field moves monthly
Today is late 2026. Before writing about models, APIs, SDKs, frameworks, prices or benchmarks, **search the web for the current
state** (official docs, provider changelogs, release notes) and use current names and APIs. Prefer stable concepts in prose and keep
volatile details (model names, prices, context sizes) few, dated (“as of 2026”) and sourced. Never invent a model name, API
parameter, benchmark score or price.

## Files
- `assets/course.js`: `COURSE` manifest (phases `a0`–`a9`, lesson `id`, `title`, `href`, `mins`), sidebar, prev/next, progress,
  quizzes, theme, and `window.MX` helpers (randn, seeded(seed)→rng, linspace, sum, mean, std, dot, norm, cosine, softmax(z, T),
  sample(p), entropy, clamp, fmt(v,d), css(var)…). Storage keys start with `aicourse.`. The manifest is the source of truth.
- `assets/ai.js`: **offline NLP toy kit** for playgrounds: `AI.countTokens`, `AI.toyTokens`, `AI.words`, `AI.sentences`,
  `AI.embed(text)` (toy semantic embedding with synonym groups; extend with `AI.addConcept([...])`), `AI.cosine`, `AI.bm25`,
  `AI.search(docs, q, {mode: "dense"|"bm25"|"hybrid", k})`, `AI.rrf`, `AI.mmr`, `AI.chunk(text, {method, size, overlap})`.
  Read its header. Playgrounds must not call real model APIs; simulate LLM behaviour with clearly-labelled, scripted or rule-based
  stand-ins, and say in the text that a real model is used in practice.
- `assets/flow.js`: animated pipeline/architecture diagrams: `new Flow(svg, {width, height})` → `.node(id, {x, y, w, h, label, sub,
  kind})` (x, y = center; kind letters `cl sv db nw ca er lt`: use cl ≈ input, nw ≈ model, db ≈ context/data, sv ≈ tool, ca ≈ eval,
  er ≈ risk, lt ≈ cost) → `.edge(a, b, {dashed, label})` → `.render()`; `.set(id, {sub, label, down, hot, load})`,
  `.edgeStyle(a, b, {color, width, dashed, label})`, `await .packet([ids…], {color, dur, back})`.
- `assets/plot.js`: `Plot.line(canvas, {series:[{label,color,data:[y]|[[x,y]],dash,dots,fill,width}], xMin,xMax,yMin,yMax,xLabel,yLabel,hlines})`,
  `Plot.bars(canvas, {bars:[{label,value,color}], yMin,yMax})`. Use `<canvas class="chart">`.
- Extra CSS: `.doc` (document view), `.chunk` + `.c0–.c3`, `.prompt` (monospace prompt block), `.hit` / `.hit.miss` (retrieved result rows).
- `phase3/3-chunking.html` is the **reference lesson**. Copy its structure, tone and level of interactivity.
- `tools/test-pages.mjs` (headless harness), `tools/build-review-bank.py` (quiz bank for `review.html`).

## Lesson anatomy (keep this order)
`<body data-lesson="a<P>-<k>" data-root="..">`, title `P.k Title · AI Engineer`, same head tags as the reference.
1. `.kicker` + `h1` + `.subtitle`. 2. `.box.prereq` “🧰 You'll need” (skip only in 0.1). 3. `.box.hook` “🎯 Hook”: a real, verified story/result.
4. Content: **numbers first → plain sentence → diagram/formula/prompt.** `.box.bigidea`, `.box.analogy`, `.box.walk` “✏️ Worked example”
   (tokens, dollars, latency, recall… by hand), comparison tables, and **≥ 2 `.playground`s** that are genuinely interactive and let the
   learner cause the failure and see the fix (e.g. retrieval misses, judge bias, injection succeeding, agent looping).
5. `.box.tools` “🧰 Tools you'll actually use”: real libraries/services with a short, correct, current code snippet (Python preferred;
   OpenAI/Anthropic/Gemini SDKs, LangChain, LlamaIndex, LangGraph, Hugging Face, vLLM, Ollama, pgvector, Qdrant, RAGAS…).
6. `.box.tradeoff` “⚖️ Trade-offs”. 7. `.box.startup` “🚀 Startup vs big tech” (`.cols`). 8. `.box.interview` “🎤 Interview angle”.
9. `.box.used` “🏭 In the real world”: 3–5 `.use` rows with **How:** / **Why this method:** and a linked primary source (papers, official
   docs, company engineering blogs). Tags: `tag co` 🏢 Company · year, `tag res` 🧪 Research · year, `tag oss` 🧰 Open source, `tag hist` 📜 History.
10. `.box.fact` “🤯 Fun fact” (verified). 11. `.box.bridge` “🌉 How it connects” (lessons by number; link sibling courses where it truly
    helps: `../../math-course/…`, `../../rl-course/…`, `../../system-design-course/…`; list files to get exact names).
12. `.box.explore` “📚 Explore more” (2–3 verified links). 13. **Exactly 3 `.quiz`** (one `data-ok`, 3 options, teaching `.explain`;
    one estimation/calculation, one scenario). 14. `.box.tldr` “📝 TL;DR”.
Scripts at the end: `../assets/course.js`, then `plot.js` / `flow.js` / `ai.js` as needed, then one IIFE. Lessons are typically 25–45 KB.

## Recap page (`phaseP/recap.html`, id `aP-r`)
Cheat-sheet cards (one per lesson: key rule/number + one line + tools), a teach-back prompt phrased as an interviewer's question, an
interactive game (e.g. diagnose the failing RAG pipeline, match symptom ↔ fix), a 🏗️ “Build it yourself” problem with a
`<details class="mathbox">` reveal (requirements, pipeline, eval plan, costs, pitfalls), a boss quiz of **5 `.quiz`**, and a 🌉
“What's next” box. Phase 9's recap is the graduation page: an interactive skills map by goal, a portfolio/project plan, and where to
go next (verified resources).

## Color language (CSS classes)
`.in` user input/query/prompt (blue) · `.md` model/LLM/generation (purple) · `.cx` context/retrieved docs/data (green) ·
`.tl` tool/agent action/code (orange) · `.ev` eval/score/metric (teal) · `.rk` risk/hallucination/attack (red) ·
`.tk` tokens/cost/latency (gold). CSS vars `--c-in --c-md --c-cx --c-tl --c-ev --c-rk --c-tk` (+ `--text --muted --line --accent --good --bad`).

## Clarity rules
- Every acronym and term is explained at first use in each lesson (RAG, ANN, HNSW, BM25, RRF, MMR, LoRA, SFT, DPO, KV cache, MCP, TTFT…).
- No term before it's taught; mention later ideas in plain words + “(Lesson X.Y)”. Short sentences, concrete numbers.
- Label typical numbers as typical; prices as illustrative unless sourced (with date).
- Prose inside JS string literals: no raw double quotes; use ‘ ’ or “ ”.
- Verify every number you compute with node or python3. Prototype playground behaviour in node before claiming it in prose.

## Facts and links
- Check every story, number, quote and API detail against its source (fetch with curl; confirm the claim is there). If you can't, drop it.
- Links: `curl -s -o /dev/null -w '%{http_code}' -L -A 'Mozilla/5.0' URL` → 200. arXiv: `https://export.arxiv.org/api/query?id_list=…`.
  YouTube: `https://www.youtube.com/oembed?format=json&url=…`.

## Verify
- Server: `http://127.0.0.1:8766/` serves `ai-engineer/` (check `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/ai-engineer/index.html`).
- Harness (puppeteer in `/tmp/rltest`): `cp ~/ai-engineer/tools/test-pages.mjs /tmp/rltest/ai-test.mjs && cd /tmp/rltest &&
  ~/.nvm/versions/node/v20.9.0/bin/node ai-test.mjs '[{"page":"phase3/3-chunking.html","clicks":[{"id":"sweep"}],"read":["score"]}]'`.
  Add `"shot":"name","scroll":".playground"` for `/tmp/rltest/name.png`. Every page: **0 errors**, right quiz count, every button working.
- No heavy jobs on this laptop (no model downloads, training or GPU work).

## Review bank
After adding/editing/moving quizzes, rerun `python3 tools/build-review-bank.py`. Quiz answers are saved by position; don't reorder
quizzes in published lessons. Insert lessons without renumbering with a letter suffix (`phase4/2b-…`, id `a4-2b`).

/* ============ AI Engineer — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index) get theme + quizzes only.
 * Also exposes window.MX: small math/random helpers used by playgrounds,
 * and window.AICourse: progress export/import + small helpers.
 * Storage keys start with "aicourse." so progress never mixes with the sibling courses.
 */

const COURSE = [
  {
    id: "a0", num: 0, emoji: "🌱", title: "Foundations: what an AI engineer does", status: "ready",
    why: "The role, how LLMs work (just enough), tokens and cost math, the model landscape, and calling models through APIs.",
    lessons: [
      { id: "a0-1", title: "What an AI engineer does (and how it differs from ML engineering)", href: "phase0/1-ai-engineer.html", mins: 15 },
      { id: "a0-2", title: "How LLMs work: tokens, next-token prediction, transformers", href: "phase0/2-how-llms-work.html", mins: 20 },
      { id: "a0-3", title: "Tokens, context windows, cost & latency math", href: "phase0/3-tokens-cost.html", mins: 20 },
      { id: "a0-4", title: "The model landscape: closed vs open, choosing a model", href: "phase0/4-model-landscape.html", mins: 20 },
      { id: "a0-5", title: "Calling models: chat APIs, sampling, streaming, SDKs", href: "phase0/5-llm-apis.html", mins: 20 },
      { id: "a0-6", title: "When not to use an LLM: rules, classic ML & small models", href: "phase0/6-when-not-llm.html", mins: 15 },
      { id: "a0-r", title: "Recap: Foundations", href: "phase0/recap.html", mins: 20 },
    ],
  },
  {
    id: "ap", num: "P", emoji: "🐍", title: "Python for AI engineers", status: "ready",
    why: "A crash course and refresher on the Python AI engineers use every day: Pythonic idioms and typing, generators, async/await, concurrency and the GIL, memory and garbage collection, production services, and NumPy/pandas.",
    lessons: [
      { id: "ap-1", title: "Pythonic Python: data model, typing, dataclasses, decorators", href: "phasep/1-pythonic.html", mins: 25 },
      { id: "ap-2", title: "Iterators & generators: lazy pipelines and streaming", href: "phasep/2-generators.html", mins: 20 },
      { id: "ap-3", title: "async/await: the event loop, gather, semaphores & async streams", href: "phasep/3-async.html", mins: 25 },
      { id: "ap-4", title: "Concurrency & parallelism: the GIL, threads, processes", href: "phasep/4-concurrency.html", mins: 25 },
      { id: "ap-5", title: "Memory & garbage collection: references, cycles, leaks", href: "phasep/5-memory-gc.html", mins: 20 },
      { id: "ap-6", title: "Production Python for AI: FastAPI, Pydantic, streaming, uv, pytest", href: "phasep/6-production-python.html", mins: 25 },
      { id: "ap-7", title: "NumPy & pandas for AI engineers", href: "phasep/7-numpy-pandas.html", mins: 20 },
      { id: "ap-r", title: "Recap: Python for AI engineers", href: "phasep/recap.html", mins: 20 },
    ],
  },
  {
    id: "a1", num: 1, emoji: "✍️", title: "Prompt & context engineering", status: "ready",
    why: "Write prompts that work, make models reason, get reliable structured output, and engineer exactly what goes into the context window.",
    lessons: [
      { id: "a1-1", title: "Prompting fundamentals: instructions, roles, few-shot", href: "phase1/1-prompting.html", mins: 20 },
      { id: "a1-2", title: "Reasoning: chain-of-thought, self-consistency & reasoning models", href: "phase1/2-reasoning.html", mins: 20 },
      { id: "a1-3", title: "Structured output & function calling", href: "phase1/3-structured-output.html", mins: 20 },
      { id: "a1-4", title: "Context engineering: long context, ordering, caching, memory", href: "phase1/4-context-engineering.html", mins: 20 },
      { id: "a1-5", title: "Prompts as code: templates, versioning & experiments", href: "phase1/5-prompt-ops.html", mins: 15 },
      { id: "a1-r", title: "Recap: Prompt & context engineering", href: "phase1/recap.html", mins: 20 },
    ],
  },
  {
    id: "a2", num: 2, emoji: "🧭", title: "Embeddings & vector search", status: "ready",
    why: "Meaning as vectors: embedding models, similarity, approximate nearest-neighbour indexes, vector databases, hybrid search and other embedding uses.",
    lessons: [
      { id: "a2-1", title: "Embeddings: turning meaning into vectors", href: "phase2/1-embeddings.html", mins: 20 },
      { id: "a2-2", title: "Similarity search & ANN indexes: HNSW, IVF, PQ", href: "phase2/2-ann.html", mins: 25 },
      { id: "a2-3", title: "Vector databases: pgvector, Pinecone, Qdrant, Weaviate & friends", href: "phase2/3-vector-databases.html", mins: 20 },
      { id: "a2-4", title: "Hybrid search: BM25 + dense, fusion & sparse vectors", href: "phase2/4-hybrid-search.html", mins: 20 },
      { id: "a2-5", title: "Beyond search: classification, clustering, dedup, recommendations", href: "phase2/5-embedding-uses.html", mins: 15 },
      { id: "a2-r", title: "Recap: Embeddings & vector search", href: "phase2/recap.html", mins: 20 },
    ],
  },
  {
    id: "a3", num: 3, emoji: "📚", title: "RAG fundamentals", status: "ready",
    why: "Retrieval-augmented generation from first principles: why it exists, ingesting documents, chunking, retrieval and reranking, grounded generation, and a full pipeline.",
    lessons: [
      { id: "a3-1", title: "Why RAG: hallucination, cut-offs, private data (vs fine-tuning)", href: "phase3/1-why-rag.html", mins: 15 },
      { id: "a3-2", title: "Ingestion: parsing PDFs, HTML & tables; cleaning & metadata", href: "phase3/2-ingestion.html", mins: 20 },
      { id: "a3-3", title: "Chunking strategies: the decision that makes or breaks RAG", href: "phase3/3-chunking.html", mins: 25 },
      { id: "a3-4", title: "Retrieval & reranking: top-k, cross-encoders, MMR, filters", href: "phase3/4-retrieval-reranking.html", mins: 20 },
      { id: "a3-5", title: "Grounded generation: citations, abstaining & context packing", href: "phase3/5-grounded-generation.html", mins: 20 },
      { id: "a3-6", title: "Build a RAG pipeline end to end", href: "phase3/6-naive-rag.html", mins: 25 },
      { id: "a3-r", title: "Recap: RAG fundamentals", href: "phase3/recap.html", mins: 20 },
    ],
  },
  {
    id: "a4", num: 4, emoji: "🧪", title: "Advanced RAG strategies", status: "ready",
    why: "Every major RAG pattern: query transformations, advanced indexing, GraphRAG, agentic/self-correcting RAG, multimodal and structured-data RAG, and RAG in production.",
    lessons: [
      { id: "a4-1", title: "Query transformation: rewriting, multi-query, HyDE, decomposition", href: "phase4/1-query-transformation.html", mins: 20 },
      { id: "a4-2", title: "Advanced indexing: parent-child, hierarchical, multi-vector, ColBERT", href: "phase4/2-advanced-indexing.html", mins: 25 },
      { id: "a4-3", title: "GraphRAG & knowledge graphs", href: "phase4/3-graphrag.html", mins: 25 },
      { id: "a4-4", title: "Agentic, self-correcting & adaptive RAG (Self-RAG, CRAG, routing)", href: "phase4/4-agentic-rag.html", mins: 25 },
      { id: "a4-5", title: "Multimodal RAG: images, tables & visual documents", href: "phase4/5-multimodal-rag.html", mins: 20 },
      { id: "a4-6", title: "Structured data: text-to-SQL & tabular RAG", href: "phase4/6-structured-rag.html", mins: 20 },
      { id: "a4-7", title: "RAG in production: freshness, permissions, caching & cost", href: "phase4/7-rag-production.html", mins: 20 },
      { id: "a4-8", title: "RAG pipelines at scale: ingestion orchestration, batch embedding & backfills", href: "phase4/8-rag-pipelines.html", mins: 20 },
      { id: "a4-r", title: "Recap: Advanced RAG strategies", href: "phase4/recap.html", mins: 20 },
    ],
  },
  {
    id: "a5", num: 5, emoji: "📏", title: "Evaluation", status: "ready",
    why: "Replace vibes with measurement: golden datasets, metrics, LLM-as-judge, RAG and agent evals, and the tooling to run evals in CI and production.",
    lessons: [
      { id: "a5-1", title: "Why evals: eval-driven development & golden datasets", href: "phase5/1-why-evals.html", mins: 20 },
      { id: "a5-2", title: "Metrics: from exact match to semantic similarity", href: "phase5/2-metrics.html", mins: 15 },
      { id: "a5-3", title: "LLM-as-judge: rubrics, pairwise & judge bias", href: "phase5/3-llm-judge.html", mins: 20 },
      { id: "a5-4", title: "Evaluating RAG: retrieval & faithfulness metrics", href: "phase5/4-rag-evals.html", mins: 25 },
      { id: "a5-5", title: "Evaluating agents: trajectories, tool calls & benchmarks", href: "phase5/5-agent-evals.html", mins: 20 },
      { id: "a5-6", title: "Eval tooling, CI regression tests & online evals", href: "phase5/6-eval-ops.html", mins: 20 },
      { id: "a5-7", title: "Building eval datasets: real logs, synthetic & adversarial cases, coverage", href: "phase5/7-eval-datasets.html", mins: 20 },
      { id: "a5-8", title: "Human evaluation & annotation: guidelines, agreement, experts", href: "phase5/8-human-eval.html", mins: 20 },
      { id: "a5-9", title: "Evals by task: classification, extraction, summarization, code, chat", href: "phase5/9-task-evals.html", mins: 25 },
      { id: "a5-10", title: "Safety, red-team & robustness evals", href: "phase5/10-safety-evals.html", mins: 20 },
      { id: "a5-11", title: "Comparing models & prompts rigorously: benchmarks, arenas, statistics", href: "phase5/11-model-comparison.html", mins: 20 },
      { id: "a5-r", title: "Recap: Evaluation", href: "phase5/recap.html", mins: 20 },
    ],
  },
  {
    id: "a6", num: 6, emoji: "🤖", title: "Agents & tools", status: "ready",
    why: "From one LLM call to agents: the agent loop, tool design and MCP, workflow patterns, memory, multi-agent frameworks, and making agents reliable.",
    lessons: [
      { id: "a6-1", title: "What an agent is: the loop, ReAct & when not to use one", href: "phase6/1-agent-loop.html", mins: 20 },
      { id: "a6-2", title: "Tool design & the Model Context Protocol (MCP)", href: "phase6/2-tools-mcp.html", mins: 20 },
      { id: "a6-3", title: "Workflow patterns: chaining, routing, parallel, orchestrator-workers", href: "phase6/3-workflows.html", mins: 20 },
      { id: "a6-4", title: "Agent memory: short-term, long-term & summarization", href: "phase6/4-memory.html", mins: 20 },
      { id: "a6-5", title: "Multi-agent systems & frameworks (LangGraph, Agents SDKs, CrewAI)", href: "phase6/5-frameworks.html", mins: 20 },
      { id: "a6-6", title: "Reliable agents: guardrails, human-in-the-loop, sandboxes, computer use", href: "phase6/6-agent-reliability.html", mins: 20 },
      { id: "a6-7", title: "Agent architectures: planner–executor, DAGs, tree-of-thought, reflection", href: "phase6/7-architectures.html", mins: 20 },
      { id: "a6-8", title: "Build an agent from scratch: the raw loop, parsing, errors & testing", href: "phase6/8-from-scratch.html", mins: 25 },
      { id: "a6-9", title: "Building & deploying MCP servers: local, remote, auth & security", href: "phase6/9-mcp-servers.html", mins: 20 },
      { id: "a6-10", title: "Agents in practice: coding, data analysis, browsing & assistants", href: "phase6/10-agent-use-cases.html", mins: 20 },
      { id: "a6-r", title: "Recap: Agents & tools", href: "phase6/recap.html", mins: 20 },
    ],
  },
  {
    id: "a7", num: 7, emoji: "🛠️", title: "Customizing & serving models", status: "ready",
    why: "When and how to fine-tune (LoRA/QLoRA), preference tuning and distillation, open models and local inference, and serving LLMs at scale.",
    lessons: [
      { id: "a7-1", title: "Fine-tuning: when it beats prompting, SFT data & LoRA", href: "phase7/1-fine-tuning.html", mins: 25 },
      { id: "a7-2", title: "Preference tuning, distillation & synthetic data", href: "phase7/2-preference-distillation.html", mins: 20 },
      { id: "a7-3", title: "Open models & local inference: Hugging Face, Ollama, llama.cpp", href: "phase7/3-open-models.html", mins: 20 },
      { id: "a7-4", title: "Serving LLMs at scale: batching, KV cache, vLLM, quantization", href: "phase7/4-serving.html", mins: 25 },
      { id: "a7-5", title: "Domain adaptation: making an LLM an expert in your domain", href: "phase7/5-domain-adaptation.html", mins: 25 },
      { id: "a7-r", title: "Recap: Customizing & serving models", href: "phase7/recap.html", mins: 20 },
    ],
  },
  {
    id: "a8", num: 8, emoji: "🛡️", title: "Production, safety & multimodal", status: "ready",
    why: "LLMOps and observability, latency and cost optimization, guardrails and prompt injection, multimodal AI, and responsible AI.",
    lessons: [
      { id: "a8-1", title: "LLMOps: tracing, gateways, fallbacks & cost tracking", href: "phase8/1-llmops.html", mins: 20 },
      { id: "a8-2", title: "Cutting cost & latency: caching, routing, batching", href: "phase8/2-cost-latency.html", mins: 20 },
      { id: "a8-3", title: "Guardrails & security: prompt injection, jailbreaks, PII", href: "phase8/3-guardrails.html", mins: 25 },
      { id: "a8-4", title: "Multimodal AI: vision, speech, image generation & realtime voice", href: "phase8/4-multimodal.html", mins: 20 },
      { id: "a8-5", title: "Responsible AI: bias, privacy, governance & regulation", href: "phase8/5-responsible-ai.html", mins: 15 },
      { id: "a8-r", title: "Recap: Production, safety & multimodal", href: "phase8/recap.html", mins: 20 },
    ],
  },
  {
    id: "ar", num: "R", emoji: "🗡️", title: "AI red teaming & security", status: "ready",
    why: "Think like an attacker to defend AI systems: threat modelling, prompt injection and jailbreak techniques, attacks on agents and tools, data and model attacks, and running a red-team engagement with automated tools.",
    lessons: [
      { id: "ar-1", title: "Threat modelling for LLM applications", href: "phaser/1-threat-modeling.html", mins: 20 },
      { id: "ar-2", title: "Prompt injection & jailbreaks: how attacks work", href: "phaser/2-injection-jailbreaks.html", mins: 25 },
      { id: "ar-3", title: "Attacking agents & tools: exfiltration, tool misuse, MCP risks", href: "phaser/3-agent-attacks.html", mins: 25 },
      { id: "ar-4", title: "Data & model attacks: poisoning, extraction, membership inference", href: "phaser/4-data-model-attacks.html", mins: 20 },
      { id: "ar-5", title: "Running a red-team engagement: tools, automation & reporting", href: "phaser/5-red-team-engagement.html", mins: 20 },
      { id: "ar-r", title: "Recap: AI red teaming & security", href: "phaser/recap.html", mins: 20 },
    ],
  },
  {
    id: "a9", num: 9, emoji: "🧭", title: "Becoming a senior AI engineer", status: "ready",
    why: "Design complete AI systems, build AI products people trust, and prepare for senior AI engineer interviews with a portfolio that proves it.",
    lessons: [
      { id: "a9-1", title: "AI system design: end-to-end case studies", href: "phase9/1-ai-system-design.html", mins: 25 },
      { id: "a9-2", title: "AI product & UX: trust, feedback loops & metrics", href: "phase9/2-ai-product.html", mins: 20 },
      { id: "a9-3", title: "The senior AI engineer interview & portfolio", href: "phase9/3-interview-portfolio.html", mins: 20 },
      { id: "a9-4", title: "Vertical AI: building for legal, health, finance & other domains", href: "phase9/4-vertical-ai.html", mins: 20 },
      { id: "a9-5", title: "Forward deployed engineering: shipping AI with customers", href: "phase9/5-forward-deployed.html", mins: 20 },
      { id: "a9-r", title: "Graduation: the senior AI engineer's map", href: "phase9/recap.html", mins: 20 },
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
  key: "aicourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "a3-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "aicourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "aicourse.theme";
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
    <a class="logo" href="${root}/index.html">AI <span>Engineer</span></a>
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
window.AICourse = {
  COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("aicourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "ai-engineer", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like an AI Engineer progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("aicourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
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

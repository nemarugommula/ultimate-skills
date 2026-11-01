/* ============ RL from Zero — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index, glossary) get theme + quizzes only.
 * Also exposes window.RL: small math/random helpers used by playgrounds,
 * and window.RLCourse: progress export/import + small helpers.
 */
const COURSE = [
  {
    id: "p0", num: 0, emoji: "🌱", title: "What is RL?", status: "ready",
    why: "Get the big picture: an agent tries things, gets scores, and gets better. Learn the 6 words used in every RL paper.",
    lessons: [
      { id: "p0-1", title: "Learning by trial & error", href: "phase0/1-trial-and-error.html", mins: 10 },
      { id: "p0-2", title: "The 6 words of RL", href: "phase0/2-vocabulary.html", mins: 15 },
      { id: "p0-3", title: "Rewards, returns & discounting", href: "phase0/3-returns.html", mins: 15 },
      { id: "p0-4", title: "RL's greatest hits & why LLMs need it", href: "phase0/4-why-llms.html", mins: 10 },
      { id: "p0-5", title: "One run lies: mean, spread & error bars", href: "phase0/5-noisy-results.html", mins: 15 },
      { id: "p0-6", title: "Is it an RL problem? When (not) to use RL", href: "phase0/6-is-it-rl.html", mins: 15 },
      { id: "p0-r", title: "Recap + Lab 0", href: "phase0/recap.html", mins: 30 },
    ],
  },
  {
    id: "p1", num: 1, emoji: "🎰", title: "Bandits: Explore vs Exploit", status: "ready",
    why: "The simplest RL problem. Learn to estimate values from noisy rewards and the hardest choice in RL: try something new, or stick with what works?",
    lessons: [
      { id: "p1-1", title: "The slot machine problem", href: "phase1/1-bandit-problem.html", mins: 15 },
      { id: "p1-2", title: "The one-line update rule", href: "phase1/2-estimating-values.html", mins: 15 },
      { id: "p1-3", title: "Explore vs exploit: ε-greedy", href: "phase1/3-explore-exploit.html", mins: 15 },
      { id: "p1-4", title: "Smarter exploration: UCB & Thompson", href: "phase1/4-ucb-thompson.html", mins: 20 },
      { id: "p1-r", title: "Recap + Lab 1", href: "phase1/recap.html", mins: 45 },
    ],
  },
  {
    id: "p2", num: 2, emoji: "🗺️", title: "MDPs & Bellman: planning with a map", status: "ready",
    why: "Add states and the future. Learn what “value” means and the Bellman equation, the most important equation in RL.",
    lessons: [
      { id: "p2-1", title: "The Markov property: the present is enough", href: "phase2/1-markov-property.html", mins: 15 },
      { id: "p2-2", title: "Markov Decision Processes (MDPs)", href: "phase2/2-mdps.html", mins: 15 },
      { id: "p2-3", title: "Value functions V(s) and Q(s,a)", href: "phase2/3-value-functions.html", mins: 20 },
      { id: "p2-4", title: "The Bellman equation", href: "phase2/4-bellman-equation.html", mins: 20 },
      { id: "p2-5", title: "Policy evaluation & policy iteration", href: "phase2/5-policy-iteration.html", mins: 20 },
      { id: "p2-6", title: "Value iteration", href: "phase2/6-value-iteration.html", mins: 20 },
      { id: "p2-r", title: "Recap + Lab 2: solve the gridworld", href: "phase2/recap.html", mins: 45 },
    ],
  },
  {
    id: "p3", num: 3, emoji: "👣", title: "Learning from experience (no map)", status: "ready",
    why: "Real agents don't get the rules of the world. Learn from trial runs: Monte Carlo, TD learning, SARSA and Q-learning.",
    lessons: [
      { id: "p3-1", title: "Monte Carlo: learn from complete episodes", href: "phase3/1-monte-carlo.html", mins: 15 },
      { id: "p3-2", title: "Temporal Difference (TD) learning", href: "phase3/2-td-learning.html", mins: 20 },
      { id: "p3-3", title: "SARSA: on-policy control", href: "phase3/3-sarsa.html", mins: 15 },
      { id: "p3-4", title: "Q-learning: off-policy control", href: "phase3/4-q-learning.html", mins: 15 },
      { id: "p3-5", title: "n-step returns & eligibility traces", href: "phase3/5-nstep-traces.html", mins: 20 },
      { id: "p3-r", title: "Recap + Lab 3: learning without a model", href: "phase3/recap.html", mins: 45 },
    ],
  },
  {
    id: "p4", num: 4, emoji: "🧠", title: "Deep RL: neural networks meet RL", status: "ready",
    why: "Tables don't scale to Atari or language. Swap them for neural networks, and learn the tricks that keep training stable.",
    lessons: [
      { id: "p4-1", title: "Function approximation: why tables fail", href: "phase4/1-function-approximation.html", mins: 15 },
      { id: "p4-2", title: "Neural nets & gradient descent refresher", href: "phase4/2-neural-nets.html", mins: 20 },
      { id: "p4-3", title: "DQN: replay buffer & target network", href: "phase4/3-dqn.html", mins: 20 },
      { id: "p4-4", title: "Double, Dueling, PER & Rainbow", href: "phase4/4-rainbow.html", mins: 15 },
      { id: "p4-r", title: "Recap + Lab 4: DQN on CartPole", href: "phase4/recap.html", mins: 60 },
    ],
  },
  {
    id: "p5", num: 5, emoji: "🎯", title: "Policy gradients → PPO", status: "ready",
    why: "The family behind LLM training. Learn the policy directly, build up from REINFORCE to PPO one idea at a time.",
    lessons: [
      { id: "p5-1", title: "Why learn the policy directly?", href: "phase5/1-why-policies.html", mins: 15 },
      { id: "p5-2", title: "REINFORCE & the log-derivative trick", href: "phase5/2-reinforce.html", mins: 20 },
      { id: "p5-3", title: "Baselines & the advantage", href: "phase5/3-baselines.html", mins: 15 },
      { id: "p5-4", title: "Actor-Critic & A2C", href: "phase5/4-actor-critic.html", mins: 20 },
      { id: "p5-5", title: "GAE: Generalized Advantage Estimation", href: "phase5/5-gae.html", mins: 20 },
      { id: "p5-6", title: "Trust regions: TRPO → PPO", href: "phase5/6-ppo.html", mins: 25 },
      { id: "p5-r", title: "Recap + Lab 5: REINFORCE & PPO from scratch", href: "phase5/recap.html", mins: 60 },
    ],
  },
  {
    id: "p6", num: 6, emoji: "🚀", title: "Advanced RL", status: "ready",
    why: "The rest of what an RL researcher should know: continuous control, model-based RL and search, offline RL, exploration and self-play.",
    lessons: [
      { id: "p6-1", title: "Continuous actions: DDPG, TD3", href: "phase6/1-ddpg-td3.html", mins: 15 },
      { id: "p6-2", title: "Max-entropy RL: SAC", href: "phase6/2-sac.html", mins: 15 },
      { id: "p6-3", title: "Model-based RL: Dyna, MuZero, Dreamer", href: "phase6/3-model-based.html", mins: 20 },
      { id: "p6-3b", title: "Planning by search: MCTS & AlphaZero", href: "phase6/3b-mcts.html", mins: 20 },
      { id: "p6-4", title: "Offline RL: CQL, IQL", href: "phase6/4-offline-rl.html", mins: 15 },
      { id: "p6-5", title: "Exploration: curiosity, RND", href: "phase6/5-exploration.html", mins: 15 },
      { id: "p6-6", title: "Self-play & multi-agent RL", href: "phase6/6-self-play.html", mins: 15 },
      { id: "p6-7", title: "Imitation & inverse RL", href: "phase6/7-imitation.html", mins: 15 },
      { id: "p6-8", title: "RL in the brain: dopamine & the striatum", href: "phase6/8-brain.html", mins: 15 },
      { id: "p6-r", title: "Recap + Lab 6: Dyna-Q + SAC", href: "phase6/recap.html", mins: 45 },
    ],
  },
  {
    id: "p7", num: 7, emoji: "💬", title: "RL for LLMs: alignment (RLHF & friends)", status: "ready",
    why: "How a text predictor becomes ChatGPT: reward models, RLHF with PPO, the KL leash, DPO and the preference-optimization family.",
    lessons: [
      { id: "p7-1", title: "An LLM is a policy: the token-level MDP", href: "phase7/1-llm-as-policy.html", mins: 15, verified: "2026-10" },
      { id: "p7-2", title: "The post-training pipeline: pretrain → SFT → RL", href: "phase7/2-pipeline.html", mins: 15, verified: "2026-10" },
      { id: "p7-3", title: "Reward models & Bradley–Terry", href: "phase7/3-reward-models.html", mins: 15, verified: "2026-10" },
      { id: "p7-4", title: "RLHF with PPO (InstructGPT) & the KL penalty", href: "phase7/4-rlhf-ppo.html", mins: 20, verified: "2026-10" },
      { id: "p7-5", title: "DPO and the preference family (IPO, KTO, SimPO, ORPO)", href: "phase7/5-dpo.html", mins: 20, verified: "2026-10" },
      { id: "p7-6", title: "RLAIF & Constitutional AI", href: "phase7/6-rlaif.html", mins: 15, verified: "2026-10" },
      { id: "p7-7", title: "Best-of-N, rejection sampling & expert iteration (STaR)", href: "phase7/7-best-of-n.html", mins: 15, verified: "2026-10" },
      { id: "p7-r", title: "Recap + Lab 7: reward model + DPO on a small LLM", href: "phase7/recap.html", mins: 45, verified: "2026-10" },
    ],
  },
  {
    id: "p8", num: "8A", emoji: "🧩", title: "Reasoning RL: verifiable rewards & GRPO", status: "ready", isNew: true,
    why: "How o1 and DeepSeek-R1 were trained: RL with checkable rewards, GRPO and its 2025–26 fixes, process rewards, reward hacking, entropy collapse, the pass@k debate and rubric rewards.",
    lessons: [
      { id: "p8-1", title: "o1 & DeepSeek-R1: RL teaches models to think", href: "phase8/1-o1-r1.html", mins: 15, verified: "2026-10" },
      { id: "p8-2", title: "RLVR: rewards you can check (math, code, tests)", href: "phase8/2-rlvr.html", mins: 15, verified: "2026-10" },
      { id: "p8-3", title: "GRPO & RLOO: dropping the critic", href: "phase8/3-grpo-rloo.html", mins: 20, verified: "2026-10" },
      { id: "p8-4", title: "Fixing GRPO: Dr. GRPO, DAPO, GSPO, CISPO & more", href: "phase8/4-grpo-fixes.html", mins: 20, verified: "2026-10" },
      { id: "p8-5", title: "Process vs outcome rewards (PRMs)", href: "phase8/5-process-rewards.html", mins: 15, verified: "2026-10" },
      { id: "p8-6", title: "Reward hacking & over-optimization", href: "phase8/6-reward-hacking.html", mins: 15, verified: "2026-10" },
      { id: "p8-7", title: "Entropy collapse & exploration in LLMs", href: "phase8/7-entropy-collapse.html", mins: 15, verified: "2026-10" },
      { id: "p8-8", title: "Does RL create new skills? The pass@k debate", href: "phase8/8-pass-at-k.html", mins: 15, verified: "2026-10" },
      { id: "p8-9", title: "Generative reward models & rubrics as rewards", href: "phase8/9-rubrics.html", mins: 15, verified: "2026-10" },
      { id: "p8-r", title: "Recap 8A + Lab 8: GRPO on a small LLM", href: "phase8/recap.html", mins: 60, verified: "2026-10" },
    ],
  },
  {
    id: "p8b", num: "8B", emoji: "🛠️", title: "Agentic RL: tools, environments & scale", status: "ready", isNew: true,
    why: "RL for models that act: multi-turn episodes, rollout infrastructure, compute scaling, coding agents, long-horizon environments and search agents.",
    lessons: [
      { id: "p8-10", title: "Multi-turn RL: when the LLM acts in a world", href: "phase8/10-agentic-rl.html", mins: 15, verified: "2026-10" },
      { id: "p8-11", title: "RL infrastructure: rollouts, async training & mismatches", href: "phase8/11-rl-infrastructure.html", mins: 15, verified: "2026-10" },
      { id: "p8-12", title: "Scaling RL: compute curves (ScaleRL)", href: "phase8/12-scaling-open-problems.html", mins: 15, verified: "2026-10" },
      { id: "p8-13", title: "RL for coding agents: the 2025–26 recipe", href: "phase8/13-coding-agents.html", mins: 20, verified: "2026-10" },
      { id: "p8-14", title: "Scaling agentic RL: environments, long horizons & real users", href: "phase8/14-scaling-agentic-rl.html", mins: 20, verified: "2026-10" },
      { id: "p8-15", title: "Search & deep-research agents", href: "phase8/15-search-agents.html", mins: 15, verified: "2026-10" },
      { id: "p8b-r", title: "Recap 8B: agents & the frontier", href: "phase8/recap-agents.html", mins: 20, verified: "2026-10" },
    ],
  },
  {
    id: "p9", num: 9, emoji: "🔬", title: "Becoming an RL researcher", status: "ready",
    why: "Turn knowledge into a career: read papers fast, reproduce results, run experiments properly, build a portfolio and pass interviews.",
    lessons: [
      { id: "p9-1", title: "How to read an RL paper in 30 minutes", href: "phase9/1-reading-papers.html", mins: 10 },
      { id: "p9-2", title: "Reproducing a paper end to end", href: "phase9/2-reproducing.html", mins: 15 },
      { id: "p9-3", title: "Experiment hygiene: seeds, error bars, ablations + Lab 9", href: "phase9/3-experiment-hygiene.html", mins: 20 },
      { id: "p9-4", title: "Writing it up & sharing your work", href: "phase9/4-writing-sharing.html", mins: 10 },
      { id: "p9-5", title: "Portfolio projects that get you hired", href: "phase9/5-portfolio.html", mins: 10 },
      { id: "p9-6", title: "RL researcher interview prep", href: "phase9/6-interview-prep.html", mins: 20 },
      { id: "p9-7", title: "Capstone: train a reasoning model with RL", href: "phase9/7-capstone.html", mins: 15 },
      { id: "p9-8", title: "Go deeper: courses, books, videos & tools", href: "phase9/8-go-deeper.html", mins: 10 },
      { id: "p9-9", title: "Mastery check: explain, frame, solve", href: "phase9/9-mastery-check.html", mins: 30 },
      { id: "p9-r", title: "Graduation: recap & what's next", href: "phase9/recap.html", mins: 20 },
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
  key: "rlcourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "p1-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "rlcourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- one-time moves of saved answers when quizzes move between pages (keys are lessonId#index) ---------- */
const QUIZ_MOVES = [
  { v: 1, map: { "p0-r#0": "p0-5#0", "p0-r#1": "p0-5#1", "p0-r#2": "p0-r#0", "p0-r#3": "p0-r#1" } }, // Oct 2026: stats → lesson 0.5
];
(() => {
  const key = "rlcourse.quizmoves", at = +Store.get(key) || 0;
  QUIZ_MOVES.filter(m => m.v > at).forEach(m => {
    const old = QuizStore.all(), out = { ...old };
    Object.keys(m.map).forEach(from => delete out[from]);
    Object.keys(m.map).forEach(from => { if (old[from]) out[m.map[from]] = old[from]; });
    Store.set(QuizStore.key, JSON.stringify(out));
    Store.set(key, String(m.v));
  });
})();

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "rlcourse.theme";
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

function fmtMonth(ym) { // "2026-10" -> "Oct 2026"
  const m = /^(\d{4})-(\d{2})$/.exec(ym || "");
  if (!m) return ym;
  return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+m[2] - 1] + " " + m[1];
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
    <a class="logo" href="${root}/index.html">RL <span>from Zero</span></a>
    <div class="crumb">${cur ? `Phase ${cur.phase.num} · ${cur.phase.title}` : ""}</div>
    <button class="theme-btn" title="Toggle theme" aria-label="Toggle light / dark theme">🌓</button>`;

  const side = document.createElement("nav");
  side.className = "sidebar";
  side.setAttribute("aria-label", "Course lessons");
  side.innerHTML = COURSE.map(p => {
    const open = cur && cur.phase.id === p.id;
    const head = (ready, extra) => `<button type="button" class="phase-title" aria-expanded="${open}" aria-controls="side-${p.id}">` +
      `<span class="chev" aria-hidden="true">▸</span><span class="pt">${p.emoji} ${p.num}. ${p.title}</span>${extra}</button>`;
    if (p.status !== "ready")
      return `<div class="side-phase${open ? "" : " collapsed"}">${head(false, "")}<div class="phase-lessons" id="side-${p.id}"><div class="locked"><span class="tick">🔒</span>Coming soon</div></div></div>`;
    const n = p.lessons.filter(l => done.has(l.id)).length;
    return `<div class="side-phase${open ? "" : " collapsed"}">` +
      head(true, `<span class="pcount">${n}/${p.lessons.length}</span>`) +
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
      `<span>🧭 Phase ${cur.phase.num} · ${k} of ${phaseLessons.length}</span>` +
      (cur.verified ? `<span title="Facts and links in this lesson were last checked in ${fmtMonth(cur.verified)}">🔎 Last reviewed ${fmtMonth(cur.verified)}</span>` : "");
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
window.RLCourse = {
  COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("rlcourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "rl-from-zero", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like an RL from Zero progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("rlcourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
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

/* ---------- RL helpers for playgrounds ---------- */
window.RL = {
  randn() { // standard normal (Box–Muller)
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  },
  bernoulli(p) { return Math.random() < p ? 1 : 0; },
  argmax(arr) { // random tie-break, like real implementations should
    let best = -Infinity, idx = [];
    arr.forEach((v, i) => { if (v > best) { best = v; idx = [i]; } else if (v === best) idx.push(i); });
    return idx[Math.floor(Math.random() * idx.length)];
  },
  randint(n) { return Math.floor(Math.random() * n); },
  gamma(k) { // Marsaglia–Tsang
    if (k < 1) return this.gamma(k + 1) * Math.pow(Math.random(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = this.randn(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  },
  sampleBeta(a, b) { const x = this.gamma(a), y = this.gamma(b); return x / (x + y); },
  lgamma(z) { // Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - this.lgamma(1 - z);
    z -= 1; let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  },
  betaPdf(x, a, b) {
    if (x <= 0 || x >= 1) return 0;
    return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + this.lgamma(a + b) - this.lgamma(a) - this.lgamma(b));
  },
  css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); },
};

document.addEventListener("DOMContentLoaded", () => {
  if (document.body.dataset.lesson) buildShell();
  bindThemeButtons();
  initQuizzes();
});

/* ============ System Design at Scale — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index) get theme + quizzes only.
 * Also exposes window.MX: small math/random helpers used by playgrounds,
 * and window.SDCourse: progress export/import + small helpers.
 * Storage keys start with "sdcourse." so progress never mixes with the sibling courses.
 */

const COURSE = [
  {
    id: "s0", num: 0, emoji: "🌱", title: "Foundations: how big systems think", status: "ready",
    why: "How a senior engineer approaches a design: requirements, trade-offs, back-of-the-envelope math, what really happens on the network, APIs, and what “highly available” actually means.",
    lessons: [
      { id: "s0-1", title: "What system design is (and how seniors think)", href: "phase0/1-how-seniors-think.html", mins: 15 },
      { id: "s0-2", title: "What happens when you hit Enter: DNS, TCP, TLS, HTTP", href: "phase0/2-url-to-response.html", mins: 20 },
      { id: "s0-3", title: "Latency numbers & back-of-the-envelope estimation", href: "phase0/3-estimation.html", mins: 20 },
      { id: "s0-4", title: "APIs: REST, gRPC, GraphQL, WebSockets", href: "phase0/4-apis.html", mins: 20 },
      { id: "s0-5", title: "Availability, SLIs, SLOs, SLAs & the nines", href: "phase0/5-availability.html", mins: 15 },
      { id: "s0-r", title: "Recap: foundations", href: "phase0/recap.html", mins: 20 },
    ],
  },
  {
    id: "s1", num: 1, emoji: "⚖️", title: "Scaling the request path", status: "ready",
    why: "From one server to thousands: stateless services, load balancers, caching, CDNs, rate limiting and API gateways.",
    lessons: [
      { id: "s1-1", title: "Vertical vs horizontal scaling & stateless services", href: "phase1/1-scaling-out.html", mins: 15 },
      { id: "s1-2", title: "Load balancers: L4 vs L7, algorithms, health checks", href: "phase1/2-load-balancers.html", mins: 20 },
      { id: "s1-3", title: "Caching: the fastest request is the one you don't make", href: "phase1/3-caching.html", mins: 25 },
      { id: "s1-4", title: "CDNs & the edge", href: "phase1/4-cdn.html", mins: 15 },
      { id: "s1-5", title: "Rate limiting: token buckets & friends", href: "phase1/5-rate-limiting.html", mins: 20 },
      { id: "s1-6", title: "Reverse proxies & API gateways (Nginx, Envoy, Kong)", href: "phase1/6-gateways.html", mins: 15 },
      { id: "s1-r", title: "Recap: scaling the request path", href: "phase1/recap.html", mins: 20 },
    ],
  },
  {
    id: "s2", num: 2, emoji: "🗄️", title: "Data at scale", status: "ready",
    why: "Choosing and scaling databases: SQL vs NoSQL, indexes, replication, sharding, consistent hashing, transactions, CAP, and specialised stores.",
    lessons: [
      { id: "s2-1", title: "SQL vs NoSQL & data modeling", href: "phase2/1-sql-nosql.html", mins: 20 },
      { id: "s2-2", title: "Indexes & query performance (B-trees, LSM trees)", href: "phase2/2-indexes.html", mins: 20 },
      { id: "s2-3", title: "Replication: leaders, followers & replication lag", href: "phase2/3-replication.html", mins: 20 },
      { id: "s2-4", title: "Sharding & consistent hashing", href: "phase2/4-sharding.html", mins: 20 },
      { id: "s2-5", title: "Transactions, ACID & isolation levels", href: "phase2/5-transactions.html", mins: 20 },
      { id: "s2-6", title: "CAP, PACELC & consistency models", href: "phase2/6-cap.html", mins: 20 },
      { id: "s2-7", title: "Object storage, search, time-series & vector stores", href: "phase2/7-specialized-stores.html", mins: 15 },
      { id: "s2-r", title: "Recap: data at scale", href: "phase2/recap.html", mins: 20 },
    ],
  },
  {
    id: "s3", num: 3, emoji: "📨", title: "Async & distributed systems", status: "ready",
    why: "Queues, Kafka, idempotency and retries, consensus, distributed transactions and data pipelines: what happens when work leaves the request path.",
    lessons: [
      { id: "s3-1", title: "Message queues & pub/sub", href: "phase3/1-queues.html", mins: 20 },
      { id: "s3-2", title: "Kafka & event streaming", href: "phase3/2-kafka.html", mins: 20 },
      { id: "s3-2b", title: "Event sourcing & CQRS", href: "phase3/2b-event-sourcing-cqrs.html", mins: 20 },
      { id: "s3-3", title: "Retries, idempotency & exactly-once", href: "phase3/3-idempotency.html", mins: 20 },
      { id: "s3-4", title: "Consensus & leader election (Raft, etcd)", href: "phase3/4-consensus.html", mins: 20 },
      { id: "s3-4b", title: "Time without a leader: clocks, vector clocks & gossip", href: "phase3/4b-clocks-gossip.html", mins: 20 },
      { id: "s3-5", title: "Distributed transactions: 2PC, sagas & the outbox", href: "phase3/5-distributed-transactions.html", mins: 20 },
      { id: "s3-6", title: "Batch & stream processing (Spark, Flink)", href: "phase3/6-pipelines.html", mins: 15 },
      { id: "s3-r", title: "Recap: async & distributed systems", href: "phase3/recap.html", mins: 20 },
    ],
  },
  {
    id: "s4", num: 4, emoji: "🧩", title: "Microservices architecture", status: "ready",
    why: "When to split a monolith, where to draw service boundaries, how services talk, how they survive each other's failures, API evolution and auth at scale.",
    lessons: [
      { id: "s4-1", title: "Monolith vs microservices (and when not to split)", href: "phase4/1-monolith-vs-micro.html", mins: 20 },
      { id: "s4-2", title: "Service boundaries & domain-driven design", href: "phase4/2-boundaries.html", mins: 20 },
      { id: "s4-3", title: "Service communication, discovery & service mesh", href: "phase4/3-communication.html", mins: 20 },
      { id: "s4-4", title: "Resilience: timeouts, retries, circuit breakers, bulkheads", href: "phase4/4-resilience.html", mins: 20 },
      { id: "s4-5", title: "API design, versioning & backward compatibility", href: "phase4/5-api-evolution.html", mins: 15 },
      { id: "s4-6", title: "Auth at scale: OAuth2, OIDC, JWT, mTLS", href: "phase4/6-auth.html", mins: 20 },
      { id: "s4-6b", title: "Browser security for backends: CORS, CSRF & cookies", href: "phase4/6b-cors-csrf.html", mins: 20 },
      { id: "s4-r", title: "Recap: microservices architecture", href: "phase4/recap.html", mins: 20 },
    ],
  },
  {
    id: "s5", num: 5, emoji: "🐳", title: "Containers, Kubernetes & the cloud", status: "ready",
    why: "Docker, Kubernetes from pods to autoscaling, infrastructure as code with Terraform/Helm/GitOps, and the cloud building blocks every company uses.",
    lessons: [
      { id: "s5-1", title: "Docker: images, layers & containers", href: "phase5/1-docker.html", mins: 20 },
      { id: "s5-2", title: "Kubernetes core: pods, deployments, services", href: "phase5/2-kubernetes.html", mins: 25 },
      { id: "s5-3", title: "Kubernetes in production: autoscaling, config, ingress", href: "phase5/3-k8s-production.html", mins: 20 },
      { id: "s5-4", title: "Infrastructure as code: Terraform, Helm & GitOps", href: "phase5/4-iac.html", mins: 20 },
      { id: "s5-5", title: "Cloud building blocks & serverless", href: "phase5/5-cloud.html", mins: 20 },
      { id: "s5-r", title: "Recap: containers, Kubernetes & the cloud", href: "phase5/recap.html", mins: 20 },
    ],
  },
  {
    id: "s6", num: 6, emoji: "📈", title: "Shipping & running it", status: "ready",
    why: "CI/CD, safe deployments, observability with Prometheus/Grafana/OpenTelemetry, alerting and incidents, capacity and cost, and production security.",
    lessons: [
      { id: "s6-1", title: "CI/CD pipelines", href: "phase6/1-cicd.html", mins: 15 },
      { id: "s6-1b", title: "Versioning & releases: SemVer, lockfiles & changelogs", href: "phase6/1b-semver-releases.html", mins: 15 },
      { id: "s6-2", title: "Deploying safely: rolling, blue-green, canary, feature flags", href: "phase6/2-deployments.html", mins: 20 },
      { id: "s6-3", title: "Observability: logs, metrics, traces (Prometheus, Grafana, OpenTelemetry)", href: "phase6/3-observability.html", mins: 25 },
      { id: "s6-4", title: "Alerting, on-call, incidents & postmortems", href: "phase6/4-incidents.html", mins: 20 },
      { id: "s6-5", title: "Capacity planning, load testing & cost", href: "phase6/5-capacity-cost.html", mins: 20 },
      { id: "s6-6", title: "Security in production: secrets, OWASP, DDoS", href: "phase6/6-security.html", mins: 20 },
      { id: "s6-r", title: "Recap: shipping & running it", href: "phase6/recap.html", mins: 20 },
    ],
  },
  {
    id: "s7", num: 7, emoji: "🏗️", title: "Designing real systems", status: "ready",
    why: "Put it all together on the classic designs: a repeatable framework, then URL shortener, news feed, chat, video streaming, ride sharing, payments and notifications.",
    lessons: [
      { id: "s7-1", title: "The design framework: from vague prompt to architecture", href: "phase7/1-framework.html", mins: 20 },
      { id: "s7-2", title: "Design a URL shortener", href: "phase7/2-url-shortener.html", mins: 25 },
      { id: "s7-3", title: "Design a news feed (fan-out)", href: "phase7/3-news-feed.html", mins: 25 },
      { id: "s7-4", title: "Design a chat system (WhatsApp)", href: "phase7/4-chat.html", mins: 25 },
      { id: "s7-5", title: "Design video streaming (YouTube/Netflix)", href: "phase7/5-video.html", mins: 25 },
      { id: "s7-6", title: "Design ride sharing (Uber): geo at scale", href: "phase7/6-ride-sharing.html", mins: 25 },
      { id: "s7-7", title: "Design a payment system", href: "phase7/7-payments.html", mins: 25 },
      { id: "s7-8", title: "Design a notification system", href: "phase7/8-notifications.html", mins: 20 },
      { id: "s7-r", title: "Recap: designing real systems", href: "phase7/recap.html", mins: 20 },
    ],
  },
  {
    id: "s8", num: 8, emoji: "🧭", title: "Becoming the senior / lead engineer", status: "ready",
    why: "Design docs and ADRs, trade-off leadership, the founding-engineer playbook, SRE and reliability culture, and how big tech runs senior system design interviews.",
    lessons: [
      { id: "s8-1", title: "Design docs, RFCs & decision records", href: "phase8/1-design-docs.html", mins: 15 },
      { id: "s8-2", title: "Trade-offs, tech debt & build vs buy", href: "phase8/2-tradeoffs-leadership.html", mins: 15 },
      { id: "s8-3", title: "The founding-engineer playbook: 0 → 1 → 100", href: "phase8/3-founding-engineer.html", mins: 20 },
      { id: "s8-4", title: "SRE culture: error budgets & chaos engineering", href: "phase8/4-sre.html", mins: 20 },
      { id: "s8-5", title: "The senior system design interview", href: "phase8/5-interview.html", mins: 20 },
      { id: "s8-r", title: "Graduation: the senior engineer's map", href: "phase8/recap.html", mins: 20 },
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
  key: "sdcourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "s1-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "sdcourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "sdcourse.theme";
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
    <a class="logo" href="${root}/index.html">System Design <span>at Scale</span></a>
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
window.SDCourse = {
  COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("sdcourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "system-design-at-scale", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like a System Design progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("sdcourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
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

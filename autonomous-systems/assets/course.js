/* ============ Autonomous Systems — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index) get theme + quizzes only.
 * Also exposes window.MX: small math/random helpers used by playgrounds,
 * and window.ASCourse: progress export/import + small helpers.
 * Storage keys start with "ascourse." so progress never mixes with the sibling courses.
 * Lesson ids: "s<phase>-<k>" (e.g. s3-3), recaps "s<phase>-r".
 */

const COURSE = [
  {
    id: "s0", num: 0, emoji: "🔁", title: "The autonomy loop", status: "ready",
    why: "Cars, drones and robots all run the same loop: sense, perceive, localize, predict, plan, control. The SAE levels, the modular vs end-to-end debate, how Tesla, Waymo and drone makers differ, and why the last 1% is the hard part.",
    lessons: [
      { id: "s0-1", title: "What makes a machine autonomous: the sense → act loop", href: "phase0/1-autonomy-loop.html", mins: 20 },
      { id: "s0-2", title: "SAE levels 0–5: who is driving?", href: "phase0/2-sae-levels.html", mins: 15 },
      { id: "s0-3", title: "The autonomy stack, module by module", href: "phase0/3-the-stack.html", mins: 20 },
      { id: "s0-4", title: "Modular vs end-to-end: the great debate", href: "phase0/4-modular-vs-e2e.html", mins: 20 },
      { id: "s0-5", title: "Tesla vs Waymo vs drones at a glance", href: "phase0/5-three-approaches.html", mins: 20 },
      { id: "s0-6", title: "Why autonomy is hard: the long tail & safety", href: "phase0/6-long-tail.html", mins: 20 },
      { id: "s0-r", title: "Recap: The autonomy loop", href: "phase0/recap.html", mins: 25 },
    ],
  },
  {
    id: "s1", num: 1, emoji: "📐", title: "Spatial math: frames & rotations", status: "ready",
    why: "Before a robot can know where anything is, it needs a language for “where” and “which way”: coordinate frames, rotation matrices, Euler angles and gimbal lock, quaternions, and transforms you can chain, as ROS 2 tf2 does.",
    lessons: [
      { id: "s1-1", title: "Coordinate frames: world, body, sensor", href: "phase1/1-frames.html", mins: 20 },
      { id: "s1-2", title: "2D rotations & rotation matrices", href: "phase1/2-rotations-2d.html", mins: 20 },
      { id: "s1-3", title: "3D rotations: Euler angles & gimbal lock", href: "phase1/3-euler-gimbal.html", mins: 25 },
      { id: "s1-4", title: "Quaternions: why drones use them", href: "phase1/4-quaternions.html", mins: 25 },
      { id: "s1-5", title: "Homogeneous transforms: chaining frames", href: "phase1/5-transforms.html", mins: 25 },
      { id: "s1-6", title: "ROS 2 & tf2: the transform tree", href: "phase1/6-tf2.html", mins: 20 },
      { id: "s1-r", title: "Recap: Spatial math", href: "phase1/recap.html", mins: 25 },
    ],
  },
  {
    id: "s2", num: 2, emoji: "📡", title: "Sensors: how machines measure the world", status: "ready",
    why: "Cameras, lidar, radar, ultrasonics, IMUs, GNSS and wheel encoders: what each one actually measures, how it fails, how they're calibrated and time-synced, and why Tesla, Waymo and drone makers pick different sets.",
    lessons: [
      { id: "s2-1", title: "Cameras: the pinhole model", href: "phase2/1-pinhole.html", mins: 25 },
      { id: "s2-2", title: "Intrinsics, extrinsics, distortion & calibration", href: "phase2/2-camera-calibration.html", mins: 25 },
      { id: "s2-3", title: "Lidar: time of flight & point clouds", href: "phase2/3-lidar.html", mins: 25 },
      { id: "s2-4", title: "Radar: Doppler & all-weather sensing", href: "phase2/4-radar.html", mins: 20 },
      { id: "s2-5", title: "IMU: gyros, accelerometers, bias & drift", href: "phase2/5-imu.html", mins: 25 },
      { id: "s2-6", title: "GNSS & RTK, wheel odometry & ultrasonics", href: "phase2/6-gnss-odometry.html", mins: 20 },
      { id: "s2-7", title: "Time sync, cross-calibration & the sensor trade-off table", href: "phase2/7-sync-tradeoffs.html", mins: 20 },
      { id: "s2-r", title: "Recap: Sensors", href: "phase2/recap.html", mins: 25 },
    ],
  },
  {
    id: "s3", num: 3, emoji: "🎯", title: "State estimation: knowing where you are", status: "ready",
    why: "Every sensor is noisy and every model drifts. Belief and uncertainty, the Bayes filter, the Kalman filter and its nonlinear cousins (EKF, UKF), particle filters, real sensor fusion, and how to survive bad measurements.",
    lessons: [
      { id: "s3-1", title: "Belief & uncertainty: thinking in distributions", href: "phase3/1-belief.html", mins: 20 },
      { id: "s3-2", title: "The Bayes filter: predict, then correct", href: "phase3/2-bayes-filter.html", mins: 25 },
      { id: "s3-3", title: "The Kalman filter", href: "phase3/3-kalman-filter.html", mins: 35 },
      { id: "s3-4", title: "Nonlinear filters: EKF & UKF", href: "phase3/4-ekf-ukf.html", mins: 30 },
      { id: "s3-5", title: "Particle filters & Monte Carlo localization", href: "phase3/5-particle-filter.html", mins: 30 },
      { id: "s3-6", title: "Sensor fusion in practice: GPS + IMU & visual-inertial", href: "phase3/6-sensor-fusion.html", mins: 25 },
      { id: "s3-7", title: "Outliers & robustness: gating, delays & failures", href: "phase3/7-outliers.html", mins: 20 },
      { id: "s3-r", title: "Recap: State estimation", href: "phase3/recap.html", mins: 25 },
    ],
  },
  {
    id: "s4", num: 4, emoji: "👁️", title: "Geometric vision: 3D from pictures", status: "ready",
    why: "How a camera recovers depth and motion without a neural network: features and matching, RANSAC, epipolar geometry, stereo disparity, optical flow, visual odometry, triangulation, bundle adjustment and structure from motion.",
    lessons: [
      { id: "s4-1", title: "Features & matching: corners & ORB", href: "phase4/1-features.html", mins: 25 },
      { id: "s4-2", title: "RANSAC: fitting models when data lies", href: "phase4/2-ransac.html", mins: 25 },
      { id: "s4-3", title: "Epipolar geometry: two views of one world", href: "phase4/3-epipolar.html", mins: 25 },
      { id: "s4-4", title: "Stereo depth & disparity", href: "phase4/4-stereo.html", mins: 25 },
      { id: "s4-5", title: "Optical flow", href: "phase4/5-optical-flow.html", mins: 20 },
      { id: "s4-6", title: "Visual odometry & triangulation", href: "phase4/6-visual-odometry.html", mins: 25 },
      { id: "s4-7", title: "Bundle adjustment & structure from motion", href: "phase4/7-bundle-adjustment.html", mins: 25 },
      { id: "s4-r", title: "Recap: Geometric vision", href: "phase4/recap.html", mins: 25 },
    ],
  },
  {
    id: "s5", num: 5, emoji: "🗺️", title: "Mapping & localization", status: "ready",
    why: "Building a map while finding yourself in it: occupancy grids, SLAM from EKF-SLAM to pose graphs and loop closure, lidar and visual SLAM in practice, the HD-map vs mapless debate, and localizing against a prior map.",
    lessons: [
      { id: "s5-1", title: "Occupancy grids", href: "phase5/1-occupancy-grid.html", mins: 25 },
      { id: "s5-2", title: "SLAM: the chicken-and-egg problem", href: "phase5/2-slam-problem.html", mins: 25 },
      { id: "s5-3", title: "Graph SLAM, pose graphs & loop closure", href: "phase5/3-graph-slam.html", mins: 30 },
      { id: "s5-4", title: "SLAM in practice: ICP, lidar SLAM & ORB-SLAM", href: "phase5/4-slam-in-practice.html", mins: 25 },
      { id: "s5-5", title: "HD maps vs mapless driving", href: "phase5/5-hd-maps.html", mins: 25 },
      { id: "s5-6", title: "Localizing against a map", href: "phase5/6-map-localization.html", mins: 20 },
      { id: "s5-r", title: "Recap: Mapping & localization", href: "phase5/recap.html", mins: 25 },
    ],
  },
  {
    id: "s6", num: 6, emoji: "🧠", title: "Learned 3D perception", status: "ready",
    why: "Neural networks that turn raw sensor data into a 3D picture of the world: lidar detectors, camera-to-bird's-eye-view, occupancy networks, lanes and road topology, multi-object tracking, sensor fusion, monocular depth, and the data engines that feed them.",
    lessons: [
      { id: "s6-1", title: "3D object detection from lidar: PointNet → PointPillars", href: "phase6/1-lidar-detection.html", mins: 30 },
      { id: "s6-2", title: "Cameras to bird's-eye view: Lift-Splat & BEVFormer", href: "phase6/2-bev.html", mins: 30 },
      { id: "s6-3", title: "Occupancy networks", href: "phase6/3-occupancy-networks.html", mins: 25 },
      { id: "s6-4", title: "Lanes & road topology", href: "phase6/4-lanes-topology.html", mins: 20 },
      { id: "s6-5", title: "Multi-object tracking: association, Hungarian, SORT & ByteTrack", href: "phase6/5-tracking.html", mins: 30 },
      { id: "s6-6", title: "Early vs late fusion & monocular depth", href: "phase6/6-fusion-depth.html", mins: 25 },
      { id: "s6-7", title: "Auto-labeling & the data engine", href: "phase6/7-data-engine.html", mins: 20 },
      { id: "s6-r", title: "Recap: Learned 3D perception", href: "phase6/recap.html", mins: 25 },
    ],
  },
  {
    id: "s7", num: 7, emoji: "🧭", title: "Prediction & planning", status: "ready",
    why: "Guessing what everyone else will do, then choosing what to do: multimodal motion prediction, graph search (Dijkstra, A*, Hybrid A*), sampling planners (RRT, RRT*), lattice and Frenet planners, behavior and cost functions, and trajectory optimization with interaction.",
    lessons: [
      { id: "s7-1", title: "Predicting other agents: multimodal futures", href: "phase7/1-prediction.html", mins: 30 },
      { id: "s7-2", title: "Path search: Dijkstra & A*", href: "phase7/2-graph-search.html", mins: 30 },
      { id: "s7-3", title: "Planning for cars: Hybrid A* & kinematic limits", href: "phase7/3-hybrid-astar.html", mins: 25 },
      { id: "s7-4", title: "Sampling planners: RRT & RRT*", href: "phase7/4-rrt.html", mins: 25 },
      { id: "s7-5", title: "Lattice & Frenet planners", href: "phase7/5-lattice-frenet.html", mins: 25 },
      { id: "s7-6", title: "Behavior planning & cost functions", href: "phase7/6-behavior-cost.html", mins: 25 },
      { id: "s7-7", title: "Trajectory optimization & interaction", href: "phase7/7-trajectory-opt.html", mins: 25 },
      { id: "s7-r", title: "Recap: Prediction & planning", href: "phase7/recap.html", mins: 25 },
    ],
  },
  {
    id: "s8", num: 8, emoji: "🎛️", title: "Control & dynamics", status: "ready",
    why: "Turning a plan into steering, throttle and rotor speeds: PID, the kinematic bicycle model, pure pursuit and Stanley, LQR, MPC, quadrotor dynamics, cascaded flight control in PX4 and ArduPilot, and drone autonomy.",
    lessons: [
      { id: "s8-1", title: "PID control", href: "phase8/1-pid.html", mins: 30 },
      { id: "s8-2", title: "The kinematic bicycle model", href: "phase8/2-bicycle-model.html", mins: 20 },
      { id: "s8-3", title: "Path tracking: pure pursuit & Stanley", href: "phase8/3-path-tracking.html", mins: 25 },
      { id: "s8-4", title: "LQR: optimal feedback", href: "phase8/4-lqr.html", mins: 25 },
      { id: "s8-5", title: "Model predictive control", href: "phase8/5-mpc.html", mins: 30 },
      { id: "s8-6", title: "Quadrotor dynamics: why four rotors", href: "phase8/6-quadrotor.html", mins: 25 },
      { id: "s8-7", title: "Cascaded flight control & the PX4 / ArduPilot stack", href: "phase8/7-flight-stack.html", mins: 25 },
      { id: "s8-8", title: "Drone autonomy: avoidance, VIO & swarms", href: "phase8/8-drone-autonomy.html", mins: 25 },
      { id: "s8-r", title: "Recap: Control & dynamics", href: "phase8/recap.html", mins: 25 },
    ],
  },
  {
    id: "s9", num: 9, emoji: "🚀", title: "Learning-based autonomy, safety & scale", status: "ready",
    why: "Learning to drive from data and proving it's safe: imitation learning and DAgger, end-to-end driving, world models and generative simulation, sim-to-real, the fleet data engine, safety validation and standards, remote assistance, and where to go next.",
    lessons: [
      { id: "s9-1", title: "Imitation learning, covariate shift & DAgger", href: "phase9/1-imitation.html", mins: 25 },
      { id: "s9-2", title: "End-to-end driving: photons to controls", href: "phase9/2-end-to-end.html", mins: 30 },
      { id: "s9-3", title: "World models & generative simulation", href: "phase9/3-world-models.html", mins: 25 },
      { id: "s9-4", title: "Simulation & sim-to-real", href: "phase9/4-simulation.html", mins: 25 },
      { id: "s9-5", title: "The fleet data engine", href: "phase9/5-fleet-data.html", mins: 20 },
      { id: "s9-6", title: "Safety validation: scenarios, metrics & standards", href: "phase9/6-safety.html", mins: 30 },
      { id: "s9-7", title: "Remote assistance & operating a fleet", href: "phase9/7-operations.html", mins: 20 },
      { id: "s9-r", title: "Graduation: careers & where to go next", href: "phase9/recap.html", mins: 30 },
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
  key: "ascourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "a3-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "ascourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "ascourse.theme";
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
    <a class="logo" href="${root}/index.html">Autonomous <span>Systems</span></a>
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
window.ASCourse = {
  COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("ascourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "autonomous-systems", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like an Autonomous Systems progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("ascourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
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
  // ---- small linear algebra for filters (matrices are arrays of rows) ----
  eye(n) { return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))); },
  madd(A, B) { return A.map((r, i) => r.map((v, j) => v + B[i][j])); },
  msub(A, B) { return A.map((r, i) => r.map((v, j) => v - B[i][j])); },
  mscale(A, k) { return A.map(r => r.map(v => v * k)); },
  inv(M) { // Gauss–Jordan inverse of a small square matrix
    const n = M.length, A = M.map((r, i) => [...r, ...this.eye(n)[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]];
      const d = A[c][c] || 1e-12; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
      for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
    }
    return A.map(r => r.slice(n));
  },
  gauss(x, mu, sigma) { return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)); }, // normal pdf
  wrap(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }, // angle to (−π, π]
  deg(r) { return r * 180 / Math.PI; },
  rad(d) { return d * Math.PI / 180; },
  lerp(a, b, t) { return a + (b - a) * t; },
  clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); },
  fmt(v, d) { return (+v).toFixed(d == null ? 2 : d); },
  css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); },
};

document.addEventListener("DOMContentLoaded", () => {
  if (document.body.dataset.lesson) buildShell();
  bindThemeButtons();
  initQuizzes();
});

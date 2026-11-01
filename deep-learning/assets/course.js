/* ============ Deep Learning — course runtime ============
 * Builds the top bar, sidebar, prev/next nav, progress tracking, quizzes,
 * theme toggle. Each page sets <body data-lesson="id" data-root="..">.
 * Pages without data-lesson (index) get theme + quizzes only.
 * Also exposes window.MX: small math/random helpers used by playgrounds,
 * and window.DLCourse: progress export/import + small helpers.
 * Storage keys start with "dlcourse." so progress never mixes with the sibling courses.
 * The COURSE manifest below is the source of truth for lessons (ids d<phase>-<k>, recaps d<phase>-r).
 */

const COURSE = [
  {
    id: "d0", num: 0, emoji: "🧠", title: "Neurons & networks", status: "ready",
    why: "From a linear model to a neuron, why one neuron can't do XOR, why depth helps, and every activation function with the problem it solved.",
    lessons: [
      { id: "d0-1", title: "From a linear model to a neuron", href: "phase0/1-neuron.html", mins: 20 },
      { id: "d0-2", title: "The perceptron and the XOR problem", href: "phase0/2-perceptron-xor.html", mins: 20 },
      { id: "d0-3", title: "Multi-layer networks: why depth helps", href: "phase0/3-mlp-depth.html", mins: 25 },
      { id: "d0-4", title: "Activations I: step, sigmoid, tanh, ReLU (and why ReLU won)", href: "phase0/4-activations.html", mins: 35 },
      { id: "d0-5", title: "Activations II: Leaky, ELU, GELU, SiLU, SwiGLU", href: "phase0/5-modern-activations.html", mins: 30 },
      { id: "d0-6", title: "Output layers: sigmoid, softmax and what a network “says”", href: "phase0/6-softmax-outputs.html", mins: 25 },
      { id: "d0-r", title: "Recap: Neurons & networks", href: "phase0/recap.html", mins: 25 },
    ],
  },
  {
    id: "d1", num: 1, emoji: "🏋️", title: "Training a network", status: "ready",
    why: "Losses, backprop with autograd, the PyTorch training loop, initialization, optimizers, the learning rate and its schedule, and batch size.",
    lessons: [
      { id: "d1-1", title: "Loss functions: MSE, cross-entropy, and why cross-entropy for classes", href: "phase1/1-losses.html", mins: 30 },
      { id: "d1-2", title: "Backprop and autograd: how every weight learns its blame", href: "phase1/2-backprop-autograd.html", mins: 30 },
      { id: "d1-3", title: "PyTorch: tensors, modules and the training loop", href: "phase1/3-pytorch-loop.html", mins: 30 },
      { id: "d1-4", title: "Initialization: why random, why Xavier and He", href: "phase1/4-initialization.html", mins: 30 },
      { id: "d1-5", title: "Optimizers: SGD, momentum, Adam, AdamW", href: "phase1/5-optimizers.html", mins: 35 },
      { id: "d1-6", title: "The learning rate: warmup, cosine and finding it", href: "phase1/6-learning-rate.html", mins: 30 },
      { id: "d1-7", title: "Batch size: noise, speed and the scaling rules", href: "phase1/7-batch-size.html", mins: 25 },
      { id: "d1-r", title: "Recap: Training a network", href: "phase1/recap.html", mins: 25 },
    ],
  },
  {
    id: "d2", num: 2, emoji: "🔧", title: "Making it train", status: "ready",
    why: "Why deep networks refuse to learn and the fixes that made depth work: gradient clipping, residual connections, normalization, regularization, augmentation and systematic debugging.",
    lessons: [
      { id: "d2-1", title: "Vanishing and exploding gradients (and gradient clipping)", href: "phase2/1-vanishing-exploding.html", mins: 30 },
      { id: "d2-2", title: "Residual connections: the gradient highway", href: "phase2/2-residuals.html", mins: 30 },
      { id: "d2-3", title: "Normalization: BatchNorm → LayerNorm → RMSNorm", href: "phase2/3-normalization.html", mins: 35 },
      { id: "d2-4", title: "Regularization: dropout, weight decay, early stopping", href: "phase2/4-regularization.html", mins: 30 },
      { id: "d2-5", title: "Data augmentation: more data for free", href: "phase2/5-augmentation.html", mins: 25 },
      { id: "d2-6", title: "Debugging training: read the loss curve, overfit one batch", href: "phase2/6-debugging.html", mins: 35 },
      { id: "d2-r", title: "Recap: Making it train", href: "phase2/recap.html", mins: 25 },
    ],
  },
  {
    id: "d3", num: 3, emoji: "🖼️", title: "Convolutional networks", status: "ready",
    why: "Why an MLP drowns in pixels, how convolution shares weights, receptive fields and pooling, the CNN family tree from LeNet to ConvNeXt, transfer learning, and seeing what a CNN learned.",
    lessons: [
      { id: "d3-1", title: "Why not an MLP for images?", href: "phase3/1-why-cnns.html", mins: 20 },
      { id: "d3-2", title: "Convolution: filters, stride, padding, channels", href: "phase3/2-convolution.html", mins: 35 },
      { id: "d3-3", title: "Receptive fields and pooling", href: "phase3/3-receptive-field-pooling.html", mins: 25 },
      { id: "d3-4", title: "The CNN family tree: LeNet → ConvNeXt", href: "phase3/4-cnn-history.html", mins: 35 },
      { id: "d3-5", title: "Transfer learning and fine-tuning", href: "phase3/5-transfer-learning.html", mins: 25 },
      { id: "d3-6", title: "Seeing what a CNN learned", href: "phase3/6-visualizing-cnns.html", mins: 25 },
      { id: "d3-r", title: "Recap: Convolutional networks", href: "phase3/recap.html", mins: 25 },
    ],
  },
  {
    id: "d4", num: 4, emoji: "👁️", title: "Computer vision tasks", status: "ready",
    why: "Detection (IoU, NMS, mAP; R-CNN, YOLO, DETR), segmentation (U-Net, Mask R-CNN, SAM), Vision Transformers, CLIP and open-vocabulary vision, pose and video, and 3D vision basics.",
    lessons: [
      { id: "d4-1", title: "Vision tasks and detection basics: boxes, IoU, NMS, mAP", href: "phase4/1-detection-basics.html", mins: 35 },
      { id: "d4-2", title: "Object detectors: R-CNN → YOLO → DETR", href: "phase4/2-detectors.html", mins: 35 },
      { id: "d4-3", title: "Segmentation: FCN, U-Net, Mask R-CNN, panoptic", href: "phase4/3-segmentation.html", mins: 30 },
      { id: "d4-4", title: "Segment Anything: promptable segmentation (SAM)", href: "phase4/4-sam.html", mins: 25 },
      { id: "d4-5", title: "Vision Transformers: ViT, DeiT, Swin, DINOv2", href: "phase4/5-vit.html", mins: 35 },
      { id: "d4-6", title: "CLIP and open-vocabulary vision", href: "phase4/6-clip-open-vocab.html", mins: 30 },
      { id: "d4-7", title: "Pose, video and tracking", href: "phase4/7-pose-video.html", mins: 30 },
      { id: "d4-8", title: "3D vision basics: depth, point clouds, NeRF, Gaussian splats", href: "phase4/8-3d-vision.html", mins: 35 },
      { id: "d4-r", title: "Recap: Computer vision tasks", href: "phase4/recap.html", mins: 25 },
    ],
  },
  {
    id: "d5", num: 5, emoji: "📜", title: "Sequences & classic neural NLP", status: "ready",
    why: "Word embeddings, recurrent networks and why they forget, LSTM and GRU gates, seq2seq and the first attention, and 1D convolutions for text: the road to the transformer.",
    lessons: [
      { id: "d5-1", title: "Word embeddings: word2vec and GloVe", href: "phase5/1-word-embeddings.html", mins: 30 },
      { id: "d5-2", title: "Recurrent networks and backprop through time", href: "phase5/2-rnns.html", mins: 30 },
      { id: "d5-3", title: "LSTM and GRU: gates that remember", href: "phase5/3-lstm-gru.html", mins: 35 },
      { id: "d5-4", title: "Seq2seq and the first attention", href: "phase5/4-seq2seq-attention.html", mins: 30 },
      { id: "d5-5", title: "1D convolutions for text and sequences", href: "phase5/5-conv1d.html", mins: 20 },
      { id: "d5-r", title: "Recap: Sequences & classic NLP", href: "phase5/recap.html", mins: 25 },
    ],
  },
  {
    id: "d6", num: 6, emoji: "⚡", title: "Transformers", status: "ready",
    why: "Self-attention built step by step, multi-head attention, positions and RoPE, the transformer block, BERT vs GPT vs T5, tokenization, pretraining, scaling laws, efficient attention, mixture of experts and long context.",
    lessons: [
      { id: "d6-1", title: "Self-attention, step by step: Q, K, V and √d", href: "phase6/1-self-attention.html", mins: 40 },
      { id: "d6-2", title: "Multi-head attention", href: "phase6/2-multi-head.html", mins: 25 },
      { id: "d6-3", title: "Positional encodings: sinusoidal → RoPE", href: "phase6/3-positional.html", mins: 30 },
      { id: "d6-4", title: "The transformer block: pre-norm vs post-norm", href: "phase6/4-transformer-block.html", mins: 30 },
      { id: "d6-5", title: "Encoder, decoder, encoder–decoder: BERT vs GPT vs T5", href: "phase6/5-encoder-decoder.html", mins: 30 },
      { id: "d6-6", title: "Tokenization: BPE and friends", href: "phase6/6-tokenization.html", mins: 25 },
      { id: "d6-7", title: "Pretraining objectives: masked vs next-token", href: "phase6/7-pretraining.html", mins: 25 },
      { id: "d6-8", title: "Scaling laws: Kaplan → Chinchilla", href: "phase6/8-scaling-laws.html", mins: 30 },
      { id: "d6-9", title: "Efficient attention: KV cache, FlashAttention, GQA", href: "phase6/9-efficient-attention.html", mins: 35 },
      { id: "d6-10", title: "Mixture of experts", href: "phase6/10-moe.html", mins: 30 },
      { id: "d6-11", title: "Long context", href: "phase6/11-long-context.html", mins: 30 },
      { id: "d6-r", title: "Recap: Transformers", href: "phase6/recap.html", mins: 30 },
    ],
  },
  {
    id: "d7", num: 7, emoji: "🎨", title: "Generative models", status: "ready",
    why: "Autoencoders, VAEs, GANs, diffusion and latent diffusion, flow matching, classifier-free guidance, and autoregressive image and video models.",
    lessons: [
      { id: "d7-1", title: "Autoencoders: compress, then rebuild", href: "phase7/1-autoencoders.html", mins: 25 },
      { id: "d7-2", title: "VAEs and the reparameterization trick", href: "phase7/2-vae.html", mins: 35 },
      { id: "d7-3", title: "GANs and mode collapse", href: "phase7/3-gans.html", mins: 30 },
      { id: "d7-4", title: "Diffusion models: noise in, image out", href: "phase7/4-diffusion.html", mins: 35 },
      { id: "d7-5", title: "Latent diffusion and Stable Diffusion", href: "phase7/5-latent-diffusion.html", mins: 30 },
      { id: "d7-6", title: "Flow matching: straight paths from noise", href: "phase7/6-flow-matching.html", mins: 30 },
      { id: "d7-7", title: "Guidance: steering with classifier-free guidance", href: "phase7/7-guidance.html", mins: 25 },
      { id: "d7-8", title: "Autoregressive image and video models", href: "phase7/8-autoregressive-visual.html", mins: 30 },
      { id: "d7-r", title: "Recap: Generative models", href: "phase7/recap.html", mins: 25 },
    ],
  },
  {
    id: "d8", num: 8, emoji: "🎧", title: "Audio, speech & multimodal", status: "ready",
    why: "Sound as spectrograms, speech recognition with CTC and Whisper, text-to-speech, audio language models, and how an image becomes tokens for a vision-language model.",
    lessons: [
      { id: "d8-1", title: "Sound as pictures: waveforms and spectrograms", href: "phase8/1-spectrograms.html", mins: 30 },
      { id: "d8-2", title: "Speech recognition: CTC and Whisper", href: "phase8/2-speech-recognition.html", mins: 35 },
      { id: "d8-3", title: "Text-to-speech: from text to waveform", href: "phase8/3-tts.html", mins: 30 },
      { id: "d8-4", title: "Audio tokens and audio language models", href: "phase8/4-audio-llms.html", mins: 30 },
      { id: "d8-5", title: "Vision-language models: how an image becomes tokens", href: "phase8/5-vlms.html", mins: 35 },
      { id: "d8-r", title: "Recap: Audio, speech & multimodal", href: "phase8/recap.html", mins: 25 },
    ],
  },
  {
    id: "d9", num: 9, emoji: "🔭", title: "Frontier & robustness", status: "ready",
    why: "Self-supervised learning, graph neural networks, state-space models like Mamba, interpretability from saliency to sparse autoencoders, adversarial examples, and knowing when a model doesn't know.",
    lessons: [
      { id: "d9-1", title: "Self-supervised learning: SimCLR, MAE, DINO", href: "phase9/1-self-supervised.html", mins: 35 },
      { id: "d9-2", title: "Graph neural networks", href: "phase9/2-gnns.html", mins: 30 },
      { id: "d9-3", title: "State-space models: Mamba and hybrids", href: "phase9/3-ssm-mamba.html", mins: 35 },
      { id: "d9-4", title: "Interpretability: saliency, probes, circuits, SAEs", href: "phase9/4-interpretability.html", mins: 35 },
      { id: "d9-5", title: "Adversarial examples and robustness", href: "phase9/5-adversarial.html", mins: 30 },
      { id: "d9-6", title: "Uncertainty, calibration and out-of-distribution data", href: "phase9/6-uncertainty-ood.html", mins: 30 },
      { id: "d9-r", title: "Recap: Frontier & robustness", href: "phase9/recap.html", mins: 25 },
    ],
  },
  {
    id: "d10", num: 10, emoji: "🚀", title: "Scale & practice", status: "ready",
    why: "Why GPUs love matrix math, mixed precision, data/model/pipeline parallelism, quantization, pruning and distillation, exporting and deploying, experiment tracking, reading papers, and your career.",
    lessons: [
      { id: "d10-1", title: "GPU basics: compute, memory and why matmul is king", href: "phase10/1-gpu-basics.html", mins: 35 },
      { id: "d10-2", title: "Mixed precision: fp32, fp16, bf16, fp8", href: "phase10/2-mixed-precision.html", mins: 30 },
      { id: "d10-3", title: "Parallelism: DDP, FSDP/ZeRO, tensor and pipeline", href: "phase10/3-parallelism.html", mins: 35 },
      { id: "d10-4", title: "Efficiency: quantization, pruning, distillation", href: "phase10/4-efficiency.html", mins: 35 },
      { id: "d10-5", title: "Export and deploy: torch.compile, ONNX, TensorRT", href: "phase10/5-deploy.html", mins: 30 },
      { id: "d10-6", title: "Experiment tracking and reproducibility", href: "phase10/6-experiment-tracking.html", mins: 25 },
      { id: "d10-7", title: "Reading deep learning papers", href: "phase10/7-reading-papers.html", mins: 25 },
      { id: "d10-8", title: "Career, interviews and portfolio", href: "phase10/8-career.html", mins: 30 },
      { id: "d10-r", title: "Graduation: Deep learning, end to end", href: "phase10/recap.html", mins: 30 },
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
  key: "dlcourse.done",
  all() { const a = Store.json(this.key, []); return new Set(Array.isArray(a) ? a : []); },
  has(id) { return this.all().has(id); },
  set(id, on) {
    const s = this.all(); on ? s.add(id) : s.delete(id);
    Store.set(this.key, JSON.stringify([...s]));
  },
};

/* ---------- quiz results: { "a3-3#0": { pick: 2, ok: true }, ... } ---------- */
const QuizStore = {
  key: "dlcourse.quiz",
  all() { const o = Store.json(this.key, {}); return o && typeof o === "object" && !Array.isArray(o) ? o : {}; },
  get(id) { return this.all()[id]; },
  set(id, val) { const o = this.all(); if (val) o[id] = val; else delete o[id]; Store.set(this.key, JSON.stringify(o)); },
};

/* ---------- theme: saved choice, else follow the OS ---------- */
const THEME_KEY = "dlcourse.theme";
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
    <a class="logo" href="${root}/index.html">Deep <span>Learning</span></a>
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
window.DLCourse = {
  COURSE,
  flatLessons,
  toast,
  hours(mins) { return Math.round(mins / 6) / 10; }, // minutes -> hours, 1 decimal
  exportProgress() {
    const data = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.indexOf("dlcourse.") === 0 && k !== THEME_KEY) data[k] = localStorage.getItem(k);
      }
    } catch (e) { /* storage blocked: export what we have (nothing) */ }
    return JSON.stringify({ app: "deep-learning", version: 1, exported: new Date().toISOString(), data }, null, 2);
  },
  importProgress(json) { // merges; returns the number of keys imported
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    const data = obj && obj.data;
    if (!data || typeof data !== "object") throw new Error("This doesn't look like a Deep Learning progress file.");
    let n = 0;
    Object.keys(data).forEach(k => {
      const v = data[k];
      if (k.indexOf("dlcourse.") !== 0 || k === THEME_KEY || typeof v !== "string") return;
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

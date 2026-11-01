/* ============ AI: tiny offline NLP toolkit for playgrounds (no dependencies, no network) ============
 * Real systems use real tokenizers and embedding models. These are honest toys that behave the same way
 * qualitatively, so playgrounds can run instantly in the browser. Say so in the lesson text when you use them.
 *
 * AI.countTokens(text)          ≈ tokens (rule of thumb: ~4 characters of English per token)
 * AI.toyTokens(text)            → ["The", " cat", " sat", " on", " the", " mat", "."] (approximate pieces, for visuals)
 * AI.words(text, {stop})        → lowercase word list (stopwords removed unless stop: false)
 * AI.sentences(text)            → sentence strings
 * AI.embed(text)                → unit vector (Float64Array). Knows synonym groups ("money back" ≈ "refund"),
 *                                 plus hashed dimensions for every other word, so it is "semantic-ish".
 * AI.cosine(a, b)               → cosine similarity of two vectors
 * AI.bm25(docs, query, {k1, b}) → BM25 keyword score per doc (docs: array of strings)
 * AI.search(docs, query, {mode: "dense"|"bm25"|"hybrid", k}) → [{i, score}] best first (hybrid = RRF of both)
 * AI.rrf(rankings, k = 60)      → fused [{i, score}] from several ranked lists of indices (reciprocal rank fusion)
 * AI.mmr(qVec, docVecs, k, lambda = 0.7) → indices chosen by maximal marginal relevance (relevance vs diversity)
 * AI.chunk(text, {method: "fixed"|"sentence"|"paragraph"|"recursive", size: 60 tokens, overlap: 0})
 *                               → [{text, start, end}] character offsets into text
 * AI.concepts                   → the synonym groups behind AI.embed (extend with AI.addConcept([...words]))
 */
(function () {
  const STOP = new Set("a an the and or but if of to in on at for from by with without is are was were be been being do does did have has had i me my you your we our they their it its this that these those as so not no can could should would will just than then there here what which who whom how when where why about into over under up down out off".split(" "));

  // synonym groups: each group is one meaning dimension of the toy embedding
  const CONCEPTS = [
    ["refund", "refunds", "refunded", "money back", "reimburse", "reimbursement", "return", "returns", "returned"],
    ["cancel", "cancellation", "cancelled", "canceled", "terminate", "unsubscribe", "stop subscription"],
    ["price", "prices", "pricing", "cost", "costs", "fee", "fees", "charge", "charged", "billing", "bill", "invoice", "pay", "payment", "paid", "expensive", "cheap"],
    ["password", "passwords", "login", "log in", "sign in", "signin", "credentials", "locked out", "2fa", "two-factor", "reset"],
    ["shipping", "ship", "shipped", "delivery", "deliver", "delivered", "courier", "arrive", "arrives", "parcel", "package", "tracking"],
    ["error", "errors", "bug", "bugs", "crash", "crashes", "fail", "fails", "failed", "failure", "broken", "exception", "outage", "down"],
    ["warranty", "guarantee", "repair", "repairs", "defect", "defective", "faulty"],
    ["days", "day", "week", "weeks", "month", "months", "deadline", "within", "period", "window"],
    ["battery", "charging", "charger", "power", "battery life"],
    ["vacation", "leave", "holiday", "holidays", "pto", "time off", "annual leave"],
    ["salary", "pay raise", "compensation", "bonus", "payroll", "wages"],
    ["remote", "work from home", "wfh", "hybrid work", "office"],
    ["security", "secure", "breach", "attack", "vulnerability", "encryption", "encrypted"],
    ["privacy", "personal data", "pii", "gdpr", "consent"],
    ["dog", "dogs", "puppy", "puppies", "canine"],
    ["cat", "cats", "kitten", "kittens", "feline"],
    ["car", "cars", "vehicle", "vehicles", "automobile", "drive", "driving"],
    ["doctor", "physician", "medical", "medicine", "hospital", "clinic", "health"],
    ["capital", "city", "cities"],
    ["france", "french", "paris"],
    ["japan", "japanese", "tokyo"],
    ["model", "models", "llm", "llms", "language model", "gpt", "claude", "gemini"],
    ["database", "databases", "db", "sql", "postgres", "table", "tables"],
    ["vector", "vectors", "embedding", "embeddings"],
    ["fast", "faster", "speed", "quick", "quickly", "latency", "slow"],
    ["revenue", "sales", "income", "earnings", "profit"],
  ];
  const HASH_DIMS = 96;
  let conceptIndex = null; // phrase -> concept id
  function buildIndex() {
    conceptIndex = new Map();
    CONCEPTS.forEach((g, ci) => g.forEach(w => conceptIndex.set(w, ci)));
  }
  const hash = s => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };
  const stem = w => w.length > 4 ? w.replace(/(ing|ed|es|s)$/, "") : w;

  function words(text, o) {
    const keepStop = o && o.stop === false;
    return (String(text).toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || []).filter(w => keepStop || !STOP.has(w));
  }

  function embed(text) {
    if (!conceptIndex) buildIndex();
    const dim = CONCEPTS.length + HASH_DIMS, v = new Float64Array(dim);
    const t = " " + String(text).toLowerCase().replace(/[^a-z0-9' -]+/g, " ") + " ";
    const used = new Set();
    // multi-word phrases first
    conceptIndex.forEach((ci, phrase) => {
      if (phrase.includes(" ") && t.includes(" " + phrase + " ")) { v[ci] += 1.2; phrase.split(" ").forEach(w => used.add(w)); }
    });
    words(text).forEach(w => {
      if (used.has(w)) return;
      const ci = conceptIndex.has(w) ? conceptIndex.get(w) : conceptIndex.get(stem(w));
      if (ci != null) v[ci] += 1;
      else v[CONCEPTS.length + (hash(stem(w)) % HASH_DIMS)] += 0.55; // unknown words: weaker, hashed meaning
    });
    let n = 0; for (let i = 0; i < dim; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < dim; i++) v[i] /= n;
    return v;
  }
  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na * nb) || 1);
  }

  function bm25(docs, query, o) {
    const k1 = (o && o.k1) || 1.5, b = (o && o.b != null) ? o.b : 0.75;
    const toks = docs.map(d => words(d).map(stem));
    const N = toks.length, avg = toks.reduce((s, t) => s + t.length, 0) / (N || 1);
    const df = new Map();
    toks.forEach(t => new Set(t).forEach(w => df.set(w, (df.get(w) || 0) + 1)));
    const q = words(query).map(stem);
    return toks.map(t => {
      const tf = new Map(); t.forEach(w => tf.set(w, (tf.get(w) || 0) + 1));
      let s = 0;
      q.forEach(w => {
        const f = tf.get(w) || 0; if (!f) return;
        const n = df.get(w) || 0, idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
        s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * t.length / (avg || 1)));
      });
      return s;
    });
  }
  function rrf(rankings, k) {
    k = k || 60; const sc = new Map();
    rankings.forEach(r => r.forEach((i, rank) => sc.set(i, (sc.get(i) || 0) + 1 / (k + rank + 1))));
    return [...sc.entries()].map(([i, score]) => ({ i, score })).sort((a, b) => b.score - a.score);
  }
  function search(docs, query, o) {
    o = Object.assign({ mode: "dense", k: docs.length }, o || {});
    const dense = () => { const q = embed(query); return docs.map((d, i) => ({ i, score: cosine(q, embed(d)) })).sort((a, b) => b.score - a.score); };
    const sparse = () => bm25(docs, query).map((score, i) => ({ i, score })).sort((a, b) => b.score - a.score);
    let out;
    if (o.mode === "bm25") out = sparse();
    else if (o.mode === "hybrid") out = rrf([dense().map(r => r.i), sparse().filter(r => r.score > 0).map(r => r.i)]);
    else out = dense();
    return out.slice(0, o.k);
  }
  function mmr(qVec, docVecs, k, lambda) {
    lambda = lambda == null ? 0.7 : lambda;
    const chosen = [], left = docVecs.map((_, i) => i);
    while (chosen.length < Math.min(k, docVecs.length)) {
      let best = -1, bestS = -Infinity;
      left.forEach(i => {
        const rel = cosine(qVec, docVecs[i]);
        const red = chosen.length ? Math.max(...chosen.map(j => cosine(docVecs[i], docVecs[j]))) : 0;
        const s = lambda * rel - (1 - lambda) * red;
        if (s > bestS) { bestS = s; best = i; }
      });
      chosen.push(best); left.splice(left.indexOf(best), 1);
    }
    return chosen;
  }

  function sentences(text) {
    return (String(text).match(/[^.!?\n]+[.!?]+["')\]]*\s*|[^.!?\n]+$/gm) || []).map(s => s.trim()).filter(Boolean);
  }
  const countTokens = text => Math.max(1, Math.round(String(text).length / 4));
  function toyTokens(text) {
    const out = [];
    (String(text).match(/\s*[A-Za-z]+|\s*\d|\s*[^\sA-Za-z\d]/g) || []).forEach(p => {
      const lead = p.match(/^\s*/)[0], w = p.slice(lead.length);
      if (w.length <= 6 || !/^[A-Za-z]+$/.test(w)) { out.push(p); return; }
      // long words split into pieces, like subword tokens
      let first = true;
      for (let i = 0; i < w.length; i += 4) { out.push((first ? lead : "") + w.slice(i, i + 4)); first = false; }
    });
    return out;
  }

  function chunk(text, o) {
    o = Object.assign({ method: "fixed", size: 60, overlap: 0 }, o || {});
    const maxC = o.size * 4, ovC = o.overlap * 4, out = [];
    const push = (s, e) => { const t = text.slice(s, e); if (t.trim()) out.push({ text: t, start: s, end: e }); };
    if (o.method === "fixed") {
      const step = Math.max(1, maxC - ovC);
      for (let s = 0; s < text.length; s += step) { push(s, Math.min(text.length, s + maxC)); if (s + maxC >= text.length) break; }
      return out;
    }
    // unit boundaries: sentences or paragraphs (with character offsets)
    const units = [];
    const re = o.method === "paragraph" ? /[^\n]+(?:\n(?!\n)[^\n]+)*/g : /[^.!?\n]+[.!?]*["')\]]*/g;
    let m; while ((m = re.exec(text))) if (m[0].trim()) units.push([m.index, m.index + m[0].length]);
    if (o.method === "paragraph") { units.forEach(([s, e]) => push(s, e)); return out; }
    if (o.method === "sentence") {
      // pack whole sentences up to the size; overlap = repeat the last sentence(s) worth ~overlap tokens
      let i = 0;
      while (i < units.length) {
        let j = i, s = units[i][0], e = units[i][1];
        while (j + 1 < units.length && units[j + 1][1] - s <= maxC) { j++; e = units[j][1]; }
        push(s, e);
        if (j + 1 >= units.length) break;
        let back = j + 1, used = 0;
        while (o.overlap && back - 1 > i && used + (units[back - 1][1] - units[back - 1][0]) <= ovC) { back--; used += units[back][1] - units[back][0]; }
        i = Math.max(i + 1, back);
      }
      return out;
    }
    // recursive: try paragraphs, then sentences, then fixed, keeping each chunk under the size
    const paras = chunk(text, { method: "paragraph" });
    paras.forEach(p => {
      if (p.end - p.start <= maxC) { out.push(p); return; }
      chunk(p.text, { method: "sentence", size: o.size, overlap: o.overlap }).forEach(c => {
        if (c.end - c.start <= maxC * 1.15) out.push({ text: c.text, start: p.start + c.start, end: p.start + c.end });
        else chunk(c.text, { method: "fixed", size: o.size, overlap: o.overlap }).forEach(f =>
          out.push({ text: f.text, start: p.start + c.start + f.start, end: p.start + c.start + f.end }));
      });
    });
    return out;
  }

  window.AI = {
    concepts: CONCEPTS,
    addConcept(group) { CONCEPTS.push(group.map(w => w.toLowerCase())); conceptIndex = null; },
    countTokens, toyTokens, words, sentences, embed, cosine, bm25, rrf, search, mmr, chunk,
  };
})();

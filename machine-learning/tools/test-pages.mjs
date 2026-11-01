// Headless test harness for the Machine Learning course (forked from the AI Engineer course).
// usage:
//   node ml-test.mjs all            every page in the manifest (+ index, review, glossary)
//   node ml-test.mjs phase4         every page of one phase
//   node ml-test.mjs '<json>'       [{page, clicks:[{id, wait, select}], read:[ids], shot, scroll}]
// Checks per page: 0 console/page/HTTP errors, quiz count (3 per lesson, 5 per recap), a 🤔 Why this? box and
// ≥ 2 playgrounds in lessons, no blank canvases. With "all"/"phaseN" it also clicks every button and moves every
// range input inside .playground elements (and reports errors that causes). Add CLICK=0 to skip that.
import puppeteer from "puppeteer-core";
import fs from "fs";
import os from "os";

const BASE = process.env.BASE || "http://127.0.0.1:8771/machine-learning/";
const ROOT = process.env.ML_ROOT || os.homedir() + "/machine-learning/";
const arg = process.argv[2] || "all";
let specs;
if (arg === "all" || /^phase\d$/.test(arg)) {
  const src = fs.readFileSync(ROOT + "assets/course.js", "utf8");
  const hrefs = [...src.matchAll(/href: "([^"]+)"/g)].map(m => m[1]);
  const pages = arg === "all" ? ["index.html", "review.html", "glossary.html", ...hrefs] : hrefs.filter(h => h.startsWith(arg + "/"));
  specs = pages.filter(p => fs.existsSync(ROOT + p)).map(page => ({ page, auto: process.env.CLICK !== "0" }));
  const missing = pages.filter(p => !fs.existsSync(ROOT + p));
  if (missing.length) console.log("⚠️ missing files:", missing.join(", "));
} else specs = JSON.parse(arg);

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true, timeout: 180000, protocolTimeout: 300000,
  args: ["--no-first-run", "--no-default-browser-check"],
});
let failures = 0;
for (const spec of specs) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1300, height: 900 });
  const errs = [];
  page.on("pageerror", e => errs.push("pageerror: " + e.message));
  page.on("console", m => { if (m.type() === "error" && !/favicon/.test(m.text())) errs.push("console: " + m.text()); });
  page.on("requestfailed", r => errs.push("requestfailed: " + r.url()));
  page.on("response", r => { if (r.status() >= 400) errs.push(`HTTP ${r.status()}: ${r.url()}`); });
  page.on("dialog", d => d.dismiss());
  try { await page.goto(BASE + spec.page + "?t=" + Date.now(), { waitUntil: "load", timeout: 60000 }); }
  catch (e) { errs.push("goto: " + e.message); }
  await new Promise(r => setTimeout(r, 300));
  for (const c of spec.clicks || []) {
    try {
      if (c.select) await page.select("#" + c.id, c.select);
      else if (c.value != null) await page.$eval("#" + c.id, (el, v) => { el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); }, String(c.value));
      else await page.$eval("#" + c.id, el => el.click());
    } catch (e) { errs.push(`click #${c.id} failed: ${e.message}`); }
    await new Promise(r => setTimeout(r, c.wait ?? 400));
  }
  const info = await page.evaluate(ids => {
    const q = document.querySelectorAll(".quiz").length;
    const why = document.querySelectorAll(".box.why").length;
    const pg = document.querySelectorAll(".playground").length;
    const blank = [...document.querySelectorAll("canvas")].filter(c => c.width === 0).length;
    const nav = document.querySelector(".lesson-nav")?.innerText.replace(/\s+/g, " ") ?? "-";
    const lesson = document.body.dataset.lesson || "";
    const reads = ids.map(id => `${id}=${(document.getElementById(id)?.innerText ?? "∅").replace(/\s+/g, " ").slice(0, 160)}`);
    const kb = Math.round(document.documentElement.outerHTML.length / 1024);
    return { q, why, pg, blank, nav, lesson, reads, kb };
  }, spec.read || []);
  let clicked = 0;
  if (spec.auto) {
    clicked = await page.evaluate(async () => {
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      let n = 0;
      for (const pgEl of document.querySelectorAll(".playground")) {
        for (const r of pgEl.querySelectorAll("input[type=range]")) {
          for (const v of [r.min, r.max, (+r.min + +r.max) / 2]) { r.value = v; r.dispatchEvent(new Event("input", { bubbles: true })); r.dispatchEvent(new Event("change", { bubbles: true })); n++; }
        }
        for (const s of pgEl.querySelectorAll("select")) { for (const o of s.options) { s.value = o.value; s.dispatchEvent(new Event("change", { bubbles: true })); s.dispatchEvent(new Event("input", { bubbles: true })); n++; } }
        for (const c of pgEl.querySelectorAll("input[type=checkbox]")) { c.click(); n++; }
        for (const b of pgEl.querySelectorAll("button")) { if (b.disabled) continue; b.click(); n++; await sleep(15); }
      }
      await sleep(400);
      return n;
    });
    await new Promise(r => setTimeout(r, 300));
  }
  const problems = [];
  const isRecap = /-r$/.test(info.lesson), isLesson = info.lesson && !isRecap;
  if (isLesson && info.q !== 3) problems.push(`expected 3 quizzes, found ${info.q}`);
  if (isRecap && info.q !== 5) problems.push(`expected 5 quizzes, found ${info.q}`);
  if (isLesson && info.why < 1) problems.push("no .box.why");
  if (isLesson && info.pg < 2) problems.push(`only ${info.pg} playground(s)`);
  if (info.blank) problems.push(`${info.blank} blank canvas(es)`);
  if (errs.length || problems.length) failures++;
  console.log(`\n▶ ${spec.page} (${info.kb} KB)\n  errors: ${errs.length ? errs.join(" || ") : "none"}${problems.length ? "\n  ❗ " + problems.join("; ") : ""}\n  quizzes=${info.q} why=${info.why} playgrounds=${info.pg}${spec.auto ? ` autoClicks=${clicked}` : ""}\n  nav: ${info.nav.slice(0, 200)}`);
  info.reads.forEach(r => console.log("  " + r));
  if (spec.shot) {
    if (spec.scroll) await page.$eval(spec.scroll, el => el.scrollIntoView({ block: "start" }));
    await new Promise(r => setTimeout(r, 300));
    const el = spec.scroll ? await page.$(spec.scroll) : null;
    await (el ? el.screenshot({ path: `/tmp/rltest/${spec.shot}.png` }) : page.screenshot({ path: `/tmp/rltest/${spec.shot}.png` }));
  }
  await page.close();
}
await browser.close();
console.log(`\n${failures ? "❌" : "✅"} ${specs.length - failures}/${specs.length} pages without errors or problems`);

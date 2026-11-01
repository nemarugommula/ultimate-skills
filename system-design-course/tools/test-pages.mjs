// Headless test harness for the System Design course (forked from the Math for AI course).
// usage: node test.mjs '<json>'  where json = [{page, clicks:[{id, wait}], read:[ids]}]
import puppeteer from "puppeteer-core";

const BASE = process.env.BASE || "http://127.0.0.1:8766/system-design-course/";
const specs = JSON.parse(process.argv[2]);
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
  await page.goto(BASE + spec.page + "?t=" + Date.now(), { waitUntil: "load" });
  await new Promise(r => setTimeout(r, 300));
  for (const c of spec.clicks || []) {
    try {
      if (c.select) await page.select("#" + c.id, c.select);
      else await page.$eval("#" + c.id, el => el.click());
    } catch (e) { errs.push(`click #${c.id} failed: ${e.message}`); }
    await new Promise(r => setTimeout(r, c.wait ?? 400));
  }
  const info = await page.evaluate(ids => {
    const q = document.querySelectorAll(".quiz").length;
    const blank = [...document.querySelectorAll("canvas")].filter(c => c.width === 0).length;
    const nav = document.querySelector(".lesson-nav")?.innerText.replace(/\s+/g, " ") ?? "-";
    const reads = ids.map(id => `${id}=${(document.getElementById(id)?.innerText ?? "∅").replace(/\s+/g, " ").slice(0, 160)}`);
    return { q, blank, nav, reads };
  }, spec.read || []);
  if (errs.length) failures++;
  console.log(`\n▶ ${spec.page}\n  errors: ${errs.length ? errs.join(" || ") : "none"}\n  quizzes=${info.q} blankCanvas=${info.blank}\n  nav: ${info.nav}`);
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
console.log(`\n${failures ? "❌" : "✅"} ${specs.length - failures}/${specs.length} pages without errors`);

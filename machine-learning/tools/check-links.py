"""Check every link in the Machine Learning course.

python3 tools/check-links.py            internal links (relative hrefs/srcs, #anchors) on every page, incl. sibling courses
python3 tools/check-links.py --external also curl every external http(s) link (UA Mozilla/5.0) and report non-200s
python3 tools/check-links.py --external phase4/2-random-forests.html   only that page
"""
import os, re, sys, subprocess, html
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.dirname(ROOT)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ext = "--external" in sys.argv
pages = args or sorted(os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) for f in fs if f.endswith(".html"))
bad, external = [], {}
for p in pages:
    src = open(os.path.join(ROOT, p), encoding="utf-8").read()
    ids = set(re.findall(r'\bid="([^"]+)"', src))
    for url in re.findall(r'(?:href|src)="([^"]+)"', src):
        url = html.unescape(url)
        if url.startswith(("http://", "https://")):
            external.setdefault(url, []).append(p); continue
        if url.startswith(("data:", "mailto:", "javascript:")) or "${" in url: continue
        if url.startswith("#"):
            if url[1:] and url[1:] not in ids: bad.append((p, url, "missing anchor"))
            continue
        path = url.split("#")[0].split("?")[0]
        target = os.path.normpath(os.path.join(ROOT, os.path.dirname(p), path))
        if not os.path.exists(target): bad.append((p, url, "missing file"))
print(f"{len(pages)} pages, {len(external)} distinct external links")
for b in bad: print("❌ internal", *b)
if ext:
    def code(u):
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L", "--max-time", "25", "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36", u], capture_output=True, text=True)
        return u, r.stdout.strip()
    with ThreadPoolExecutor(12) as ex:
        for u, c in ex.map(code, sorted(external)):
            if c != "200": print(f"⚠️ {c} {u}  ← {', '.join(sorted(set(external[u])))[:120]}")
print("✅ internal links OK" if not bad else f"❌ {len(bad)} broken internal links")

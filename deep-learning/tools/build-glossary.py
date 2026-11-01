"""Merge tools/glossary/phase*.json into glossary.html (Deep Learning course).

Each JSON file is a list of {"term", "kind", "meaning", "tools", "lessons"}. Duplicate terms (same name, case-insensitive)
are merged: the first meaning wins, tools and lessons are combined.   Run:  python3 tools/build-glossary.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
entries, seen = [], {}
for path in sorted(glob.glob(os.path.join(ROOT, "tools", "glossary", "phase*.json")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
    for e in json.load(open(path, encoding="utf-8")):
        k = re.sub(r"<[^>]+>", "", e["term"]).strip().lower()
        if k in seen:
            old = seen[k]
            old["lessons"] = " ".join(dict.fromkeys((old.get("lessons", "") + " " + e.get("lessons", "")).split()))
            if e.get("tools") and e["tools"] not in (old.get("tools") or ""):
                old["tools"] = ", ".join(x for x in [old.get("tools"), e["tools"]] if x)
            continue
        e = {"term": e["term"], "kind": e.get("kind", ""), "meaning": e["meaning"], "tools": e.get("tools", ""), "lessons": e.get("lessons", "")}
        seen[k] = e
        entries.append(e)
html_path = os.path.join(ROOT, "glossary.html")
html = open(html_path, encoding="utf-8").read()
data = json.dumps(entries, ensure_ascii=False, indent=0)
html = re.sub(r"/\*GLOSSARY-DATA\*/.*?/\*END-GLOSSARY-DATA\*/", lambda m: "/*GLOSSARY-DATA*/" + data + "/*END-GLOSSARY-DATA*/", html, flags=re.S)
open(html_path, "w", encoding="utf-8").write(html)
print("glossary.html: %d terms" % len(entries))

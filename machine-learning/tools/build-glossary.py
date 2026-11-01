"""Build glossary.html's data from tools/glossary/*.json (one file per phase, written by each phase's author).

python3 tools/build-glossary.py
Each JSON: {"tools": [[name, kind, category, what, when, alternatives, "4.2 4.3"]],
            "acronyms": [[acr, expansion, meaning, tools, where-used, "lessons"]],
            "terms": [[term, meaning, tools, where-used, "lessons"]]}
Duplicates (same name, case-insensitive) are merged: the longer text wins and the lesson lists are united.
"""
import glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = {"tools": {}, "acronyms": {}, "terms": {}}
LESSON_COL = {"tools": 6, "acronyms": 5, "terms": 4}
WIDTH = {"tools": 7, "acronyms": 6, "terms": 5}
for path in sorted(glob.glob(os.path.join(ROOT, "tools", "glossary", "*.json"))):
    src = json.load(open(path, encoding="utf-8"))
    for sec in data:
        for row in src.get(sec, []):
            row = [str(x) if x is not None else "" for x in row]
            if len(row) != WIDTH[sec]:
                raise SystemExit(f"{os.path.basename(path)} {sec}: row needs {WIDTH[sec]} fields: {row[:2]}")
            key = re.sub(r"<[^>]+>", "", row[0]).strip().lower()
            if key in data[sec]:
                old = data[sec][key]
                merged = [a if len(a) >= len(b) else b for a, b in zip(old, row)]
                lc = LESSON_COL[sec]
                ls = []
                for n in (old[lc] + " " + row[lc]).split():
                    if n not in ls: ls.append(n)
                merged[lc] = " ".join(ls)
                data[sec][key] = merged
            else:
                data[sec][key] = row
js = "\n".join(f"  const {name} = {json.dumps(list(data[sec].values()), ensure_ascii=False, indent=0)};"
               for name, sec in (("TOOLS", "tools"), ("ACRONYMS", "acronyms"), ("TERMS", "terms")))
p = os.path.join(ROOT, "glossary.html")
html = open(p, encoding="utf-8").read()
a, b = html.index("/*GLOSSARY-DATA*/") + len("/*GLOSSARY-DATA*/"), html.index("  /*END-GLOSSARY-DATA*/")
html = html[:a] + "\n" + js + "\n" + html[b:]
open(p, "w", encoding="utf-8").write(html)
print("glossary: %d tools, %d acronyms, %d terms" % tuple(len(data[s]) for s in ("tools", "acronyms", "terms")))

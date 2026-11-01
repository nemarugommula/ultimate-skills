"""Build glossary.html from tools/glossary/phase*.json (one file per phase, written by the lesson writers).

Run after editing any glossary JSON:   python3 tools/build-glossary.py
Each entry: {"term", "kind": "acronym"|"term"|"tool", "expansion", "meaning", "tools", "used", "lessons": "3.3 3.4"}.
Duplicate terms across phases are merged: the first meaning wins, lesson lists are combined.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    merged, order = {}, []
    for path in sorted(glob.glob(os.path.join(ROOT, "tools", "glossary", "phase*.json")), key=lambda p: int(re.search(r"phase(\d+)", p).group(1))):
        for e in json.load(open(path, encoding="utf-8")):
            e = {k: (e.get(k) or "") for k in ("term", "kind", "expansion", "meaning", "tools", "used", "lessons")}
            if e["kind"] not in ("acronym", "term", "tool"):
                e["kind"] = "term"
            k = re.sub(r"\W+", "", e["term"].lower())
            if k in merged:
                old = merged[k]
                ls = old["lessons"].split() + [n for n in e["lessons"].split() if n not in old["lessons"].split()]
                old["lessons"] = " ".join(ls)
                for f in ("tools", "used", "expansion"):
                    if not old[f] and e[f]:
                        old[f] = e[f]
            else:
                merged[k] = e
                order.append(k)
    entries = [merged[k] for k in order]
    tpl = open(os.path.join(ROOT, "tools", "glossary-template.html"), encoding="utf-8").read()
    data = json.dumps(entries, ensure_ascii=False, indent=0).replace("</", "<\\/")
    open(os.path.join(ROOT, "glossary.html"), "w", encoding="utf-8").write(tpl.replace("/*DATA*/[]", data))
    kinds = {k: sum(e["kind"] == k for e in entries) for k in ("tool", "acronym", "term")}
    print("wrote glossary.html: %d entries (%s)" % (len(entries), ", ".join("%d %ss" % (v, k) for k, v in kinds.items())))


if __name__ == "__main__":
    main()

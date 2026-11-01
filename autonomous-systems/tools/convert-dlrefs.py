"""Turn Deep Learning course references into links where the target file exists.

Writers mark references as <span class="dlref" data-href="phase4/8-3d-vision.html">the Deep Learning course, Phase 4 (…)</span>.
This rewrites each marker whose ~/deep-learning/<data-href> exists into <a class="dlref" href="../../deep-learning/…">…</a>
(and leaves the others as plain words). Safe to rerun.
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL = os.path.join(os.path.dirname(ROOT), "deep-learning")
PAT = re.compile(r'<span class="dlref" data-href="([^"]+)">(.*?)</span>', re.S)


def main():
    linked = kept = 0
    for p in glob.glob(os.path.join(ROOT, "phase*", "*.html")):
        html = open(p, encoding="utf-8").read()

        def fix(m):
            nonlocal linked, kept
            if os.path.exists(os.path.join(DL, m.group(1))):
                linked += 1
                return '<a class="dlref" href="../../deep-learning/%s">%s</a>' % (m.group(1), m.group(2))
            kept += 1
            return m.group(0)
        new = PAT.sub(fix, html)
        if new != html:
            open(p, "w", encoding="utf-8").write(new)
    print("linked %d Deep Learning references; %d left as words (file not there yet)" % (linked, kept))


if __name__ == "__main__":
    main()

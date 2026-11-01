"""Check every page: internal links/scripts resolve to files, manifest lessons exist, quiz counts are right.

python3 tools/check-links.py            # internal checks
python3 tools/check-links.py --external # also print every external URL (one per line) to tools/external-links.txt
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.dirname(ROOT)


def manifest():
    src = open(os.path.join(ROOT, "assets", "course.js"), encoding="utf-8").read()
    m = src[src.index("const COURSE = ["):src.index("\n];\n")]
    return re.findall(r'\{ id: "([^"]+)", title: "(?:[^"\\]|\\.)*", href: "([^"]+)"', m)


def main():
    bad, ext = [], set()
    for lid, href in manifest():
        p = os.path.join(ROOT, href)
        if not os.path.exists(p):
            bad.append("missing lesson %s %s" % (lid, href))
            continue
        html = open(p, encoding="utf-8").read()
        n = len(re.findall(r'<div class="quiz">', html))
        want = 5 if lid.endswith("-r") else 3
        if n != want:
            bad.append("%s: %d quizzes (want %d)" % (href, n, want))
        if 'data-lesson="%s"' % lid not in html:
            bad.append("%s: data-lesson is not %s" % (href, lid))
    pages = glob.glob(os.path.join(ROOT, "*.html")) + glob.glob(os.path.join(ROOT, "phase*", "*.html"))
    for p in pages:
        html = open(p, encoding="utf-8").read()
        for attr, url in re.findall(r'(?<![-\w])(href|src)="([^"]+)"', html):
            if url.startswith(("http://", "https://")):
                ext.add(url.replace("&amp;", "&"))
                continue
            if url.startswith(("#", "data:", "mailto:", "javascript:")) or "${" in url:
                continue
            path = url.split("#")[0].split("?")[0]
            if not path:
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(p), path))
            if not os.path.exists(target):
                bad.append("%s: broken %s=%s" % (os.path.relpath(p, ROOT), attr, url))
            elif not target.startswith(HOME):
                bad.append("%s: link leaves the home folder %s" % (os.path.relpath(p, ROOT), url))
    print("\n".join(bad) if bad else "all internal links, lessons and quiz counts OK")
    if "--external" in sys.argv:
        open(os.path.join(ROOT, "tools", "external-links.txt"), "w").write("\n".join(sorted(ext)) + "\n")
        print("%d external URLs → tools/external-links.txt" % len(ext))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

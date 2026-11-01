#!/bin/bash
# curl every external URL found by tools/check-links.py --external; prints non-200 results.
cd "$(dirname "$0")/.." && python3 tools/check-links.py --external >/dev/null
check() {
  u="$1"
  case "$u" in
    *arxiv.org/abs/*) id="${u##*/abs/}"; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 "https://export.arxiv.org/api/query?id_list=${id%v*}");;
    *youtube.com/watch*|*youtu.be/*) code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 "https://www.youtube.com/oembed?format=json&url=$u");;
    *) code=$(curl -s -o /dev/null -w '%{http_code}' -L -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36' --max-time 25 "$u");;
  esac
  [ "$code" = "200" ] || echo "$code $u"
}
export -f check
xargs -P 12 -I{} bash -c 'check "$@"' _ {} < tools/external-links.txt

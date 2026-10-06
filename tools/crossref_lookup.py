"""按标题查 Crossref,打印可直接粘进 refs.bib 的字段。

为什么要有这个脚本:本仓库曾凭记忆编造过三条参考文献(见 memory
`citations-must-be-crossref-verified`),而搜索摘要也会失真。规矩是**每一条新增文献
的题目/作者/卷期页/年/DOI 必须来自 Crossref REST API 的原始响应**,不是记忆、不是
搜索结果页的转述。

用法: python tools/crossref_lookup.py "标题关键字" ["标题关键字" ...]
      python tools/crossref_lookup.py --doi 10.1016/j.ymssp.2022.108981
"""
import json
import sys
import urllib.parse
import urllib.request

try:                      # Windows 控制台默认 GBK,作者名里的重音字符会打崩
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

API = "https://api.crossref.org/works"
UA = "ClawsGO-Agent/1.0 (mailto:noreply@clawsgo.cn)"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _show(it):
    au = it.get("author", [])
    names = "; ".join(f"{a.get('family', '?')}, {a.get('given', '?')}"
                      for a in au[:8]) or "(无作者字段)"
    if len(au) > 8:
        names += "; et al."
    dp = it.get("issued", {}).get("date-parts", [[None]])[0]
    year = dp[0] if dp else None
    ctype = it.get("type", "")
    venue = (it.get("container-title") or [""])[0]
    print(f"  title   : {(it.get('title') or ['?'])[0]}")
    print(f"  authors : {names}")
    print(f"  venue   : {venue}   [{ctype}]")
    print(f"  vol/iss : {it.get('volume', '-')} / {it.get('issue', '-')}"
          f"   pages {it.get('page', '-')}   year {year}")
    print(f"  doi     : {it.get('DOI')}")
    print(f"  score   : {it.get('score'):.1f}" if it.get("score") else "")
    print()


def by_title(q):
    print("=" * 78)
    print(f"查询:{q}")
    url = f"{API}?query.bibliographic={urllib.parse.quote(q)}&rows=3"
    msg = _get(url)["message"]
    for it in msg["items"]:
        _show(it)


def by_doi(doi):
    print("=" * 78)
    print(f"DOI:{doi}")
    try:
        _show(_get(f"{API}/{urllib.parse.quote(doi)}")["message"])
    except urllib.error.HTTPError as e:
        print(f"  **{e.code} —— 这个 DOI 在 Crossref 里不存在,不能写进 refs.bib**\n")


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "--doi":
        for d in a[1:]:
            by_doi(d)
    else:
        for q in a:
            by_title(q)


if __name__ == "__main__":
    main()

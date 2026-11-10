import urllib.request, time, os

ids = """1706.03762 1810.04805 1907.11692 1910.10683 2005.14165 2005.11401
2004.04906 2002.08909 2004.12832 1908.10084 2112.09118 2212.10496 2310.11511
2312.10997 2307.03172 2104.08663 2210.03629 2201.11903 2203.02155 2106.09685
2305.14314 2302.13971 2307.09288 2310.06825 2001.08361 2203.15556 2205.14135
2010.11929 2103.00020 1512.03385 1412.6980 1301.3781 2302.04761 2303.08774
2312.00752 2404.16130""".split()

os.makedirs("data/papers", exist_ok=True)
for i in ids:
    path = f"data/papers/{i}.pdf"
    if os.path.exists(path):
        continue
    req = urllib.request.Request(f"https://arxiv.org/pdf/{i}", headers={"User-Agent": "raglab-student"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())
    print("saved", i)
    time.sleep(3)

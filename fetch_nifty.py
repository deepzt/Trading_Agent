import urllib.request, csv, io, sys

url = sys.argv[1] if len(sys.argv) > 1 else "https://nsearchives.nseindia.com/content/indices/ind_niftyfullmidcap100list.csv"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
data = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig")
rows = list(csv.DictReader(io.StringIO(data)))
print(f"count={len(rows)}")
for r in rows:
    print(r.get("Symbol") or r.get("symbol") or list(r.values())[2])

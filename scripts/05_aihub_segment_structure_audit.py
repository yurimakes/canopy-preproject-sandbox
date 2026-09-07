from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv
import io
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"

name_re = re.compile(
    r"^TMC-GPS-([^-]+)-([^-]+)-([^-]+)-Dataset\.csv$",
    re.I
)

def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")

segments = []
missing = Counter()
label_values = Counter()
detail_values = Counter()

# GPS
for pkg in sorted(PACKAGES.glob("*_GPS_*.zip")):
    cls = pkg.stem.split("_")[-1]

    with ZipFile(pkg) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue

            base = PurePosixPath(member).name
            m = name_re.match(base)
            if not m:
                continue

            uid, tid, sid = m.groups()

            reader = csv.DictReader(io.StringIO(decode(z.read(member))))
            rows = list(reader)

            ts = []

            for r in rows:
                for col in ["accuracy", "latitude", "longitude", "altitude"]:
                    v = str(r.get(col, "")).strip()
                    if v == "":
                        missing[col] += 1
                    else:
                        try:
                            float(v)
                        except:
                            missing[col + "_nonnumeric"] += 1

                try:
                    ts.append(int(float(r["timestamp"])))
                except:
                    pass

            if ts:
                segments.append({
                    "uid": uid,
                    "tid": tid,
                    "sid": sid,
                    "class": cls,
                    "start": min(ts),
                    "end": max(ts),
                    "points": len(rows)
                })

# LABEL
for pkg in sorted(PACKAGES.glob("*_LABEL_*.zip")):
    with ZipFile(pkg) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue

            reader = csv.DictReader(io.StringIO(decode(z.read(member))))

            for r in reader:
                label_values[str(r.get("label", "")).strip()] += 1
                detail_values[str(r.get("detail_label", "")).strip()] += 1

# UID/TID 단위 segment 묶기
groups = defaultdict(list)

for s in segments:
    groups[(s["uid"], s["tid"])].append(s)

sid_counts = Counter()
gap_counts = Counter()
class_change_tids = 0

all_gaps = []

for key, segs in groups.items():
    segs.sort(key=lambda x: x["start"])

    sid_counts[len(segs)] += 1

    if len({s["class"] for s in segs}) > 1:
        class_change_tids += 1

    for a, b in zip(segs, segs[1:]):
        gap = b["start"] - a["end"]
        all_gaps.append(gap)
        gap_counts[gap] += 1

def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[round((len(v)-1)*p)]

print("=== BASIC ===")
print("SEGMENTS =", len(segments))
print("UNIQUE UID/TID =", len(groups))

print()
print("=== SEGMENTS PER TID ===")
for k in sorted(sid_counts)[:20]:
    print(k, "segments :", sid_counts[k], "TIDs")

print()
print("TID WITH >=2 SEGMENTS =", sum(v for k,v in sid_counts.items() if k >= 2))
print("TID WITH >=3 SEGMENTS =", sum(v for k,v in sid_counts.items() if k >= 3))
print("CLASS-CHANGE TIDs     =", class_change_tids)

print()
print("=== BETWEEN-SEGMENT GAP (ms) ===")
if all_gaps:
    print("min    =", min(all_gaps))
    print("p25    =", pct(all_gaps, .25))
    print("median =", pct(all_gaps, .50))
    print("p75    =", pct(all_gaps, .75))
    print("p95    =", pct(all_gaps, .95))
    print("max    =", max(all_gaps))

    print()
    print("Most common gaps:")
    for gap, n in gap_counts.most_common(10):
        print(gap, "ms :", n)

print()
print("=== MISSING GPS VALUES ===")
for k,v in sorted(missing.items()):
    print(k, "=", v)

print()
print("=== LABEL VALUES ===")
print(label_values)

print()
print("=== DETAIL LABEL VALUES ===")
print(detail_values)


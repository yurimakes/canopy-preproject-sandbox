from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv
import io
import re

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"

NAME_RE = re.compile(
    r"^TMC-GPS-([^-]+)-([^-]+)-([^-]+)-Dataset\.csv$",
    re.I
)

FIELDS = ["accuracy", "latitude", "longitude", "altitude"]


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def unavailable(value):
    value = str(value if value is not None else "").strip()

    if value == "":
        return True

    try:
        float(value)
        return False
    except ValueError:
        return True


segments = []

row_stats = defaultdict(Counter)
segment_stats = defaultdict(Counter)

for pkg in sorted(PACKAGES.glob("*_GPS_*.zip")):

    if pkg.name.startswith("TS_"):
        split = "training"
    elif pkg.name.startswith("VS_"):
        split = "validation"
    else:
        continue

    cls = pkg.stem.split("_")[-1].upper()

    with ZipFile(pkg) as z:
        for member in z.namelist():

            if not member.lower().endswith(".csv"):
                continue

            base = PurePosixPath(member).name
            m = NAME_RE.match(base)

            if not m:
                continue

            uid, tid, sid = m.groups()

            reader = csv.DictReader(
                io.StringIO(decode(z.read(member)))
            )
            rows = list(reader)

            timestamps = []

            missing_rows = 0
            all4_missing_rows = 0
            coord_missing_rows = 0

            for row in rows:

                row_stats[cls]["rows"] += 1

                unavailable_fields = {
                    field: unavailable(row.get(field))
                    for field in FIELDS
                }

                for field, is_missing in unavailable_fields.items():
                    if is_missing:
                        row_stats[cls][f"missing_{field}"] += 1

                if any(unavailable_fields.values()):
                    missing_rows += 1

                if all(unavailable_fields.values()):
                    all4_missing_rows += 1

                if (
                    unavailable_fields["latitude"]
                    or unavailable_fields["longitude"]
                ):
                    coord_missing_rows += 1

                try:
                    timestamps.append(
                        int(float(row["timestamp"]))
                    )
                except Exception:
                    pass

            segment_stats[cls]["segments"] += 1

            if missing_rows == 0:
                segment_stats[cls]["complete_segments"] += 1
            elif missing_rows == len(rows):
                segment_stats[cls]["fully_missing_segments"] += 1
            else:
                segment_stats[cls]["partial_missing_segments"] += 1

            segment_stats[cls]["missing_rows"] += missing_rows
            segment_stats[cls]["all4_missing_rows"] += all4_missing_rows
            segment_stats[cls]["coord_missing_rows"] += coord_missing_rows

            if timestamps:
                segments.append({
                    "uid": uid,
                    "tid": tid,
                    "sid": sid,
                    "class": cls,
                    "split": split,
                    "start": min(timestamps),
                    "end": max(timestamps),
                })


print("=== MISSING GPS BY CLASS ===")

for cls in ["WALK", "BIKE", "CAR", "BUS", "SUBWAY", "ETC"]:

    rs = row_stats[cls]
    ss = segment_stats[cls]

    total_rows = rs["rows"]
    coord_missing = ss["coord_missing_rows"]

    pct = (
        coord_missing / total_rows * 100
        if total_rows else 0
    )

    print()
    print(cls)
    print("rows                    =", total_rows)
    print("missing accuracy        =", rs["missing_accuracy"])
    print("missing latitude        =", rs["missing_latitude"])
    print("missing longitude       =", rs["missing_longitude"])
    print("missing altitude        =", rs["missing_altitude"])
    print("all-4 missing rows      =", ss["all4_missing_rows"])
    print("coordinate missing %    =", round(pct, 4))
    print("segments                =", ss["segments"])
    print("complete segments       =", ss["complete_segments"])
    print("partial missing segments=", ss["partial_missing_segments"])
    print("fully missing segments  =", ss["fully_missing_segments"])


train_tids = {
    (s["uid"], s["tid"])
    for s in segments
    if s["split"] == "training"
}

val_tids = {
    (s["uid"], s["tid"])
    for s in segments
    if s["split"] == "validation"
}

print()
print("=== OFFICIAL SPLIT TID CHECK ===")
print("TRAIN UID/TID =", len(train_tids))
print("VAL UID/TID   =", len(val_tids))
print("UID/TID OVERLAP =", len(train_tids & val_tids))


groups = defaultdict(list)

for s in segments:
    groups[(s["uid"], s["tid"])].append(s)


gap_counter = Counter()

negative_pairs = 0
minus59000_pairs = 0
minus59000_same_start = 0
minus59000_cross_split = 0

duplicate_start_groups = 0
duplicate_start_extra_segments = 0


for segs in groups.values():

    start_counts = Counter(
        s["start"] for s in segs
    )

    for count in start_counts.values():
        if count > 1:
            duplicate_start_groups += 1
            duplicate_start_extra_segments += count - 1

    segs = sorted(
        segs,
        key=lambda x: (
            x["start"],
            x["end"],
            x["sid"]
        )
    )

    for a, b in zip(segs, segs[1:]):

        gap = b["start"] - a["end"]
        gap_counter[gap] += 1

        if gap < 0:
            negative_pairs += 1

        if gap == -59000:

            minus59000_pairs += 1

            if a["start"] == b["start"]:
                minus59000_same_start += 1

            if a["split"] != b["split"]:
                minus59000_cross_split += 1


print()
print("=== OVERLAPPING SEGMENT STRUCTURE ===")
print("negative-gap pairs            =", negative_pairs)
print("-59000 ms pairs               =", minus59000_pairs)
print("-59000 with identical start   =", minus59000_same_start)
print("-59000 crossing official split=", minus59000_cross_split)
print("duplicate-start groups        =", duplicate_start_groups)
print("duplicate-start extra segments=", duplicate_start_extra_segments)

print()
print("=== MOST COMMON GAPS ===")

for gap, count in gap_counter.most_common(15):
    print(gap, "ms =", count)


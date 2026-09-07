from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv
import hashlib
import io
import re

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
SPLIT_FILE = ROOT / "splits" / "uid_split_v1.csv"

NAME_RE = re.compile(
    r"^TMC-GPS-([^-]+)-([^-]+)-([^-]+)-Dataset\.csv$",
    re.I
)

FIELDS = [
    "timestamp",
    "accuracy",
    "latitude",
    "longitude",
    "altitude",
]


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------
# CANOPY UID-disjoint split
# ---------------------------------------------------------

uid_split = {}

with SPLIT_FILE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    field_map = {
        name.lower(): name
        for name in reader.fieldnames
    }

    uid_col = field_map["uid"]
    split_col = field_map["split"]

    for row in reader:
        uid_split[row[uid_col]] = row[split_col]


# ---------------------------------------------------------
# Read GPS segment metadata + content hash
# ---------------------------------------------------------

segments = []

for pkg in sorted(PACKAGES.glob("*_GPS_*.zip")):

    if pkg.name.startswith("TS_"):
        source_split = "official_training"
    elif pkg.name.startswith("VS_"):
        source_split = "official_validation"
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

            if uid not in uid_split:
                raise RuntimeError(
                    f"UID missing from split manifest: {uid}"
                )

            rows = list(
                csv.DictReader(
                    io.StringIO(decode(z.read(member)))
                )
            )

            timestamps = []
            normalized_rows = []
            missing_coords = 0

            for row in rows:

                values = [
                    str(row.get(field, "")).strip()
                    for field in FIELDS
                ]

                normalized_rows.append("|".join(values))

                try:
                    timestamps.append(
                        int(float(row["timestamp"]))
                    )
                except Exception:
                    raise RuntimeError(
                        f"Bad timestamp: {base}"
                    )

                lat = str(row.get("latitude", "")).strip()
                lon = str(row.get("longitude", "")).strip()

                if lat == "" or lon == "":
                    missing_coords += 1

            content_hash = hashlib.sha256(
                "\n".join(normalized_rows).encode("utf-8")
            ).hexdigest()

            segments.append({
                "uid": uid,
                "tid": tid,
                "sid": sid,
                "class": cls,
                "source_split": source_split,
                "our_split": uid_split[uid],
                "start": min(timestamps),
                "end": max(timestamps),
                "points": len(rows),
                "missing_coords": missing_coords,
                "hash": content_hash,
            })


print("=== BASIC ===")
print("RAW SEGMENTS =", len(segments))


# ---------------------------------------------------------
# Duplicate-start audit
# ---------------------------------------------------------

slots = defaultdict(list)

for s in segments:
    slots[
        (s["uid"], s["tid"], s["start"])
    ].append(s)


duplicate_groups = [
    group
    for group in slots.values()
    if len(group) > 1
]

size_distribution = Counter(
    len(group)
    for group in duplicate_groups
)

exact_content_groups = 0
different_content_groups = 0
cross_official_groups = 0
missing_count_disagreement = 0

for group in duplicate_groups:

    hashes = {
        x["hash"]
        for x in group
    }

    if len(hashes) == 1:
        exact_content_groups += 1
    else:
        different_content_groups += 1

    if len({
        x["source_split"]
        for x in group
    }) > 1:
        cross_official_groups += 1

    if len({
        x["missing_coords"]
        for x in group
    }) > 1:
        missing_count_disagreement += 1


print()
print("=== DUPLICATE START AUDIT ===")
print("duplicate-start groups       =", len(duplicate_groups))
print("group-size distribution      =", dict(sorted(size_distribution.items())))
print("exact-content groups         =", exact_content_groups)
print("different-content groups     =", different_content_groups)
print("cross official split groups  =", cross_official_groups)
print("missing-count disagreement   =", missing_count_disagreement)


# Do not collapse conflicting observations.
if different_content_groups > 0:
    print()
    print("=== SEQUENCE AUDIT STOPPED ===")
    print(
        "Same-start groups with different GPS content exist."
    )
    print(
        "Duplicate handling policy must be audited before sequence construction."
    )
    raise SystemExit(0)


# ---------------------------------------------------------
# Exact duplicate time slots:
# collapse only for structural analysis.
# Raw files are NOT modified.
# ---------------------------------------------------------

collapsed = []

for (uid, tid, start), group in slots.items():

    first = group[0]

    collapsed.append({
        "uid": uid,
        "tid": tid,
        "start": start,
        "end": first["end"],
        "class": first["class"],
        "our_split": first["our_split"],
        "copies": len(group),
    })


tid_groups = defaultdict(list)

for segment in collapsed:
    tid_groups[
        (segment["uid"], segment["tid"])
    ].append(segment)


# ---------------------------------------------------------
# Contiguous 60-sample slot runs
# ---------------------------------------------------------

negative_after_collapse = Counter()
run_lengths = []
tid_max_run = {}

for key, segs in tid_groups.items():

    segs.sort(
        key=lambda x: (
            x["start"],
            x["end"]
        )
    )

    current_run = 1
    runs = []

    for a, b in zip(segs, segs[1:]):

        gap = b["start"] - a["end"]

        if gap == 1000:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1

            if gap < 0:
                negative_after_collapse[gap] += 1

    runs.append(current_run)

    run_lengths.extend(runs)
    tid_max_run[key] = max(runs)


print()
print("=== AFTER EXACT-DUPLICATE SLOT COLLAPSE ===")
print("raw segments       =", len(segments))
print("unique time slots  =", len(collapsed))
print(
    "negative-gap pairs =",
    sum(negative_after_collapse.values())
)

print("negative-gap distribution:")
for gap, count in negative_after_collapse.most_common():
    print(gap, "ms =", count)


print()
print("=== CONTIGUOUS RUN STRUCTURE ===")
print("total runs              =", len(run_lengths))
print("maximum consecutive slots=", max(run_lengths))
print(
    "runs >= 2 slots         =",
    sum(x >= 2 for x in run_lengths)
)
print(
    "runs >= 4 slots         =",
    sum(x >= 4 for x in run_lengths)
)


# Each exact contiguous slot contains 60 samples.
#
# 2 slots = 120 samples
# first-to-last timestamp span = 119 seconds.
#
# 4 slots = 240 samples,
# therefore structurally enough observations for
# a 200-sample window.
#
# This is structural capacity only.
# It does NOT imply that all samples contain valid GPS coordinates.

eligible_120 = {
    key
    for key, run in tid_max_run.items()
    if run >= 2
}

eligible_200 = {
    key
    for key, run in tid_max_run.items()
    if run >= 4
}


print()
print("=== STRUCTURAL WINDOW CAPACITY ===")
print("TOTAL UID/TID                  =", len(tid_groups))
print("TID with >=120 sample capacity =", len(eligible_120))
print("TID with >=200 sample capacity =", len(eligible_200))


split_names = sorted(
    set(uid_split.values())
)

for split in split_names:

    tids = {
        key
        for key, segs in tid_groups.items()
        if segs[0]["our_split"] == split
    }

    print()
    print("===", split.upper(), "===")
    print("UID/TID                  =", len(tids))
    print(
        ">=120 sample capacity    =",
        len(tids & eligible_120)
    )
    print(
        ">=200 sample capacity    =",
        len(tids & eligible_200)
    )


print()
print("NOTE:")
print(
    "Capacity above is timestamp/sample-count feasibility only."
)
print(
    "Missing-coordinate validity and final preprocessing are not yet applied."
)

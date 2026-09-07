from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv
import hashlib
import io
import math
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

WINDOWS = [120, 200]
THRESHOLDS = [1.00, 0.95, 0.90, 0.80]


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def finite_number(value):
    value = str(value if value is not None else "").strip()

    if value == "":
        return False

    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


# ---------------------------------------------------------
# UID-disjoint split
# ---------------------------------------------------------

uid_split = {}

with SPLIT_FILE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    fields = {
        x.lower(): x
        for x in reader.fieldnames
    }

    for row in reader:
        uid_split[row[fields["uid"]]] = row[fields["split"]]


# ---------------------------------------------------------
# Index ZIP members by UID/TID without loading all GPS rows
# ---------------------------------------------------------

package_paths = sorted(
    PACKAGES.glob("*_GPS_*.zip")
)

zip_handles = {
    path: ZipFile(path)
    for path in package_paths
}

tid_refs = defaultdict(list)

for path, z in zip_handles.items():

    if path.name.startswith("TS_"):
        source_split = "official_training"
    elif path.name.startswith("VS_"):
        source_split = "official_validation"
    else:
        continue

    cls = path.stem.split("_")[-1].upper()

    for member in z.namelist():

        if not member.lower().endswith(".csv"):
            continue

        base = PurePosixPath(member).name
        m = NAME_RE.match(base)

        if not m:
            continue

        uid, tid, sid = m.groups()

        tid_refs[(uid, tid)].append({
            "path": path,
            "member": member,
            "sid": sid,
            "class": cls,
            "source_split": source_split,
        })


# ---------------------------------------------------------
# Aggregates
# ---------------------------------------------------------

negative_gap_counter = Counter()

negative_pairs = 0
negative_cross_official = 0
negative_shared_exact = 0
negative_shared_conflict = 0
negative_without_shared_timestamp = 0

point_overlap_observations = 0
point_overlap_exact = 0
point_overlap_conflict = 0

conflict_tids = 0
class_change_tids = 0

window_counts = {
    L: {
        threshold: 0
        for threshold in THRESHOLDS
    }
    for L in WINDOWS
}

structural_window_counts = Counter()

tid_eligible = {
    L: {
        threshold: set()
        for threshold in THRESHOLDS
    }
    for L in WINDOWS
}

structural_tid_eligible = {
    L: set()
    for L in WINDOWS
}

split_window_counts = defaultdict(Counter)
split_tid_eligible = defaultdict(set)

class_window_counts = defaultdict(Counter)
class_tid_eligible = defaultdict(set)


# ---------------------------------------------------------
# Process one UID/TID at a time
# ---------------------------------------------------------

for key, refs in tid_refs.items():

    uid, tid = key

    if uid not in uid_split:
        raise RuntimeError(
            "UID missing from split manifest"
        )

    split = uid_split[uid]

    segment_records = []

    classes = {
        ref["class"]
        for ref in refs
    }

    if len(classes) != 1:
        class_change_tids += 1

    folder_class = sorted(classes)[0]

    # ---------------------------------------------
    # Read all segments belonging to this TID
    # ---------------------------------------------

    for ref in refs:

        z = zip_handles[ref["path"]]

        rows = list(
            csv.DictReader(
                io.StringIO(
                    decode(z.read(ref["member"]))
                )
            )
        )

        points = {}
        normalized_rows = []

        for row in rows:

            try:
                ts = int(float(row["timestamp"]))
            except Exception:
                raise RuntimeError(
                    "Invalid timestamp encountered"
                )

            values = tuple(
                str(row.get(field, "")).strip()
                for field in FIELDS[1:]
            )

            lat = row.get("latitude", "")
            lon = row.get("longitude", "")

            coord_valid = (
                finite_number(lat)
                and finite_number(lon)
            )

            points[ts] = {
                "values": values,
                "coord_valid": coord_valid,
            }

            normalized_rows.append(
                str(ts)
                + "|"
                + "|".join(values)
            )

        content_hash = hashlib.sha256(
            "\n".join(normalized_rows).encode("utf-8")
        ).hexdigest()

        segment_records.append({
            "sid": ref["sid"],
            "class": ref["class"],
            "source_split": ref["source_split"],
            "start": min(points),
            "end": max(points),
            "points": points,
            "hash": content_hash,
        })


    # ---------------------------------------------
    # Collapse exact same-start segment duplicates
    # only in this analytical representation
    # ---------------------------------------------

    start_groups = defaultdict(list)

    for segment in segment_records:
        start_groups[segment["start"]].append(segment)

    collapsed_segments = []

    for group in start_groups.values():

        if len(group) == 1:
            collapsed_segments.append(group[0])
            continue

        hashes = {
            x["hash"]
            for x in group
        }

        if len(hashes) == 1:
            collapsed_segments.append(group[0])
        else:
            # 07 audit currently found zero such groups.
            # Keep all observations if this changes.
            collapsed_segments.extend(group)


    collapsed_segments.sort(
        key=lambda x: (
            x["start"],
            x["end"],
            x["sid"]
        )
    )


    # ---------------------------------------------
    # Audit remaining negative-overlap segment pairs
    # ---------------------------------------------

    for a, b in zip(
        collapsed_segments,
        collapsed_segments[1:]
    ):

        gap = b["start"] - a["end"]

        if gap >= 0:
            continue

        negative_pairs += 1
        negative_gap_counter[gap] += 1

        if a["source_split"] != b["source_split"]:
            negative_cross_official += 1

        shared = (
            set(a["points"])
            & set(b["points"])
        )

        if not shared:
            negative_without_shared_timestamp += 1
            continue

        conflicts = 0

        for ts in shared:

            if (
                a["points"][ts]["values"]
                != b["points"][ts]["values"]
            ):
                conflicts += 1

        if conflicts == 0:
            negative_shared_exact += 1
        else:
            negative_shared_conflict += 1


    # ---------------------------------------------
    # Point-level merge by timestamp
    # ---------------------------------------------

    merged = {}
    conflict_timestamps = set()

    for segment in collapsed_segments:

        for ts, point in segment["points"].items():

            if ts not in merged:
                merged[ts] = point
                continue

            point_overlap_observations += 1

            if (
                merged[ts]["values"]
                == point["values"]
            ):
                point_overlap_exact += 1
            else:
                point_overlap_conflict += 1
                conflict_timestamps.add(ts)


    if conflict_timestamps:
        conflict_tids += 1


    # ---------------------------------------------
    # Build true 1-second continuous timestamp runs
    # ---------------------------------------------

    timestamps = sorted(merged)

    if not timestamps:
        continue

    runs = []
    current = [timestamps[0]]

    for previous, current_ts in zip(
        timestamps,
        timestamps[1:]
    ):

        if current_ts - previous == 1000:
            current.append(current_ts)
        else:
            runs.append(current)
            current = [current_ts]

    runs.append(current)


    # ---------------------------------------------
    # Window coverage audit
    # ---------------------------------------------

    for run in runs:

        n = len(run)

        validity = [
            (
                1
                if (
                    merged[ts]["coord_valid"]
                    and ts not in conflict_timestamps
                )
                else 0
            )
            for ts in run
        ]

        prefix = [0]

        for value in validity:
            prefix.append(prefix[-1] + value)

        for L in WINDOWS:

            if n < L:
                continue

            number_of_windows = n - L + 1

            structural_window_counts[L] += number_of_windows
            structural_tid_eligible[L].add(key)

            split_window_counts[
                (split, L)
            ]["structural"] += number_of_windows

            split_tid_eligible[
                (split, L, "structural")
            ].add(key)

            class_window_counts[
                (folder_class, L)
            ]["structural"] += number_of_windows

            class_tid_eligible[
                (folder_class, L, "structural")
            ].add(key)

            for start_index in range(number_of_windows):

                valid_points = (
                    prefix[start_index + L]
                    - prefix[start_index]
                )

                ratio = valid_points / L

                for threshold in THRESHOLDS:

                    if ratio < threshold:
                        continue

                    window_counts[L][threshold] += 1
                    tid_eligible[L][threshold].add(key)

                    label = int(threshold * 100)

                    split_window_counts[
                        (split, L)
                    ][label] += 1

                    split_tid_eligible[
                        (split, L, label)
                    ].add(key)

                    class_window_counts[
                        (folder_class, L)
                    ][label] += 1

                    class_tid_eligible[
                        (folder_class, L, label)
                    ].add(key)


# ---------------------------------------------------------
# Close ZIP files
# ---------------------------------------------------------

for z in zip_handles.values():
    z.close()


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

print("=== REMAINING NEGATIVE OVERLAP AUDIT ===")
print("negative pairs                  =", negative_pairs)
print("cross official split            =", negative_cross_official)
print("shared timestamps, all exact    =", negative_shared_exact)
print("shared timestamps, conflicts    =", negative_shared_conflict)
print("no shared timestamp             =", negative_without_shared_timestamp)

print()
print("negative-gap distribution:")
for gap, count in negative_gap_counter.most_common():
    print(gap, "ms =", count)


print()
print("=== POINT-LEVEL OVERLAP AUDIT ===")
print("overlap observations            =", point_overlap_observations)
print("exact duplicate observations    =", point_overlap_exact)
print("conflicting observations        =", point_overlap_conflict)
print("TIDs containing conflicts       =", conflict_tids)
print("folder-class change TIDs        =", class_change_tids)


for L in WINDOWS:

    print()
    print(
        f"=== {L}-SAMPLE CONTINUOUS WINDOW COVERAGE ==="
    )

    print(
        "structural candidate windows =",
        structural_window_counts[L]
    )
    print(
        "TIDs with structural window  =",
        len(structural_tid_eligible[L])
    )

    for threshold in THRESHOLDS:

        label = int(threshold * 100)

        print(
            f">= {label}% valid-coordinate windows =",
            window_counts[L][threshold]
        )

        print(
            f"TIDs with >= {label}% window       =",
            len(tid_eligible[L][threshold])
        )


print()
print("=== 200-SAMPLE COVERAGE BY UID SPLIT ===")

for split in sorted(set(uid_split.values())):

    key = (split, 200)

    print()
    print(split.upper())

    print(
        "structural windows =",
        split_window_counts[key]["structural"]
    )

    print(
        "structural TIDs    =",
        len(
            split_tid_eligible[
                (split, 200, "structural")
            ]
        )
    )

    for label in [100, 95, 90, 80]:

        print(
            f">={label}% windows      =",
            split_window_counts[key][label]
        )

        print(
            f">={label}% TIDs         =",
            len(
                split_tid_eligible[
                    (split, 200, label)
                ]
            )
        )


print()
print("=== 200-SAMPLE COVERAGE BY FOLDER CLASS ===")

for cls in [
    "WALK",
    "BIKE",
    "CAR",
    "BUS",
    "SUBWAY",
    "ETC",
]:

    key = (cls, 200)

    print()
    print(cls)

    print(
        "structural windows =",
        class_window_counts[key]["structural"]
    )

    print(
        "structural TIDs    =",
        len(
            class_tid_eligible[
                (cls, 200, "structural")
            ]
        )
    )

    for label in [100, 95, 90, 80]:

        print(
            f">={label}% windows      =",
            class_window_counts[key][label]
        )

        print(
            f">={label}% TIDs         =",
            len(
                class_tid_eligible[
                    (cls, 200, label)
                ]
            )
        )


print()
print("NOTE:")
print(
    "No GPS values were imputed, deleted, or overwritten."
)
print(
    "Validity means observed finite latitude/longitude only."
)
print(
    "Threshold results are audit scenarios, not a final preprocessing policy."
)


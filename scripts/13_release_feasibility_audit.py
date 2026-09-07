from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import defaultdict, Counter
import csv
import io
import math
import re

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
SPLIT_FILE = ROOT / "splits" / "uid_split_v1.csv"

REPORT_DIR = ROOT / "reports" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = REPORT_DIR / "13_release_feasibility_by_class_20260908.csv"

NAME_RE = re.compile(
    r"^TMC-GPS-([^-]+)-([^-]+)-([^-]+)-Dataset\.csv$",
    re.I,
)

CLASSES = [
    "WALK",
    "BIKE",
    "CAR",
    "BUS",
    "SUBWAY",
]

POINTS_NEEDED = 201

CAPS = [
    ("strict_1s", 1000, True),
    ("max_gap_2s", 2000, False),
    ("max_gap_5s", 5000, False),
    ("max_gap_10s", 10000, False),
    ("max_gap_30s", 30000, False),
    ("max_gap_60s", 60000, False),
    ("unbounded", None, False),
]


def decode(raw):
    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass

    return raw.decode(
        "utf-8",
        errors="replace",
    )


def parse_float(value):
    value = str(
        value
        if value is not None
        else ""
    ).strip()

    if value == "":
        return None

    try:
        x = float(value)
    except ValueError:
        return None

    if not math.isfinite(x):
        return None

    return x


def valid_coord(lat, lon):
    return (
        lat is not None
        and lon is not None
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def has_201_point_run(
    timestamps,
    cap_ms,
    strict_exact=False,
):
    """
    True if this TID contains at least 201 valid GPS points
    whose 200 adjacent time gaps satisfy the condition.

    strict_1s:
        every dt must equal exactly 1000 ms

    released caps:
        every dt must be <= cap_ms

    unbounded:
        any 201 ordered valid points
    """

    if len(timestamps) < POINTS_NEEDED:
        return False

    if cap_ms is None:
        return True

    run = 1

    for a, b in zip(
        timestamps,
        timestamps[1:],
    ):
        dt = b - a

        if strict_exact:
            acceptable = (
                dt == 1000
            )
        else:
            acceptable = (
                0 < dt <= cap_ms
            )

        if acceptable:
            run += 1
        else:
            run = 1

        if run >= POINTS_NEEDED:
            return True

    return False


# ============================================================
# UID-disjoint split manifest
# ============================================================

uid_split = {}

with SPLIT_FILE.open(
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    fields = {
        x.lower(): x
        for x in reader.fieldnames
    }

    for row in reader:
        uid_split[
            row[fields["uid"]]
        ] = row[fields["split"]]


# ============================================================
# Index GPS CSV references
# ============================================================

tid_refs = defaultdict(list)
zip_handles = {}

for path in sorted(
    PACKAGES.glob("*_GPS_*.zip")
):

    if not (
        path.name.startswith("TS_")
        or path.name.startswith("VS_")
    ):
        continue

    class_name = (
        path.stem
        .split("_")[-1]
        .upper()
    )

    if class_name not in CLASSES:
        continue

    z = ZipFile(path)
    zip_handles[path] = z

    for member in z.namelist():

        if not member.lower().endswith(
            ".csv"
        ):
            continue

        base = PurePosixPath(
            member
        ).name

        match = NAME_RE.match(base)

        if not match:
            continue

        uid, tid, sid = match.groups()

        tid_refs[(uid, tid)].append({
            "path": path,
            "member": member,
            "class": class_name,
        })


print(
    "INDEXED UID/TID =",
    len(tid_refs),
)


# ============================================================
# Aggregates
# ============================================================

tid_counts = Counter()
eligible_counts = Counter()

gap_stats = defaultdict(
    lambda: {
        "valid_points": 0,
        "edges": 0,
        "eq_1s": 0,
        "gt_1s": 0,
        "gt_5s": 0,
        "gt_10s": 0,
        "gt_30s": 0,
        "gt_60s": 0,
        "max_gap_ms": 0,
    }
)

duplicate_conflict_timestamps = 0


# ============================================================
# TID audit
# ============================================================

for (uid, tid), refs in tid_refs.items():

    if uid not in uid_split:
        raise RuntimeError(
            "UID missing from split manifest"
        )

    split = uid_split[uid]

    classes = {
        ref["class"]
        for ref in refs
    }

    if len(classes) != 1:
        raise RuntimeError(
            "Multiple folder classes "
            "inside one UID/TID"
        )

    class_name = next(
        iter(classes)
    )

    tid_counts[
        (split, class_name)
    ] += 1


    # --------------------------------------------------------
    # Merge same timestamps.
    # Conflicting duplicates are excluded from valid points.
    # --------------------------------------------------------

    points = {}
    conflicts = set()

    for ref in refs:

        z = zip_handles[
            ref["path"]
        ]

        text = decode(
            z.read(
                ref["member"]
            )
        )

        reader = csv.DictReader(
            io.StringIO(text)
        )

        for row in reader:

            try:
                ts = int(
                    float(
                        row["timestamp"]
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            lat = parse_float(
                row.get("latitude")
            )

            lon = parse_float(
                row.get("longitude")
            )

            value = (
                lat,
                lon,
            )

            if ts in points:

                if points[ts] != value:
                    conflicts.add(ts)

            else:
                points[ts] = value


    duplicate_conflict_timestamps += len(
        conflicts
    )


    # --------------------------------------------------------
    # Valid ordered GPS points only.
    #
    # No imputation.
    # No coordinate filling.
    # Conflicting duplicate timestamps excluded.
    # --------------------------------------------------------

    timestamps = []

    for ts in sorted(points):

        if ts in conflicts:
            continue

        lat, lon = points[ts]

        if not valid_coord(
            lat,
            lon,
        ):
            continue

        timestamps.append(ts)


    stat = gap_stats[
        class_name
    ]

    stat[
        "valid_points"
    ] += len(timestamps)


    # --------------------------------------------------------
    # Gap distribution between adjacent VALID points.
    #
    # Note:
    # after invalid GPS points are removed, a larger dt may
    # represent missing observations and/or original sampling gap.
    # --------------------------------------------------------

    if len(timestamps) >= 2:

        for a, b in zip(
            timestamps,
            timestamps[1:],
        ):

            dt = b - a

            if dt <= 0:
                raise RuntimeError(
                    "Non-positive timestamp gap "
                    "after sorting unique timestamps"
                )

            stat["edges"] += 1

            if dt == 1000:
                stat["eq_1s"] += 1

            if dt > 1000:
                stat["gt_1s"] += 1

            if dt > 5000:
                stat["gt_5s"] += 1

            if dt > 10000:
                stat["gt_10s"] += 1

            if dt > 30000:
                stat["gt_30s"] += 1

            if dt > 60000:
                stat["gt_60s"] += 1

            stat["max_gap_ms"] = max(
                stat["max_gap_ms"],
                dt,
            )


    # --------------------------------------------------------
    # Feasibility under each release policy
    # --------------------------------------------------------

    for (
        cap_name,
        cap_ms,
        strict_exact,
    ) in CAPS:

        eligible = has_201_point_run(
            timestamps,
            cap_ms,
            strict_exact,
        )

        if eligible:
            eligible_counts[
                (
                    split,
                    class_name,
                    cap_name,
                )
            ] += 1


for z in zip_handles.values():
    z.close()


# ============================================================
# Console: gap distribution by class
# ============================================================

print()
print(
    "=== VALID-POINT GAP SUMMARY BY CLASS ==="
)

for class_name in CLASSES:

    s = gap_stats[
        class_name
    ]

    edges = s["edges"]

    def pct(value):
        if edges == 0:
            return 0.0

        return (
            value
            / edges
            * 100
        )

    print()
    print(class_name)

    print(
        "valid_points =",
        s["valid_points"],
    )

    print(
        "adjacent_valid_edges =",
        edges,
    )

    print(
        "dt == 1s =",
        s["eq_1s"],
        f"({pct(s['eq_1s']):.4f}%)",
    )

    print(
        "dt > 1s =",
        s["gt_1s"],
        f"({pct(s['gt_1s']):.4f}%)",
    )

    print(
        "dt > 5s =",
        s["gt_5s"],
        f"({pct(s['gt_5s']):.4f}%)",
    )

    print(
        "dt > 10s =",
        s["gt_10s"],
        f"({pct(s['gt_10s']):.4f}%)",
    )

    print(
        "dt > 30s =",
        s["gt_30s"],
        f"({pct(s['gt_30s']):.4f}%)",
    )

    print(
        "dt > 60s =",
        s["gt_60s"],
        f"({pct(s['gt_60s']):.4f}%)",
    )

    print(
        "max_gap_sec =",
        round(
            s["max_gap_ms"] / 1000,
            3,
        ),
    )


# ============================================================
# Console + CSV: eligible TIDs
# ============================================================

rows = []

print()
print(
    "=== 201 VALID-POINT FEASIBILITY ==="
)

for split in [
    "train",
    "validation",
    "internal_test",
]:

    print()
    print(split.upper())

    for class_name in CLASSES:

        total = tid_counts[
            (
                split,
                class_name,
            )
        ]

        result = {
            "split": split,
            "class_name": class_name,
            "total_tids": total,
        }

        for (
            cap_name,
            _,
            _,
        ) in CAPS:

            count = eligible_counts[
                (
                    split,
                    class_name,
                    cap_name,
                )
            ]

            result[
                cap_name
            ] = count

        strict = result[
            "strict_1s"
        ]

        result[
            "gain_max_gap_2s"
        ] = (
            result["max_gap_2s"]
            - strict
        )

        result[
            "gain_max_gap_5s"
        ] = (
            result["max_gap_5s"]
            - strict
        )

        result[
            "gain_max_gap_10s"
        ] = (
            result["max_gap_10s"]
            - strict
        )

        result[
            "gain_max_gap_30s"
        ] = (
            result["max_gap_30s"]
            - strict
        )

        result[
            "gain_max_gap_60s"
        ] = (
            result["max_gap_60s"]
            - strict
        )

        result[
            "gain_unbounded"
        ] = (
            result["unbounded"]
            - strict
        )

        rows.append(result)

        print(
            class_name,
            "total=",
            total,
            "strict=",
            strict,
            "2s=",
            result["max_gap_2s"],
            "5s=",
            result["max_gap_5s"],
            "10s=",
            result["max_gap_10s"],
            "30s=",
            result["max_gap_30s"],
            "60s=",
            result["max_gap_60s"],
            "unbounded=",
            result["unbounded"],
        )


# ============================================================
# Split totals
# ============================================================

print()
print(
    "=== TOTAL ELIGIBLE TIDS BY SPLIT ==="
)

for split in [
    "train",
    "validation",
    "internal_test",
]:

    print()
    print(split.upper())

    total = sum(
        tid_counts[
            (split, c)
        ]
        for c in CLASSES
    )

    print(
        "total_tids =",
        total,
    )

    for (
        cap_name,
        _,
        _,
    ) in CAPS:

        count = sum(
            eligible_counts[
                (
                    split,
                    c,
                    cap_name,
                )
            ]
            for c in CLASSES
        )

        print(
            cap_name,
            "=",
            count,
        )


# ============================================================
# Save privacy-safe aggregate report
# ============================================================

fieldnames = [
    "split",
    "class_name",
    "total_tids",
]

fieldnames += [
    x[0]
    for x in CAPS
]

fieldnames += [
    "gain_max_gap_2s",
    "gain_max_gap_5s",
    "gain_max_gap_10s",
    "gain_max_gap_30s",
    "gain_max_gap_60s",
    "gain_unbounded",
]


with OUT_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "duplicate_conflict_timestamps =",
    duplicate_conflict_timestamps,
)

print(
    "OUTPUT =",
    OUT_CSV,
)

print()
print(
    "NOTE:"
)

print(
    "- No missing-value imputation was performed."
)

print(
    "- No label-dependent speed filtering was performed."
)

print(
    "- No released windows were generated yet."
)

print(
    "- unbounded is diagnostic only and does NOT mean "
    "arbitrary gaps should be accepted."
)

print(
    "- This is a release-feasibility audit, not a model experiment."
)

print()
print(
    "RELEASE_FEASIBILITY_AUDIT_OK"
)

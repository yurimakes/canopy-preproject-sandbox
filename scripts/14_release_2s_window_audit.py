from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import defaultdict
import csv
import io
import math
import re

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
SPLIT_FILE = ROOT / "splits" / "uid_split_v1.csv"

REPORT_DIR = ROOT / "reports" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = REPORT_DIR / "14_release_2s_recovered_summary_20260908.csv"

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
        value if value is not None else ""
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


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    a = min(
        1.0,
        max(0.0, a),
    )

    return (
        2
        * r
        * math.asin(math.sqrt(a))
    )


def acceptable_gap(dt, policy):
    if policy == "strict":
        return dt == 1000

    if policy == "2s":
        return 0 < dt <= 2000

    raise ValueError(policy)


def find_first_window(
    timestamps,
    policy,
):
    """
    Find first 201-point run satisfying the gap policy.
    """

    if len(timestamps) < POINTS_NEEDED:
        return None

    run_start = 0

    for right in range(
        1,
        len(timestamps),
    ):
        dt = (
            timestamps[right]
            - timestamps[right - 1]
        )

        if not acceptable_gap(
            dt,
            policy,
        ):
            run_start = right

        run_len = (
            right
            - run_start
            + 1
        )

        if run_len >= POINTS_NEEDED:
            return timestamps[
                run_start:
                run_start + POINTS_NEEDED
            ]

    return None


def percentile(values, q):
    if not values:
        return None

    xs = sorted(values)

    if len(xs) == 1:
        return float(xs[0])

    pos = (
        (len(xs) - 1)
        * q
    )

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return float(xs[lo])

    weight = pos - lo

    return (
        xs[lo] * (1 - weight)
        + xs[hi] * weight
    )


def median(values):
    return percentile(
        values,
        0.5,
    )


# ============================================================
# UID split
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
# Index GPS files
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

        m = NAME_RE.match(base)

        if not m:
            continue

        uid, tid, sid = m.groups()

        tid_refs[
            (uid, tid)
        ].append({
            "path": path,
            "member": member,
            "class": class_name,
        })


print(
    "INDEXED UID/TID =",
    len(tid_refs),
)


# ============================================================
# Strict-fail / 2s-pass TIDs only
# ============================================================

recovered = defaultdict(list)

for (uid, tid), refs in tid_refs.items():

    split = uid_split.get(uid)

    # --------------------------------------------------------
    # Do NOT use Internal Test in release-policy selection.
    # --------------------------------------------------------

    if split not in (
        "train",
        "validation",
    ):
        continue

    classes = {
        x["class"]
        for x in refs
    }

    if len(classes) != 1:
        raise RuntimeError(
            "Multiple classes inside UID/TID"
        )

    class_name = next(
        iter(classes)
    )


    # --------------------------------------------------------
    # Merge timestamps
    # --------------------------------------------------------

    points = {}
    conflicts = set()

    for ref in refs:

        z = zip_handles[
            ref["path"]
        ]

        reader = csv.DictReader(
            io.StringIO(
                decode(
                    z.read(
                        ref["member"]
                    )
                )
            )
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


    # --------------------------------------------------------
    # Valid ordered points only
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


    # --------------------------------------------------------
    # Recover only TIDs that:
    #
    # strict 1s => FAIL
    # max-gap 2s => PASS
    # --------------------------------------------------------

    strict_window = find_first_window(
        timestamps,
        "strict",
    )

    if strict_window is not None:
        continue

    released_window = find_first_window(
        timestamps,
        "2s",
    )

    if released_window is None:
        continue


    # --------------------------------------------------------
    # Window gap structure
    # --------------------------------------------------------

    gaps_ms = [
        b - a
        for a, b in zip(
            released_window,
            released_window[1:],
        )
    ]

    if len(gaps_ms) != 200:
        raise RuntimeError(
            "Unexpected gap count"
        )

    non_1s = sum(
        dt != 1000
        for dt in gaps_ms
    )

    eq_2s = sum(
        dt == 2000
        for dt in gaps_ms
    )

    between_1_2s = sum(
        1000 < dt < 2000
        for dt in gaps_ms
    )

    sub_1s = sum(
        0 < dt < 1000
        for dt in gaps_ms
    )

    wall_span_sec = (
        released_window[-1]
        - released_window[0]
    ) / 1000.0

    max_gap_sec = (
        max(gaps_ms)
        / 1000.0
    )


    # --------------------------------------------------------
    # Calculate speeds using ACTUAL dt
    #
    # No assumption that dt == 1 second.
    # --------------------------------------------------------

    speeds = []

    for t1, t2 in zip(
        released_window,
        released_window[1:],
    ):

        lat1, lon1 = points[t1]
        lat2, lon2 = points[t2]

        dt_sec = (
            t2 - t1
        ) / 1000.0

        distance_m = haversine_m(
            lat1,
            lon1,
            lat2,
            lon2,
        )

        speed_kmh = (
            distance_m
            / dt_sec
            * 3.6
        )

        speeds.append(
            speed_kmh
        )


    recovered[
        (split, class_name)
    ].append({
        "non_1s_edges": non_1s,
        "eq_2s_edges": eq_2s,
        "between_1_2s_edges": between_1_2s,
        "sub_1s_edges": sub_1s,
        "wall_span_sec": wall_span_sec,
        "max_gap_sec": max_gap_sec,
        "max_speed_kmh": max(speeds),
        "over_200_edges": sum(
            x > 200
            for x in speeds
        ),
    })


for z in zip_handles.values():
    z.close()


# ============================================================
# Aggregate
# ============================================================

rows = []

print()
print(
    "=== STRICT-FAIL / 2S-PASS RECOVERED WINDOWS ==="
)

for split in (
    "train",
    "validation",
):

    print()
    print(split.upper())

    for class_name in CLASSES:

        items = recovered[
            (split, class_name)
        ]

        count = len(items)

        non1 = [
            x["non_1s_edges"]
            for x in items
        ]

        eq2 = [
            x["eq_2s_edges"]
            for x in items
        ]

        spans = [
            x["wall_span_sec"]
            for x in items
        ]

        max_speeds = [
            x["max_speed_kmh"]
            for x in items
        ]

        over200_total = sum(
            x["over_200_edges"]
            for x in items
        )

        result = {
            "split": split,
            "class_name": class_name,
            "recovered_tids": count,

            "non_1s_edges_median":
                median(non1),

            "non_1s_edges_p90":
                percentile(non1, 0.90),

            "non_1s_edges_max":
                max(non1)
                if non1
                else None,

            "eq_2s_edges_median":
                median(eq2),

            "wall_span_sec_median":
                median(spans),

            "wall_span_sec_p90":
                percentile(spans, 0.90),

            "wall_span_sec_max":
                max(spans)
                if spans
                else None,

            "max_speed_kmh_median":
                median(max_speeds),

            "max_speed_kmh_p90":
                percentile(
                    max_speeds,
                    0.90,
                ),

            "over_200_speed_edges":
                over200_total,
        }

        rows.append(result)

        print(
            class_name,
            "recovered=",
            count,
            "non1s_med=",
            result[
                "non_1s_edges_median"
            ],
            "non1s_p90=",
            result[
                "non_1s_edges_p90"
            ],
            "span_med=",
            result[
                "wall_span_sec_median"
            ],
            "span_p90=",
            result[
                "wall_span_sec_p90"
            ],
            ">200speed_edges=",
            over200_total,
        )


# ============================================================
# Overall Train / Validation
# ============================================================

print()
print(
    "=== OVERALL RELEASED WINDOW SUMMARY ==="
)

for split in (
    "train",
    "validation",
):

    items = []

    for class_name in CLASSES:
        items.extend(
            recovered[
                (split, class_name)
            ]
        )

    non1 = [
        x["non_1s_edges"]
        for x in items
    ]

    spans = [
        x["wall_span_sec"]
        for x in items
    ]

    print()
    print(split.upper())

    print(
        "recovered_tids =",
        len(items),
    )

    print(
        "non_1s_edges median =",
        median(non1),
    )

    print(
        "non_1s_edges p90 =",
        percentile(
            non1,
            0.90,
        ),
    )

    print(
        "non_1s_edges max =",
        max(non1)
        if non1
        else None,
    )

    print(
        "wall_span_sec median =",
        median(spans),
    )

    print(
        "wall_span_sec p90 =",
        percentile(
            spans,
            0.90,
        ),
    )

    print(
        "wall_span_sec max =",
        max(spans)
        if spans
        else None,
    )

    print(
        ">200 km/h speed edges =",
        sum(
            x["over_200_edges"]
            for x in items
        ),
    )


# ============================================================
# Save aggregate-only CSV
# ============================================================

fieldnames = [
    "split",
    "class_name",
    "recovered_tids",
    "non_1s_edges_median",
    "non_1s_edges_p90",
    "non_1s_edges_max",
    "eq_2s_edges_median",
    "wall_span_sec_median",
    "wall_span_sec_p90",
    "wall_span_sec_max",
    "max_speed_kmh_median",
    "max_speed_kmh_p90",
    "over_200_speed_edges",
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
    "OUTPUT =",
    OUT_CSV,
)

print()
print(
    "NOTE:"
)

print(
    "- Only Train and Validation were audited."
)

print(
    "- Internal Test was NOT used."
)

print(
    "- Only strict-fail / 2s-pass TIDs were analyzed."
)

print(
    "- No interpolation or imputation was used."
)

print(
    "- Speed was calculated with each edge's actual time difference."
)

print(
    "- No model training was performed."
)

print()
print(
    "RELEASE_2S_WINDOW_AUDIT_OK"
)

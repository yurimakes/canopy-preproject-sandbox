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

OUT_CSV = REPORT_DIR / "15_release_2s_strata_20260908.csv"

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


def edge_bin(n):

    if n <= 2:
        return "01_1to2"

    if n <= 5:
        return "02_3to5"

    if n <= 10:
        return "03_6to10"

    if n <= 20:
        return "04_11to20"

    return "05_over20"


def span_bin(span):

    if span <= 202:
        return "01_le202"

    if span <= 205:
        return "02_203to205"

    if span <= 210:
        return "03_206to210"

    if span <= 220:
        return "04_211to220"

    return "05_over220"


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
# Index GPS
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
# Recover strict-fail / 2s-pass windows
# ============================================================

items = []

for (uid, tid), refs in tid_refs.items():

    split = uid_split.get(uid)

    # Internal Test intentionally excluded.
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
            "Multiple classes in UID/TID"
        )

    class_name = next(
        iter(classes)
    )


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


    timestamps = []

    for ts in sorted(points):

        if ts in conflicts:
            continue

        lat, lon = points[ts]

        if valid_coord(
            lat,
            lon,
        ):
            timestamps.append(ts)


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


    gaps = [
        b - a
        for a, b in zip(
            released_window,
            released_window[1:],
        )
    ]

    assert len(gaps) == 200


    non_1s = sum(
        dt != 1000
        for dt in gaps
    )

    eq_2s = sum(
        dt == 2000
        for dt in gaps
    )

    between_1_2s = sum(
        1000 < dt < 2000
        for dt in gaps
    )

    sub_1s = sum(
        0 < dt < 1000
        for dt in gaps
    )

    wall_span = (
        released_window[-1]
        - released_window[0]
    ) / 1000.0


    items.append({
        "split": split,
        "class_name": class_name,
        "non_1s_edges": non_1s,
        "eq_2s_edges": eq_2s,
        "between_1_2s_edges": between_1_2s,
        "sub_1s_edges": sub_1s,
        "wall_span_sec": wall_span,
        "edge_bin": edge_bin(
            non_1s
        ),
        "span_bin": span_bin(
            wall_span
        ),
    })


for z in zip_handles.values():
    z.close()


# ============================================================
# Sanity check against Audit 14
# ============================================================

train_total = sum(
    x["split"] == "train"
    for x in items
)

val_total = sum(
    x["split"] == "validation"
    for x in items
)

print()
print("=== SANITY CHECK ===")

print(
    "Train recovered =",
    train_total
)

print(
    "Validation recovered =",
    val_total
)

assert train_total == 818
assert val_total == 94


# ============================================================
# Non-1s edge strata
# ============================================================

print()
print(
    "=== NON-1S EDGE STRATA ==="
)

edge_counts = Counter(
    (
        x["split"],
        x["edge_bin"],
    )
    for x in items
)

for split in (
    "train",
    "validation",
):

    print()
    print(split.upper())

    total = sum(
        x["split"] == split
        for x in items
    )

    for bucket in (
        "01_1to2",
        "02_3to5",
        "03_6to10",
        "04_11to20",
        "05_over20",
    ):

        count = edge_counts[
            (
                split,
                bucket,
            )
        ]

        pct = (
            count / total * 100
            if total
            else 0
        )

        print(
            bucket,
            "=",
            count,
            f"({pct:.2f}%)",
        )


# ============================================================
# Wall-clock span strata
# ============================================================

print()
print(
    "=== WALL-SPAN STRATA ==="
)

span_counts = Counter(
    (
        x["split"],
        x["span_bin"],
    )
    for x in items
)

for split in (
    "train",
    "validation",
):

    print()
    print(split.upper())

    total = sum(
        x["split"] == split
        for x in items
    )

    for bucket in (
        "01_le202",
        "02_203to205",
        "03_206to210",
        "04_211to220",
        "05_over220",
    ):

        count = span_counts[
            (
                split,
                bucket,
            )
        ]

        pct = (
            count / total * 100
            if total
            else 0
        )

        print(
            bucket,
            "=",
            count,
            f"({pct:.2f}%)",
        )


# ============================================================
# Cumulative mild-release candidates
# ============================================================

CANDIDATES = [
    (
        "edge_le2_span_le202",
        2,
        202,
    ),
    (
        "edge_le5_span_le205",
        5,
        205,
    ),
    (
        "edge_le10_span_le210",
        10,
        210,
    ),
    (
        "edge_le20_span_le220",
        20,
        220,
    ),
    (
        "all_2s_released",
        None,
        None,
    ),
]


print()
print(
    "=== CUMULATIVE RELEASE CANDIDATES ==="
)

output_rows = []

for split in (
    "train",
    "validation",
):

    print()
    print(split.upper())

    split_items = [
        x
        for x in items
        if x["split"] == split
    ]

    for (
        name,
        edge_cap,
        span_cap,
    ) in CANDIDATES:

        if edge_cap is None:

            selected = split_items

        else:

            selected = [
                x
                for x in split_items
                if (
                    x["non_1s_edges"]
                    <= edge_cap
                    and
                    x["wall_span_sec"]
                    <= span_cap
                )
            ]


        print()
        print(
            name,
            "=",
            len(selected),
        )

        for class_name in CLASSES:

            class_count = sum(
                x["class_name"]
                == class_name
                for x in selected
            )

            print(
                " ",
                class_name,
                "=",
                class_count,
            )

            output_rows.append({
                "split": split,
                "candidate": name,
                "class_name": class_name,
                "recovered_tids": class_count,
            })


# ============================================================
# Gap type check
# ============================================================

print()
print(
    "=== NON-1S GAP TYPE CHECK ==="
)

for split in (
    "train",
    "validation",
):

    split_items = [
        x
        for x in items
        if x["split"] == split
    ]

    print()
    print(split.upper())

    print(
        "eq_2s_edges =",
        sum(
            x["eq_2s_edges"]
            for x in split_items
        ),
    )

    print(
        "between_1_2s_edges =",
        sum(
            x[
                "between_1_2s_edges"
            ]
            for x in split_items
        ),
    )

    print(
        "sub_1s_edges =",
        sum(
            x["sub_1s_edges"]
            for x in split_items
        ),
    )


# ============================================================
# Save aggregate-only CSV
# ============================================================

with OUT_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "candidate",
            "class_name",
            "recovered_tids",
        ],
    )

    writer.writeheader()
    writer.writerows(
        output_rows
    )


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
    "- Train and Validation only."
)

print(
    "- Internal Test was NOT used."
)

print(
    "- No model training."
)

print(
    "- No interpolation / imputation."
)

print(
    "- This audit is only for choosing a mild-release candidate."
)

print()
print(
    "RELEASE_2S_STRATA_AUDIT_OK"
)

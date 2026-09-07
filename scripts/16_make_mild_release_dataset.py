from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import defaultdict, Counter
import csv
import hashlib
import io
import math
import re

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
SPLIT_FILE = ROOT / "splits" / "uid_split_v1.csv"

STRICT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "speedtransformer_smoke_v1"
    / "speed200_smoke_v1.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "speedtransformer_mild_release_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "speed200_mild_release_v1.csv"
)

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

LABEL_MAP = {
    "WALK": 0,
    "BIKE": 1,
    "CAR": 2,
    "BUS": 3,
    "SUBWAY": 4,
}

POINTS_NEEDED = 201
SPEEDS_NEEDED = 200

MAX_2S_EDGES = 5
MAX_WALL_SPAN_SEC = 205.0

ADD_PER_CLASS = 29
SEED = 316


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


def sample_id_for(uid, tid):
    raw = f"{uid}|{tid}".encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()[:16]


def selection_key(uid, tid):
    raw = (
        f"{SEED}|{uid}|{tid}"
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    r = 6371008.8

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

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
        * math.asin(
            math.sqrt(a)
        )
    )


def find_strict_window(
    timestamps,
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

        if dt != 1000:
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


def find_first_2s_window(
    timestamps,
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

        if not (
            0 < dt <= 2000
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


# ============================================================
# Load existing strict dataset
# ============================================================

if not STRICT_CSV.exists():
    raise FileNotFoundError(
        f"Strict dataset not found: {STRICT_CSV}"
    )


strict_rows = []

with STRICT_CSV.open(
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    required = {
        "sample_id",
        "split",
        "model_label",
        "step",
        "speed_kmh",
    }

    missing = (
        required
        - set(reader.fieldnames or [])
    )

    if missing:
        raise RuntimeError(
            f"Strict CSV missing columns: {sorted(missing)}"
        )

    for row in reader:

        # Internal Test intentionally excluded
        if row["split"] not in (
            "train",
            "validation",
        ):
            continue

        strict_rows.append(row)


# ============================================================
# Verify exact strict sample composition
# ============================================================

strict_sample_meta = {}

for row in strict_rows:

    sid = row["sample_id"]

    split = row["split"]

    label = int(
        row["model_label"]
    )

    meta = (
        split,
        label,
    )

    if sid in strict_sample_meta:

        if strict_sample_meta[sid] != meta:
            raise RuntimeError(
                "Inconsistent strict sample metadata"
            )

    else:
        strict_sample_meta[sid] = meta


strict_counts = Counter(
    strict_sample_meta.values()
)


print(
    "=== STRICT DATASET CHECK ==="
)

for split in (
    "train",
    "validation",
):

    for class_name in CLASSES:

        label = LABEL_MAP[
            class_name
        ]

        count = strict_counts[
            (
                split,
                label,
            )
        ]

        print(
            split,
            class_name,
            "=",
            count,
        )


for class_name in CLASSES:

    label = LABEL_MAP[
        class_name
    ]

    assert strict_counts[
        ("train", label)
    ] == 100

    assert strict_counts[
        ("validation", label)
    ] == 30


assert len(
    strict_sample_meta
) == 650


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
# Index GPS package files
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


print()
print(
    "INDEXED UID/TID =",
    len(tid_refs),
)


# ============================================================
# Build Train mild-release candidates
# ============================================================

candidates = defaultdict(list)

for (uid, tid), refs in tid_refs.items():

    split = uid_split.get(uid)

    # Only additional TRAIN data is generated here.
    if split != "train":
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
    # Merge timestamp -> coordinate
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


    # --------------------------------------------------------
    # Must FAIL strict policy
    # --------------------------------------------------------

    if find_strict_window(
        timestamps
    ) is not None:
        continue


    # --------------------------------------------------------
    # Must PASS 2-second continuity
    # --------------------------------------------------------

    window = find_first_2s_window(
        timestamps
    )

    if window is None:
        continue


    gaps_ms = [
        b - a
        for a, b in zip(
            window,
            window[1:],
        )
    ]

    assert len(gaps_ms) == 200


    # Dataset audit showed all non-1s edges were exactly 2 sec.
    # Keep explicit checks anyway.
    invalid_gap = any(
        dt not in (
            1000,
            2000,
        )
        for dt in gaps_ms
    )

    if invalid_gap:
        continue


    two_sec_edges = sum(
        dt == 2000
        for dt in gaps_ms
    )

    wall_span_sec = (
        window[-1]
        - window[0]
    ) / 1000.0


    # --------------------------------------------------------
    # Mild-release policy
    # --------------------------------------------------------

    if two_sec_edges > MAX_2S_EDGES:
        continue

    if wall_span_sec > MAX_WALL_SPAN_SEC:
        continue


    # --------------------------------------------------------
    # Compute 200 speeds using ACTUAL dt
    # --------------------------------------------------------

    speeds = []

    for t1, t2 in zip(
        window,
        window[1:],
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

        if not math.isfinite(
            speed_kmh
        ):
            raise RuntimeError(
                "Non-finite speed"
            )

        speeds.append(
            speed_kmh
        )


    assert len(speeds) == SPEEDS_NEEDED


    sid = sample_id_for(
        uid,
        tid,
    )


    # Should not overlap strict samples.
    if sid in strict_sample_meta:
        raise RuntimeError(
            "Mild candidate overlaps strict sample"
        )


    candidates[
        class_name
    ].append({
        "sample_id": sid,
        "selection_key":
            selection_key(
                uid,
                tid,
            ),
        "model_label":
            LABEL_MAP[
                class_name
            ],
        "class_name":
            class_name,
        "speeds":
            speeds,
        "two_sec_edges":
            two_sec_edges,
        "wall_span_sec":
            wall_span_sec,
    })


for z in zip_handles.values():
    z.close()


# ============================================================
# Candidate sanity check against Audit 15
# ============================================================

EXPECTED_CANDIDATES = {
    "WALK": 148,
    "BIKE": 29,
    "CAR": 199,
    "BUS": 134,
    "SUBWAY": 85,
}


print()
print(
    "=== MILD CANDIDATE CHECK ==="
)

for class_name in CLASSES:

    count = len(
        candidates[class_name]
    )

    expected = EXPECTED_CANDIDATES[
        class_name
    ]

    print(
        class_name,
        "candidates =",
        count,
        "expected =",
        expected,
    )

    assert count == expected


# ============================================================
# Deterministic balanced selection
# ============================================================

selected = []

for class_name in CLASSES:

    ordered = sorted(
        candidates[class_name],
        key=lambda x:
            x["selection_key"],
    )

    chosen = ordered[
        :ADD_PER_CLASS
    ]

    assert len(chosen) == ADD_PER_CLASS

    selected.extend(
        chosen
    )


print()
print(
    "=== SELECTED MILD TRAIN SAMPLES ==="
)

selected_counts = Counter(
    x["class_name"]
    for x in selected
)

for class_name in CLASSES:

    print(
        class_name,
        "=",
        selected_counts[
            class_name
        ],
    )

    assert selected_counts[
        class_name
    ] == ADD_PER_CLASS


assert len(selected) == 145


# ============================================================
# Construct clean output rows
# ============================================================

out_rows = []


# Existing strict Train + Validation
for row in strict_rows:

    label = int(
        row["model_label"]
    )

    class_name = CLASSES[
        label
    ]

    out_rows.append({
        "sample_id":
            row["sample_id"],

        "split":
            row["split"],

        "model_label":
            label,

        "class_name":
            class_name,

        "step":
            int(
                row["step"]
            ),

        "speed_kmh":
            float(
                row["speed_kmh"]
            ),

        "source_policy":
            "strict_1s",

        "two_sec_edges":
            0,

        "wall_span_sec":
            200.0,
    })


# Additional mild-release Train
for sample in selected:

    for step, speed in enumerate(
        sample["speeds"]
    ):

        out_rows.append({
            "sample_id":
                sample["sample_id"],

            "split":
                "train",

            "model_label":
                sample[
                    "model_label"
                ],

            "class_name":
                sample[
                    "class_name"
                ],

            "step":
                step,

            "speed_kmh":
                speed,

            "source_policy":
                "mild_2s_le5_span205",

            "two_sec_edges":
                sample[
                    "two_sec_edges"
                ],

            "wall_span_sec":
                sample[
                    "wall_span_sec"
                ],
        })


# ============================================================
# Final integrity checks
# ============================================================

sample_meta = {}

sample_steps = defaultdict(list)

for row in out_rows:

    sid = row["sample_id"]

    meta = (
        row["split"],
        int(
            row["model_label"]
        ),
        row["source_policy"],
    )

    if sid in sample_meta:

        if sample_meta[sid] != meta:
            raise RuntimeError(
                "Sample metadata conflict"
            )

    else:
        sample_meta[sid] = meta

    sample_steps[
        sid
    ].append(
        int(
            row["step"]
        )
    )


for sid, steps in sample_steps.items():

    if sorted(steps) != list(
        range(200)
    ):
        raise RuntimeError(
            "Sample does not contain exactly steps 0..199"
        )


final_counts = Counter(
    (
        split,
        label,
    )
    for (
        split,
        label,
        policy,
    ) in sample_meta.values()
)


print()
print(
    "=== FINAL DATASET SAMPLE COUNTS ==="
)

for split in (
    "train",
    "validation",
):

    print()
    print(
        split.upper()
    )

    for class_name in CLASSES:

        label = LABEL_MAP[
            class_name
        ]

        count = final_counts[
            (
                split,
                label,
            )
        ]

        print(
            class_name,
            "=",
            count,
        )


for class_name in CLASSES:

    label = LABEL_MAP[
        class_name
    ]

    assert final_counts[
        ("train", label)
    ] == 129

    assert final_counts[
        ("validation", label)
    ] == 30


assert len(sample_meta) == 795
assert len(out_rows) == 159000


policy_counts = Counter(
    policy
    for (
        split,
        label,
        policy,
    ) in sample_meta.values()
)


print()
print(
    "=== SOURCE POLICY COUNTS ==="
)

for key in sorted(
    policy_counts
):
    print(
        key,
        "=",
        policy_counts[key],
    )


assert policy_counts[
    "strict_1s"
] == 650

assert policy_counts[
    "mild_2s_le5_span205"
] == 145


# ============================================================
# Privacy / leakage sanity checks
# ============================================================

train_ids = {
    sid
    for sid, (
        split,
        label,
        policy,
    ) in sample_meta.items()
    if split == "train"
}

val_ids = {
    sid
    for sid, (
        split,
        label,
        policy,
    ) in sample_meta.items()
    if split == "validation"
}

strict_val_ids = {
    sid
    for sid, (
        split,
        label,
    ) in strict_sample_meta.items()
    if split == "validation"
}


assert train_ids.isdisjoint(
    val_ids
)

assert val_ids == strict_val_ids


# ============================================================
# Write processed CSV only after all integrity checks pass
# ============================================================

fieldnames = [
    "sample_id",
    "split",
    "model_label",
    "class_name",
    "step",
    "speed_kmh",
    "source_policy",
    "two_sec_edges",
    "wall_span_sec",
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
    writer.writerows(
        out_rows
    )


print()
print(
    "=== FINAL CHECK ==="
)

print(
    "samples =",
    len(sample_meta),
)

print(
    "rows =",
    len(out_rows),
)

print(
    "train samples =",
    len(train_ids),
)

print(
    "validation samples =",
    len(val_ids),
)

print(
    "train/validation sample_id overlap =",
    len(
        train_ids
        & val_ids
    ),
)

print(
    "strict/mild Validation sample_id sets identical =",
    val_ids == strict_val_ids,
)

print(
    "Internal Test included =",
    False,
)

print(
    "OUTPUT =",
    OUT_CSV.relative_to(
        ROOT
    ).as_posix(),
)

print()
print(
    "NOTE:"
)

print(
    "- Existing strict Validation samples were preserved unchanged."
)

print(
    "- Mild-release data was added to Train only."
)

print(
    "- Mild Train additions are balanced at 29 samples per class."
)

print(
    "- No interpolation or missing-value imputation."
)

print(
    "- No label-dependent speed cutoff."
)

print(
    "- Mild-release speed uses actual timestamp differences."
)

print(
    "- Raw UID/TID values are not written to the output."
)

print(
    "- This dataset is for a controlled Validation diagnostic, not final performance reporting."
)

print()
print(
    "MILD_RELEASE_DATASET_OK"
)

from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import defaultdict, Counter
import csv
import hashlib
import io
import math
import random
import re

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
SPLIT_FILE = ROOT / "splits" / "uid_split_v1.csv"
OUT_DIR = ROOT / "data" / "processed" / "speedtransformer_smoke_v1"

OUT_DIR.mkdir(parents=True, exist_ok=True)

NAME_RE = re.compile(
    r"^TMC-GPS-([^-]+)-([^-]+)-([^-]+)-Dataset\.csv$",
    re.I
)

CLASS_MAP = {
    "WALK": 0,
    "BIKE": 1,
    "CAR": 2,
    "BUS": 3,
    "SUBWAY": 4,
}

# Smoke-only cap.
# Not a performance evaluation dataset.
LIMITS = {
    "train": 100,
    "validation": 30,
    "internal_test": 30,
}

SEED = 316
POINTS_NEEDED = 201
SPEEDS_NEEDED = 200


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def parse_float(value):
    value = str(value if value is not None else "").strip()

    if value == "":
        return None

    try:
        x = float(value)
    except ValueError:
        return None

    if not math.isfinite(x):
        return None

    return x


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return 2 * r * math.asin(math.sqrt(a))


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
# Index GPS files
# ---------------------------------------------------------

tid_refs = defaultdict(list)
zip_handles = {}

for path in sorted(PACKAGES.glob("*_GPS_*.zip")):

    if path.name.startswith("TS_") or path.name.startswith("VS_"):
        pass
    else:
        continue

    folder_class = path.stem.split("_")[-1].upper()

    if folder_class not in CLASS_MAP:
        continue

    z = ZipFile(path)
    zip_handles[path] = z

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
            "class": folder_class,
        })


# ---------------------------------------------------------
# Build one clean 200-speed sample per TID
# ---------------------------------------------------------

candidates = defaultdict(list)

for (uid, tid), refs in tid_refs.items():

    if uid not in uid_split:
        raise RuntimeError(f"UID missing from split manifest: {uid}")

    split = uid_split[uid]

    classes = {x["class"] for x in refs}

    if len(classes) != 1:
        raise RuntimeError(
            f"Multiple folder classes inside UID/TID: {uid}/{tid}"
        )

    class_name = next(iter(classes))
    model_label = CLASS_MAP[class_name]

    # Merge identical timestamp observations.
    points = {}
    conflicts = set()

    for ref in refs:

        z = zip_handles[ref["path"]]

        rows = csv.DictReader(
            io.StringIO(
                decode(z.read(ref["member"]))
            )
        )

        for row in rows:

            ts = int(float(row["timestamp"]))

            lat = parse_float(row.get("latitude"))
            lon = parse_float(row.get("longitude"))

            value = (lat, lon)

            if ts in points and points[ts] != value:
                conflicts.add(ts)
            else:
                points[ts] = value


    timestamps = sorted(points)

    if len(timestamps) < POINTS_NEEDED:
        continue

    # True 1-second continuous runs.
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

    sample_speeds = None

    for run in runs:

        if len(run) < POINTS_NEEDED:
            continue

        # Search deterministic first clean 201-point window.
        for start in range(
            len(run) - POINTS_NEEDED + 1
        ):

            selected = run[
                start:start + POINTS_NEEDED
            ]

            valid = True

            for ts in selected:

                lat, lon = points[ts]

                if (
                    ts in conflicts
                    or lat is None
                    or lon is None
                    or not (-90 <= lat <= 90)
                    or not (-180 <= lon <= 180)
                ):
                    valid = False
                    break

            if not valid:
                continue

            speeds = []

            for t1, t2 in zip(
                selected,
                selected[1:]
            ):

                lat1, lon1 = points[t1]
                lat2, lon2 = points[t2]

                speed_mps = haversine_m(
                    lat1,
                    lon1,
                    lat2,
                    lon2,
                )

                speed_kmh = speed_mps * 3.6
                speeds.append(speed_kmh)

            if len(speeds) != SPEEDS_NEEDED:
                raise RuntimeError(
                    "Unexpected speed sequence length"
                )

            sample_speeds = speeds
            break

        if sample_speeds is not None:
            break


    if sample_speeds is None:
        continue


    # Privacy-safe deterministic sample identifier.
    sample_id = hashlib.sha256(
        f"{uid}|{tid}".encode("utf-8")
    ).hexdigest()[:16]

    candidates[
        (split, class_name)
    ].append({
        "sample_id": sample_id,
        "split": split,
        "class_name": class_name,
        "model_label": model_label,
        "speeds": sample_speeds,
    })


for z in zip_handles.values():
    z.close()


# ---------------------------------------------------------
# Deterministic balanced smoke subset
# ---------------------------------------------------------

rng = random.Random(SEED)
selected_samples = []

print("=== AVAILABLE CLEAN TIDS ===")

for split in [
    "train",
    "validation",
    "internal_test",
]:

    print()
    print(split.upper())

    for class_name in CLASS_MAP:

        group = candidates[
            (split, class_name)
        ]

        group.sort(
            key=lambda x: x["sample_id"]
        )

        rng.shuffle(group)

        available = len(group)
        take = min(
            LIMITS[split],
            available
        )

        selected = group[:take]
        selected_samples.extend(selected)

        print(
            class_name,
            "available =",
            available,
            "selected =",
            take,
        )


# ---------------------------------------------------------
# Save long-form CSV
# ---------------------------------------------------------

output_csv = OUT_DIR / "speed200_smoke_v1.csv"

with output_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample_id",
        "split",
        "class_name",
        "model_label",
        "step",
        "speed_kmh",
    ])

    for sample in selected_samples:

        for step, speed in enumerate(
            sample["speeds"]
        ):

            writer.writerow([
                sample["sample_id"],
                sample["split"],
                sample["class_name"],
                sample["model_label"],
                step,
                f"{speed:.10f}",
            ])


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

summary = Counter(
    (
        x["split"],
        x["class_name"],
    )
    for x in selected_samples
)

print()
print("=== SELECTED SMOKE DATASET ===")
print("samples =", len(selected_samples))
print("speed rows =", len(selected_samples) * 200)

for key in sorted(summary):
    print(
        key[0],
        key[1],
        "=",
        summary[key],
    )

print()
print("OUTPUT =", output_csv)
print()
print("NOTE:")
print(
    "No label-dependent speed filtering was applied."
)
print(
    "No missing-value imputation, speed clipping, or smoothing was applied."
)
print(
    "One clean 200-speed window per TID was used."
)
print(
    "This dataset is for smoke training only, not final performance evaluation."
)

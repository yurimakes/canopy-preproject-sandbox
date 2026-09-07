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

DIAGNOSTIC_SPEED_CAPS = [30, 50, 80, 100]


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def parse_coord(value):
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


def speed_bin(speed):
    if speed == 0:
        return "0"
    if speed <= 1:
        return "(0,1]"
    if speed <= 3:
        return "(1,3]"
    if speed <= 7:
        return "(3,7]"
    if speed <= 15:
        return "(7,15]"
    if speed <= 30:
        return "(15,30]"
    if speed <= 50:
        return "(30,50]"
    if speed <= 80:
        return "(50,80]"
    if speed <= 100:
        return "(80,100]"
    return ">100"


# ---------------------------------------------------------
# UID-disjoint split
# ---------------------------------------------------------

uid_split = {}

with SPLIT_FILE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    fields = {
        name.lower(): name
        for name in reader.fieldnames
    }

    for row in reader:
        uid_split[row[fields["uid"]]] = row[fields["split"]]


# ---------------------------------------------------------
# Index GPS files
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
        official_split = "official_training"
    elif path.name.startswith("VS_"):
        official_split = "official_validation"
    else:
        continue

    folder_class = path.stem.split("_")[-1].upper()

    for member in z.namelist():

        if not member.lower().endswith(".csv"):
            continue

        base = PurePosixPath(member).name
        match = NAME_RE.match(base)

        if not match:
            continue

        uid, tid, sid = match.groups()

        tid_refs[(uid, tid)].append({
            "path": path,
            "member": member,
            "sid": sid,
            "class": folder_class,
            "official_split": official_split,
        })


# ---------------------------------------------------------
# Global audit counters
# ---------------------------------------------------------

unique_points = 0
valid_coordinate_points = 0
invalid_coordinate_points = 0

coordinate_out_of_bounds = 0
zero_zero_coordinates = 0

continuous_one_second_edges = 0
valid_speed_edges = 0
invalid_speed_edges = 0
zero_distance_edges = 0

overlap_observations = 0
overlap_conflicts = 0

speed_bins = Counter()
class_speed_bins = defaultdict(Counter)

speed_cap_exceed = Counter()
class_speed_cap_exceed = defaultdict(Counter)

max_speed = 0.0
max_speed_class = None

class_valid_edges = Counter()
class_total_edges = Counter()


# ---------------------------------------------------------
# Window audit
#
# gps200:
#   200 GPS points -> 199 pairwise speed values
#
# speed200:
#   200 pairwise speed values -> 201 GPS points
# ---------------------------------------------------------

window_specs = {
    "gps200": {
        "points": 200,
        "edges": 199,
    },
    "speed200": {
        "points": 201,
        "edges": 200,
    },
}

window_counts = defaultdict(Counter)
window_tids = defaultdict(set)

split_window_counts = defaultdict(Counter)
split_window_tids = defaultdict(set)

class_window_counts = defaultdict(Counter)
class_window_tids = defaultdict(set)


# ---------------------------------------------------------
# Process each UID/TID independently
# ---------------------------------------------------------

for tid_key, refs in tid_refs.items():

    uid, tid = tid_key

    if uid not in uid_split:
        raise RuntimeError(
            f"UID missing from split manifest: {uid}"
        )

    our_split = uid_split[uid]

    classes = {
        ref["class"]
        for ref in refs
    }

    if len(classes) != 1:
        raise RuntimeError(
            f"Folder-class change inside UID/TID: {tid_key}"
        )

    folder_class = next(iter(classes))

    segments = []

    # -----------------------------------------------------
    # Read segment data
    # -----------------------------------------------------

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
                    f"Invalid timestamp in {ref['member']}"
                )

            lat = parse_coord(
                row.get("latitude")
            )

            lon = parse_coord(
                row.get("longitude")
            )

            accuracy = str(
                row.get("accuracy", "")
            ).strip()

            altitude = str(
                row.get("altitude", "")
            ).strip()

            values = (
                accuracy,
                str(row.get("latitude", "")).strip(),
                str(row.get("longitude", "")).strip(),
                altitude,
            )

            points[ts] = {
                "lat": lat,
                "lon": lon,
                "values": values,
            }

            normalized_rows.append(
                str(ts)
                + "|"
                + "|".join(values)
            )

        content_hash = hashlib.sha256(
            "\n".join(normalized_rows).encode("utf-8")
        ).hexdigest()

        segments.append({
            "sid": ref["sid"],
            "start": min(points),
            "end": max(points),
            "points": points,
            "hash": content_hash,
        })


    # -----------------------------------------------------
    # Collapse exact same-start duplicates analytically
    # Raw data remains untouched.
    # -----------------------------------------------------

    start_groups = defaultdict(list)

    for segment in segments:
        start_groups[
            segment["start"]
        ].append(segment)

    collapsed = []

    for group in start_groups.values():

        if len(group) == 1:
            collapsed.append(group[0])
            continue

        hashes = {
            item["hash"]
            for item in group
        }

        if len(hashes) != 1:
            raise RuntimeError(
                "Conflicting same-start duplicate detected"
            )

        collapsed.append(group[0])

    collapsed.sort(
        key=lambda x: (
            x["start"],
            x["end"],
            x["sid"],
        )
    )


    # -----------------------------------------------------
    # Merge by timestamp
    # -----------------------------------------------------

    merged = {}
    conflict_timestamps = set()

    for segment in collapsed:

        for ts, point in segment["points"].items():

            if ts not in merged:
                merged[ts] = point
                continue

            overlap_observations += 1

            if (
                merged[ts]["values"]
                != point["values"]
            ):
                overlap_conflicts += 1
                conflict_timestamps.add(ts)


    timestamps = sorted(merged)

    if not timestamps:
        continue


    # -----------------------------------------------------
    # Point coordinate audit
    # -----------------------------------------------------

    for ts in timestamps:

        point = merged[ts]

        unique_points += 1

        lat = point["lat"]
        lon = point["lon"]

        if lat is None or lon is None:
            invalid_coordinate_points += 1
            continue

        if (
            lat < -90
            or lat > 90
            or lon < -180
            or lon > 180
        ):
            coordinate_out_of_bounds += 1
            invalid_coordinate_points += 1
            continue

        valid_coordinate_points += 1

        if lat == 0 and lon == 0:
            zero_zero_coordinates += 1


    # -----------------------------------------------------
    # True one-second continuous runs
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Speed + window audit
    # -----------------------------------------------------

    for run in runs:

        if len(run) < 2:
            continue

        edge_valid = []
        edge_speed = []

        for previous_ts, current_ts in zip(
            run,
            run[1:]
        ):

            continuous_one_second_edges += 1
            class_total_edges[folder_class] += 1

            previous = merged[previous_ts]
            current_point = merged[current_ts]

            lat1 = previous["lat"]
            lon1 = previous["lon"]

            lat2 = current_point["lat"]
            lon2 = current_point["lon"]

            valid = (
                previous_ts not in conflict_timestamps
                and current_ts not in conflict_timestamps
                and lat1 is not None
                and lon1 is not None
                and lat2 is not None
                and lon2 is not None
                and -90 <= lat1 <= 90
                and -180 <= lon1 <= 180
                and -90 <= lat2 <= 90
                and -180 <= lon2 <= 180
            )

            if not valid:

                invalid_speed_edges += 1

                edge_valid.append(0)
                edge_speed.append(None)

                continue

            distance_m = haversine_m(
                lat1,
                lon1,
                lat2,
                lon2,
            )

            # Timestamp interval is exactly 1 second.
            speed_mps = distance_m

            valid_speed_edges += 1
            class_valid_edges[folder_class] += 1

            if distance_m == 0:
                zero_distance_edges += 1

            label = speed_bin(speed_mps)

            speed_bins[label] += 1
            class_speed_bins[
                folder_class
            ][label] += 1

            for cap in DIAGNOSTIC_SPEED_CAPS:

                if speed_mps > cap:
                    speed_cap_exceed[cap] += 1
                    class_speed_cap_exceed[
                        folder_class
                    ][cap] += 1

            if speed_mps > max_speed:
                max_speed = speed_mps
                max_speed_class = folder_class

            edge_valid.append(1)
            edge_speed.append(speed_mps)


        # -------------------------------------------------
        # Prefix sums for efficient window counting
        # -------------------------------------------------

        invalid_prefix = [0]

        for valid in edge_valid:
            invalid_prefix.append(
                invalid_prefix[-1]
                + (0 if valid else 1)
            )

        cap_prefix = {}

        for cap in DIAGNOSTIC_SPEED_CAPS:

            prefix = [0]

            for valid, speed in zip(
                edge_valid,
                edge_speed
            ):

                high = (
                    1
                    if (
                        valid
                        and speed is not None
                        and speed > cap
                    )
                    else 0
                )

                prefix.append(
                    prefix[-1] + high
                )

            cap_prefix[cap] = prefix


        # -------------------------------------------------
        # Evaluate both interpretations
        # -------------------------------------------------

        for spec_name, spec in window_specs.items():

            points_needed = spec["points"]
            edges_needed = spec["edges"]

            if len(run) < points_needed:
                continue

            number_of_windows = (
                len(run)
                - points_needed
                + 1
            )

            window_counts[
                spec_name
            ]["structural"] += number_of_windows

            window_tids[
                (spec_name, "structural")
            ].add(tid_key)

            split_window_counts[
                (our_split, spec_name)
            ]["structural"] += number_of_windows

            split_window_tids[
                (
                    our_split,
                    spec_name,
                    "structural",
                )
            ].add(tid_key)

            class_window_counts[
                (folder_class, spec_name)
            ]["structural"] += number_of_windows

            class_window_tids[
                (
                    folder_class,
                    spec_name,
                    "structural",
                )
            ].add(tid_key)


            for start in range(
                number_of_windows
            ):

                end_edge = (
                    start + edges_needed
                )

                invalid_count = (
                    invalid_prefix[end_edge]
                    - invalid_prefix[start]
                )

                if invalid_count != 0:
                    continue

                window_counts[
                    spec_name
                ]["all_valid_speed"] += 1

                window_tids[
                    (
                        spec_name,
                        "all_valid_speed",
                    )
                ].add(tid_key)

                split_window_counts[
                    (our_split, spec_name)
                ]["all_valid_speed"] += 1

                split_window_tids[
                    (
                        our_split,
                        spec_name,
                        "all_valid_speed",
                    )
                ].add(tid_key)

                class_window_counts[
                    (folder_class, spec_name)
                ]["all_valid_speed"] += 1

                class_window_tids[
                    (
                        folder_class,
                        spec_name,
                        "all_valid_speed",
                    )
                ].add(tid_key)


                for cap in DIAGNOSTIC_SPEED_CAPS:

                    high_count = (
                        cap_prefix[cap][end_edge]
                        - cap_prefix[cap][start]
                    )

                    if high_count != 0:
                        continue

                    cap_name = f"le_{cap}_mps"

                    window_counts[
                        spec_name
                    ][cap_name] += 1

                    window_tids[
                        (
                            spec_name,
                            cap_name,
                        )
                    ].add(tid_key)

                    split_window_counts[
                        (our_split, spec_name)
                    ][cap_name] += 1

                    split_window_tids[
                        (
                            our_split,
                            spec_name,
                            cap_name,
                        )
                    ].add(tid_key)

                    class_window_counts[
                        (folder_class, spec_name)
                    ][cap_name] += 1

                    class_window_tids[
                        (
                            folder_class,
                            spec_name,
                            cap_name,
                        )
                    ].add(tid_key)


# ---------------------------------------------------------
# Close ZIPs
# ---------------------------------------------------------

for z in zip_handles.values():
    z.close()


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

print("=== SPEED FEATURE BASIC AUDIT ===")
print("unique timestamp points       =", unique_points)
print("valid coordinate points       =", valid_coordinate_points)
print("invalid coordinate points     =", invalid_coordinate_points)
print("coordinate out of bounds      =", coordinate_out_of_bounds)
print("(0,0) coordinate points       =", zero_zero_coordinates)

print()
print("continuous 1-second edges     =", continuous_one_second_edges)
print("valid derived-speed edges     =", valid_speed_edges)
print("invalid derived-speed edges   =", invalid_speed_edges)
print("zero-distance speed edges     =", zero_distance_edges)

print()
print("point overlap observations    =", overlap_observations)
print("point overlap conflicts       =", overlap_conflicts)

print()
print("maximum observed speed m/s    =", round(max_speed, 6))
print("class of maximum speed        =", max_speed_class)


print()
print("=== SPEED DISTRIBUTION (VALID 1-SECOND EDGES) ===")

for label in [
    "0",
    "(0,1]",
    "(1,3]",
    "(3,7]",
    "(7,15]",
    "(15,30]",
    "(30,50]",
    "(50,80]",
    "(80,100]",
    ">100",
]:
    print(label, "m/s =", speed_bins[label])


print()
print("=== DIAGNOSTIC HIGH-SPEED COUNTS ===")

for cap in DIAGNOSTIC_SPEED_CAPS:

    count = speed_cap_exceed[cap]

    pct = (
        count / valid_speed_edges * 100
        if valid_speed_edges
        else 0
    )

    print(
        f">{cap} m/s =",
        count,
        f"({pct:.6f}%)",
    )


print()
print("=== VALID SPEED EDGES BY FOLDER CLASS ===")

for cls in [
    "WALK",
    "BIKE",
    "CAR",
    "BUS",
    "SUBWAY",
    "ETC",
]:

    total = class_total_edges[cls]
    valid = class_valid_edges[cls]

    pct = (
        valid / total * 100
        if total
        else 0
    )

    print()
    print(cls)
    print("continuous edges =", total)
    print("valid speeds     =", valid)
    print("valid speed %    =", round(pct, 4))

    for cap in DIAGNOSTIC_SPEED_CAPS:
        print(
            f">{cap} m/s         =",
            class_speed_cap_exceed[
                cls
            ][cap]
        )


for spec_name in [
    "gps200",
    "speed200",
]:

    print()
    print(
        "=== WINDOW AUDIT:",
        spec_name.upper(),
        "===",
    )

    if spec_name == "gps200":
        print(
            "interpretation = 200 GPS points -> 199 pairwise speeds"
        )
    else:
        print(
            "interpretation = 200 pairwise speeds -> 201 GPS points"
        )

    print(
        "structural windows =",
        window_counts[
            spec_name
        ]["structural"],
    )

    print(
        "structural TIDs    =",
        len(
            window_tids[
                (
                    spec_name,
                    "structural",
                )
            ]
        ),
    )

    print(
        "all-valid-speed windows =",
        window_counts[
            spec_name
        ]["all_valid_speed"],
    )

    print(
        "all-valid-speed TIDs    =",
        len(
            window_tids[
                (
                    spec_name,
                    "all_valid_speed",
                )
            ]
        ),
    )

    for cap in DIAGNOSTIC_SPEED_CAPS:

        cap_name = f"le_{cap}_mps"

        print(
            f"all-valid and <= {cap} m/s windows =",
            window_counts[
                spec_name
            ][cap_name],
        )

        print(
            f"all-valid and <= {cap} m/s TIDs    =",
            len(
                window_tids[
                    (
                        spec_name,
                        cap_name,
                    )
                ]
            ),
        )


print()
print("=== SPEED200 BY UID SPLIT ===")

for split in sorted(
    set(uid_split.values())
):

    key = (split, "speed200")

    print()
    print(split.upper())

    print(
        "structural windows =",
        split_window_counts[
            key
        ]["structural"],
    )

    print(
        "structural TIDs    =",
        len(
            split_window_tids[
                (
                    split,
                    "speed200",
                    "structural",
                )
            ]
        ),
    )

    print(
        "all-valid windows  =",
        split_window_counts[
            key
        ]["all_valid_speed"],
    )

    print(
        "all-valid TIDs     =",
        len(
            split_window_tids[
                (
                    split,
                    "speed200",
                    "all_valid_speed",
                )
            ]
        ),
    )


print()
print("=== SPEED200 BY FOLDER CLASS ===")

for cls in [
    "WALK",
    "BIKE",
    "CAR",
    "BUS",
    "SUBWAY",
    "ETC",
]:

    key = (cls, "speed200")

    print()
    print(cls)

    print(
        "structural windows =",
        class_window_counts[
            key
        ]["structural"],
    )

    print(
        "structural TIDs    =",
        len(
            class_window_tids[
                (
                    cls,
                    "speed200",
                    "structural",
                )
            ]
        ),
    )

    print(
        "all-valid windows  =",
        class_window_counts[
            key
        ]["all_valid_speed"],
    )

    print(
        "all-valid TIDs     =",
        len(
            class_window_tids[
                (
                    cls,
                    "speed200",
                    "all_valid_speed",
                )
            ]
        ),
    )

    for cap in [
        50,
        80,
        100,
    ]:

        cap_name = f"le_{cap}_mps"

        print(
            f"<= {cap} m/s TIDs      =",
            len(
                class_window_tids[
                    (
                        cls,
                        "speed200",
                        cap_name,
                    )
                ]
            ),
        )


print()
print("NOTE:")
print(
    "Speeds are Haversine distance / exactly 1-second timestamp interval."
)
print(
    "No coordinate imputation, deletion, clipping, smoothing, or speed filtering was applied."
)
print(
    "30/50/80/100 m/s are diagnostic thresholds only, not preprocessing decisions."
)
print(
    "GPS200 and SPEED200 are both reported because the final SpeedTransformer input convention has not yet been fixed."
)

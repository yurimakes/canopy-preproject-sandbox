#!/usr/bin/env python3
"""Build the balanced STRICT-EXPANDED-645 control dataset."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import random
import re
from zipfile import ZipFile


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
OUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "speedtransformer_strict_expanded_v1"
    / "speed200_strict_expanded_v1.csv"
)

NAME_RE = re.compile(
    r"^TMC-GPS-([^-]+)-([^-]+)-([^-]+)-Dataset\.csv$",
    re.I,
)
CLASSES = ("WALK", "BIKE", "CAR", "BUS", "SUBWAY")
LABEL_MAP = {class_name: label for label, class_name in enumerate(CLASSES)}
SEED = 316
POINTS_NEEDED = 201
SPEEDS_NEEDED = 200
BASE_PER_CLASS = 100
ADD_PER_CLASS = 29


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def parse_float(value: object) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def valid_coord(lat: float | None, lon: float | None) -> bool:
    return (
        lat is not None
        and lon is not None
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    value = min(1.0, max(0.0, value))
    return 2 * radius * math.asin(math.sqrt(value))


def sample_id_for(uid: str, tid: str) -> str:
    return hashlib.sha256(f"{uid}|{tid}".encode("utf-8")).hexdigest()[:16]


def load_uid_split() -> dict[str, str]:
    uid_split: dict[str, str] = {}
    with SPLIT_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {name.lower(): name for name in reader.fieldnames or ()}
        for row in reader:
            uid_split[row[fields["uid"]]] = row[fields["split"]]
    return uid_split


def load_strict_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    if not STRICT_CSV.is_file():
        raise FileNotFoundError("Strict input dataset is missing")

    rows: list[dict[str, str]] = []
    samples: dict[str, dict[str, object]] = {}
    required = {
        "sample_id",
        "split",
        "class_name",
        "model_label",
        "step",
        "speed_kmh",
    }
    with STRICT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Strict CSV missing columns: {sorted(missing)}")

        for row in reader:
            if row["split"] not in ("train", "validation"):
                continue
            clean = {name: row[name] for name in required}
            rows.append(clean)
            sample_id = row["sample_id"]
            metadata = (row["split"], int(row["model_label"]), row["class_name"])
            sample = samples.setdefault(
                sample_id,
                {"metadata": metadata, "speeds": {}},
            )
            if sample["metadata"] != metadata:
                raise RuntimeError("Strict sample metadata conflict")
            step = int(row["step"])
            speeds = sample["speeds"]
            if step in speeds:
                raise RuntimeError("Duplicate strict sample step")
            speeds[step] = row["speed_kmh"]

    for sample in samples.values():
        if sorted(sample["speeds"]) != list(range(SPEEDS_NEEDED)):
            raise RuntimeError("Strict sample does not contain steps 0..199")

    counts = Counter(sample["metadata"][:2] for sample in samples.values())
    for class_name, label in LABEL_MAP.items():
        assert counts[("train", label)] == BASE_PER_CLASS
        assert counts[("validation", label)] == 30
        assert all(
            sample["metadata"][2] == class_name
            for sample in samples.values()
            if sample["metadata"][1] == label
        )
    assert len(samples) == 650
    return rows, samples


def index_train_refs(uid_split: dict[str, str]) -> tuple[dict[tuple[str, str], list[dict[str, object]]], dict[Path, ZipFile]]:
    tid_refs: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    zip_handles: dict[Path, ZipFile] = {}

    for path in sorted(PACKAGES.glob("*_GPS_*.zip")):
        if not (path.name.startswith("TS_") or path.name.startswith("VS_")):
            continue
        class_name = path.stem.split("_")[-1].upper()
        if class_name not in LABEL_MAP:
            continue
        archive = ZipFile(path)
        zip_handles[path] = archive
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            match = NAME_RE.match(PurePosixPath(member).name)
            if not match:
                continue
            uid, tid, _ = match.groups()
            if uid_split.get(uid) != "train":
                continue
            tid_refs[(uid, tid)].append(
                {"path": path, "member": member, "class_name": class_name}
            )
    return tid_refs, zip_handles


def first_strict_sample(
    refs: list[dict[str, object]],
    zip_handles: dict[Path, ZipFile],
) -> list[float] | None:
    points: dict[int, tuple[float | None, float | None]] = {}
    conflicts: set[int] = set()
    for ref in refs:
        archive = zip_handles[ref["path"]]
        reader = csv.DictReader(io.StringIO(decode(archive.read(ref["member"]))))
        for row in reader:
            try:
                timestamp = int(float(row["timestamp"]))
            except (TypeError, ValueError):
                continue
            value = (parse_float(row.get("latitude")), parse_float(row.get("longitude")))
            if timestamp in points and points[timestamp] != value:
                conflicts.add(timestamp)
            else:
                points[timestamp] = value

    timestamps = sorted(points)
    if len(timestamps) < POINTS_NEEDED:
        return None

    runs: list[list[int]] = []
    current = [timestamps[0]]
    for previous, timestamp in zip(timestamps, timestamps[1:]):
        if timestamp - previous == 1000:
            current.append(timestamp)
        else:
            runs.append(current)
            current = [timestamp]
    runs.append(current)

    for run in runs:
        if len(run) < POINTS_NEEDED:
            continue
        for start in range(len(run) - POINTS_NEEDED + 1):
            selected = run[start : start + POINTS_NEEDED]
            if any(
                timestamp in conflicts or not valid_coord(*points[timestamp])
                for timestamp in selected
            ):
                continue
            speeds: list[float] = []
            for first, second in zip(selected, selected[1:]):
                dt_seconds = (second - first) / 1000.0
                assert dt_seconds == 1.0
                lat1, lon1 = points[first]
                lat2, lon2 = points[second]
                speed = haversine_m(lat1, lon1, lat2, lon2) / dt_seconds * 3.6
                if not math.isfinite(speed):
                    raise RuntimeError("Non-finite strict speed")
                speeds.append(speed)
            assert len(speeds) == SPEEDS_NEEDED
            return speeds
    return None


def main() -> None:
    strict_rows, strict_samples = load_strict_rows()
    uid_split = load_uid_split()
    tid_refs, zip_handles = index_train_refs(uid_split)
    print(f"INDEXED_TRAIN_UID_TID={len(tid_refs)}")

    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    try:
        for (uid, tid), refs in tid_refs.items():
            classes = {ref["class_name"] for ref in refs}
            if len(classes) != 1:
                raise RuntimeError("Multiple classes inside one Train UID/TID")
            class_name = next(iter(classes))
            speeds = first_strict_sample(refs, zip_handles)
            if speeds is None:
                continue
            candidates[class_name].append(
                {
                    "sample_id": sample_id_for(uid, tid),
                    "class_name": class_name,
                    "model_label": LABEL_MAP[class_name],
                    "speeds": speeds,
                    "all_edges_exact_1s": True,
                }
            )
    finally:
        for archive in zip_handles.values():
            archive.close()

    rng = random.Random(SEED)
    additions: list[dict[str, object]] = []
    print("=== STRICT ELIGIBLE / UNUSED / SELECTED ===")
    for class_name in CLASSES:
        group = candidates[class_name]
        group.sort(key=lambda sample: sample["sample_id"])
        rng.shuffle(group)
        eligible = len(group)
        if eligible < BASE_PER_CLASS + ADD_PER_CLASS:
            raise AssertionError(
                f"{class_name} has fewer than {ADD_PER_CLASS} unused strict candidates"
            )

        reproduced_base = group[:BASE_PER_CLASS]
        existing_ids = {
            sample_id
            for sample_id, sample in strict_samples.items()
            if sample["metadata"][:2] == ("train", LABEL_MAP[class_name])
        }
        reproduced_ids = {sample["sample_id"] for sample in reproduced_base}
        assert existing_ids == reproduced_ids

        for sample in reproduced_base:
            stored = strict_samples[sample["sample_id"]]["speeds"]
            regenerated = {
                step: f"{speed:.10f}" for step, speed in enumerate(sample["speeds"])
            }
            assert stored == regenerated

        selected = group[BASE_PER_CLASS : BASE_PER_CLASS + ADD_PER_CLASS]
        assert len(selected) == ADD_PER_CLASS
        assert all(sample["all_edges_exact_1s"] for sample in selected)
        assert not ({sample["sample_id"] for sample in selected} & set(strict_samples))
        additions.extend(selected)
        print(
            f"{class_name} eligible={eligible} unused={eligible - BASE_PER_CLASS} "
            f"selected={len(selected)}"
        )

    output_rows: list[dict[str, object]] = []
    for row in strict_rows:
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "class_name": row["class_name"],
                "model_label": row["model_label"],
                "step": row["step"],
                "speed_kmh": row["speed_kmh"],
                "source_policy": "strict_1s",
                "two_sec_edges": 0,
                "wall_span_sec": "200.0",
            }
        )
    for sample in additions:
        for step, speed in enumerate(sample["speeds"]):
            output_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "split": "train",
                    "class_name": sample["class_name"],
                    "model_label": sample["model_label"],
                    "step": step,
                    "speed_kmh": f"{speed:.10f}",
                    "source_policy": "strict_1s",
                    "two_sec_edges": 0,
                    "wall_span_sec": "200.0",
                }
            )

    output_samples: dict[str, dict[str, object]] = {}
    for row in output_rows:
        sample_id = str(row["sample_id"])
        metadata = (
            str(row["split"]),
            int(row["model_label"]),
            str(row["class_name"]),
            str(row["source_policy"]),
        )
        sample = output_samples.setdefault(
            sample_id,
            {"metadata": metadata, "values": {}},
        )
        assert sample["metadata"] == metadata
        step = int(row["step"])
        assert step not in sample["values"]
        sample["values"][step] = str(row["speed_kmh"])

    assert len(output_samples) == 795
    assert len(output_rows) == 159000
    assert all(
        sorted(sample["values"]) == list(range(SPEEDS_NEEDED))
        for sample in output_samples.values()
    )
    final_counts = Counter(sample["metadata"][:2] for sample in output_samples.values())
    for label in range(len(CLASSES)):
        assert final_counts[("train", label)] == 129
        assert final_counts[("validation", label)] == 30

    strict_train_ids = {
        sample_id
        for sample_id, sample in strict_samples.items()
        if sample["metadata"][0] == "train"
    }
    strict_val_ids = {
        sample_id
        for sample_id, sample in strict_samples.items()
        if sample["metadata"][0] == "validation"
    }
    output_train_ids = {
        sample_id
        for sample_id, sample in output_samples.items()
        if sample["metadata"][0] == "train"
    }
    output_val_ids = {
        sample_id
        for sample_id, sample in output_samples.items()
        if sample["metadata"][0] == "validation"
    }
    assert strict_train_ids.issubset(output_train_ids)
    assert len(output_train_ids - strict_train_ids) == 145
    assert strict_val_ids == output_val_ids
    assert output_train_ids.isdisjoint(output_val_ids)
    for sample_id in strict_val_ids:
        strict_sample = strict_samples[sample_id]
        output_sample = output_samples[sample_id]
        assert strict_sample["metadata"] == output_sample["metadata"][:3]
        assert strict_sample["speeds"] == output_sample["values"]
    assert all(sample["metadata"][3] == "strict_1s" for sample in output_samples.values())
    assert all(sample["metadata"][0] != "internal_test" for sample in output_samples.values())
    assert all(sample["all_edges_exact_1s"] for sample in additions)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "class_name",
        "model_label",
        "step",
        "speed_kmh",
        "source_policy",
        "two_sec_edges",
        "wall_span_sec",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    digest = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest().upper()
    print("=== FINAL CHECK ===")
    print("strict_base_train_samples=500")
    print("additional_strict_train_samples=145")
    print("strict_expanded_train_samples=645")
    print("validation_samples=150")
    print("validation_sample_ids_identical=True")
    print("validation_features_labels_identical=True")
    print("samples=795")
    print("rows=159000")
    print("train_validation_sample_id_overlap=0")
    print("internal_test_samples=0")
    print("mild_release_samples=0")
    print("all_additional_edges_exact_1s=True")
    print(f"SHA256={digest}")
    print(f"OUTPUT={OUT_CSV.relative_to(ROOT).as_posix()}")
    print("STRICT_EXPANDED_DATASET_OK")


if __name__ == "__main__":
    main()

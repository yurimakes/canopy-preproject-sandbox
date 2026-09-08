#!/usr/bin/env python3
"""Build equal-size LARGE strict/mild control datasets from Train data only."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import io
import math
from pathlib import Path
from random import Random
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STRICT_BUILDER_PATH = ROOT / "scripts" / "18_make_strict_expanded_dataset.py"
STRICT_EXPANDED_CSV = (
    ROOT
    / "data"
    / "processed"
    / "speedtransformer_strict_expanded_v1"
    / "speed200_strict_expanded_v1.csv"
)
MILD_EXPANDED_CSV = (
    ROOT
    / "data"
    / "processed"
    / "speedtransformer_mild_release_v1"
    / "speed200_mild_release_v1.csv"
)
OUT_DIR = ROOT / "data" / "processed" / "speedtransformer_large_control_v1"
STRICT_OUT = OUT_DIR / "speed200_strict_large_v1.csv"
MILD_OUT = OUT_DIR / "speed200_mild_large_v1.csv"

SEED = 316
COMMON_STRICT_PER_CLASS = 271
DIFFERENTIAL_PER_CLASS = 29
TRAIN_PER_CLASS = COMMON_STRICT_PER_CLASS + DIFFERENTIAL_PER_CLASS
VALIDATION_PER_CLASS = 30
MAX_2S_EDGES = 5
MAX_WALL_SPAN_SEC = 205.0
FIELDNAMES = (
    "sample_id",
    "split",
    "class_name",
    "model_label",
    "step",
    "speed_kmh",
    "source_policy",
    "two_sec_edges",
    "wall_span_sec",
)


def load_strict_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "large_control_strict_builder",
        STRICT_BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load script 18")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection_key(uid: str, tid: str) -> str:
    return hashlib.sha256(f"{SEED}|{uid}|{tid}".encode("utf-8")).hexdigest()


def load_processed_samples(path: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDNAMES) - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"] not in ("train", "validation"):
                continue
            sample_id = row["sample_id"]
            metadata = (
                row["split"],
                row["class_name"],
                int(row["model_label"]),
                row["source_policy"],
                int(row["two_sec_edges"]),
                float(row["wall_span_sec"]),
            )
            sample = samples.setdefault(
                sample_id,
                {"metadata": metadata, "speeds": {}},
            )
            assert sample["metadata"] == metadata
            step = int(row["step"])
            assert step not in sample["speeds"]
            sample["speeds"][step] = float(row["speed_kmh"])
    assert all(sorted(sample["speeds"]) == list(range(200)) for sample in samples.values())
    return samples


def merged_points(
    refs: list[dict[str, object]],
    zip_handles: dict[Path, Any],
    base: Any,
) -> tuple[dict[int, tuple[float | None, float | None]], set[int]]:
    points: dict[int, tuple[float | None, float | None]] = {}
    conflicts: set[int] = set()
    for ref in refs:
        archive = zip_handles[ref["path"]]
        reader = csv.DictReader(io.StringIO(base.decode(archive.read(ref["member"]))))
        for row in reader:
            try:
                timestamp = int(float(row["timestamp"]))
            except (TypeError, ValueError):
                continue
            value = (base.parse_float(row.get("latitude")), base.parse_float(row.get("longitude")))
            if timestamp in points and points[timestamp] != value:
                conflicts.add(timestamp)
            else:
                points[timestamp] = value
    return points, conflicts


def first_window(timestamps: list[int], allowed_gaps: tuple[int, ...]) -> list[int] | None:
    if len(timestamps) < 201:
        return None
    run_start = 0
    for right in range(1, len(timestamps)):
        if timestamps[right] - timestamps[right - 1] not in allowed_gaps:
            run_start = right
        if right - run_start + 1 >= 201:
            return timestamps[run_start : run_start + 201]
    return None


def mild_candidate(
    uid: str,
    tid: str,
    refs: list[dict[str, object]],
    zip_handles: dict[Path, Any],
    base: Any,
) -> dict[str, Any] | None:
    points, conflicts = merged_points(refs, zip_handles, base)
    timestamps = [
        timestamp
        for timestamp in sorted(points)
        if timestamp not in conflicts and base.valid_coord(*points[timestamp])
    ]
    if first_window(timestamps, (1000,)) is not None:
        return None
    window = first_window(timestamps, (1000, 2000))
    if window is None:
        return None
    gaps = [second - first for first, second in zip(window, window[1:])]
    two_sec_edges = sum(gap == 2000 for gap in gaps)
    wall_span_sec = (window[-1] - window[0]) / 1000.0
    if two_sec_edges > MAX_2S_EDGES or wall_span_sec > MAX_WALL_SPAN_SEC:
        return None
    speeds = []
    for first, second in zip(window, window[1:]):
        lat1, lon1 = points[first]
        lat2, lon2 = points[second]
        speed = base.haversine_m(lat1, lon1, lat2, lon2) / ((second - first) / 1000.0) * 3.6
        if not math.isfinite(speed):
            raise RuntimeError("Non-finite mild speed")
        speeds.append(speed)
    assert len(speeds) == 200
    return {
        "sample_id": base.sample_id_for(uid, tid),
        "selection_key": selection_key(uid, tid),
        "speeds": speeds,
        "two_sec_edges": two_sec_edges,
        "wall_span_sec": wall_span_sec,
    }


def make_rows(samples: list[dict[str, Any]], split: str, policy: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        for step, speed in enumerate(sample["speeds"]):
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "split": split,
                    "class_name": sample["class_name"],
                    "model_label": sample["model_label"],
                    "step": step,
                    "speed_kmh": f"{speed:.10f}" if policy == "strict_1s" else repr(speed),
                    "source_policy": policy,
                    "two_sec_edges": sample.get("two_sec_edges", 0),
                    "wall_span_sec": sample.get("wall_span_sec", 200.0),
                }
            )
    return rows


def validation_rows(samples: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        split, class_name, label, policy, two_sec_edges, wall_span_sec = sample["metadata"]
        if split != "validation":
            continue
        assert policy == "strict_1s"
        for step in range(200):
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "class_name": class_name,
                    "model_label": label,
                    "step": step,
                    "speed_kmh": repr(sample["speeds"][step]),
                    "source_policy": policy,
                    "two_sec_edges": two_sec_edges,
                    "wall_span_sec": wall_span_sec,
                }
            )
    return rows


def audit_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        metadata = (
            str(row["split"]),
            str(row["class_name"]),
            int(row["model_label"]),
            str(row["source_policy"]),
        )
        sample = samples.setdefault(sample_id, {"metadata": metadata, "steps": {}})
        assert sample["metadata"] == metadata
        step = int(row["step"])
        assert step not in sample["steps"]
        sample["steps"][step] = float(row["speed_kmh"])
    assert all(sorted(sample["steps"]) == list(range(200)) for sample in samples.values())
    assert len(rows) == 330000
    assert len(samples) == 1650
    counts = Counter((sample["metadata"][0], sample["metadata"][2]) for sample in samples.values())
    for label in range(5):
        assert counts[("train", label)] == TRAIN_PER_CLASS
        assert counts[("validation", label)] == VALIDATION_PER_CLASS
    assert all(sample["metadata"][0] != "internal_test" for sample in samples.values())
    return samples


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    base = load_strict_builder()
    strict_expanded = load_processed_samples(STRICT_EXPANDED_CSV)
    mild_expanded = load_processed_samples(MILD_EXPANDED_CSV)
    strict_expanded_ids = {
        sample_id for sample_id, sample in strict_expanded.items() if sample["metadata"][0] == "train"
    }
    existing_mild = {
        sample_id: sample
        for sample_id, sample in mild_expanded.items()
        if sample["metadata"][0] == "train" and sample["metadata"][3] == "mild_2s_le5_span205"
    }

    uid_split = base.load_uid_split()
    tid_refs, zip_handles = base.index_train_refs(uid_split)
    strict_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mild_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for (uid, tid), refs in tid_refs.items():
            classes = {str(ref["class_name"]) for ref in refs}
            if len(classes) != 1:
                raise RuntimeError("Multiple classes inside one Train UID/TID")
            class_name = next(iter(classes))
            strict_speeds = base.first_strict_sample(refs, zip_handles)
            if strict_speeds is not None:
                strict_candidates[class_name].append(
                    {
                        "sample_id": base.sample_id_for(uid, tid),
                        "class_name": class_name,
                        "model_label": base.LABEL_MAP[class_name],
                        "speeds": strict_speeds,
                    }
                )
            mild = mild_candidate(uid, tid, refs, zip_handles, base)
            if mild is not None:
                mild.update(
                    {
                        "class_name": class_name,
                        "model_label": base.LABEL_MAP[class_name],
                    }
                )
                mild_candidates[class_name].append(mild)
    finally:
        for archive in zip_handles.values():
            archive.close()

    print("=== FEASIBILITY ===")
    for class_name in base.CLASSES:
        print(
            f"{class_name} strict_candidates={len(strict_candidates[class_name])} "
            f"mild_candidates={len(mild_candidates[class_name])}"
        )
        assert len(strict_candidates[class_name]) >= TRAIN_PER_CLASS
        assert len(mild_candidates[class_name]) >= DIFFERENTIAL_PER_CLASS

    rng = Random(SEED)
    common: list[dict[str, Any]] = []
    strict_only: list[dict[str, Any]] = []
    mild_only: list[dict[str, Any]] = []
    for class_name in base.CLASSES:
        strict_group = sorted(strict_candidates[class_name], key=lambda sample: sample["sample_id"])
        rng.shuffle(strict_group)
        class_common = strict_group[:COMMON_STRICT_PER_CLASS]
        class_strict_only = strict_group[
            COMMON_STRICT_PER_CLASS : COMMON_STRICT_PER_CLASS + DIFFERENTIAL_PER_CLASS
        ]
        expanded_class_ids = {
            sample_id
            for sample_id in strict_expanded_ids
            if strict_expanded[sample_id]["metadata"][1] == class_name
        }
        assert len(expanded_class_ids) == 129
        assert expanded_class_ids.issubset({sample["sample_id"] for sample in class_common})

        mild_group = sorted(mild_candidates[class_name], key=lambda sample: sample["selection_key"])
        class_mild_only = mild_group[:DIFFERENTIAL_PER_CLASS]
        existing_class_mild_ids = {
            sample_id
            for sample_id, sample in existing_mild.items()
            if sample["metadata"][1] == class_name
        }
        assert existing_class_mild_ids == {sample["sample_id"] for sample in class_mild_only}
        for sample in class_mild_only:
            stored = existing_mild[sample["sample_id"]]
            assert stored["metadata"][4] == sample["two_sec_edges"]
            assert stored["metadata"][5] == sample["wall_span_sec"]
            assert all(stored["speeds"][step] == sample["speeds"][step] for step in range(200))
        common.extend(class_common)
        strict_only.extend(class_strict_only)
        mild_only.extend(class_mild_only)

    assert len(common) == 1355
    assert len(strict_only) == 145
    assert len(mild_only) == 145
    common_ids = {sample["sample_id"] for sample in common}
    strict_only_ids = {sample["sample_id"] for sample in strict_only}
    mild_only_ids = {sample["sample_id"] for sample in mild_only}
    assert common_ids.isdisjoint(strict_only_ids | mild_only_ids)
    assert strict_only_ids.isdisjoint(mild_only_ids)

    val_rows = validation_rows(strict_expanded)
    strict_rows = make_rows(common + strict_only, "train", "strict_1s") + val_rows
    mild_rows = (
        make_rows(common, "train", "strict_1s")
        + make_rows(mild_only, "train", "mild_2s_le5_span205")
        + val_rows
    )
    strict_samples = audit_rows(strict_rows)
    mild_samples = audit_rows(mild_rows)
    strict_val = {
        sample_id: sample for sample_id, sample in strict_samples.items() if sample["metadata"][0] == "validation"
    }
    mild_val = {
        sample_id: sample for sample_id, sample in mild_samples.items() if sample["metadata"][0] == "validation"
    }
    assert strict_val == mild_val
    strict_train_ids = {
        sample_id for sample_id, sample in strict_samples.items() if sample["metadata"][0] == "train"
    }
    mild_train_ids = {
        sample_id for sample_id, sample in mild_samples.items() if sample["metadata"][0] == "train"
    }
    assert strict_train_ids & mild_train_ids == common_ids
    assert strict_train_ids - mild_train_ids == strict_only_ids
    assert mild_train_ids - strict_train_ids == mild_only_ids
    assert strict_train_ids.isdisjoint(strict_val)
    assert mild_train_ids.isdisjoint(mild_val)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strict_hash = write_csv(STRICT_OUT, strict_rows)
    mild_hash = write_csv(MILD_OUT, mild_rows)
    print("=== LARGE CONTROL ===")
    print("selection_seed=316")
    print("common_strict_per_class=271")
    print("strict_only_per_class=29")
    print("mild_only_per_class=29")
    print("strict_large_train_per_class=300")
    print("mild_large_train_per_class=300")
    print("validation_per_class=30")
    print("validation_identical=True")
    print("train_validation_overlap=0")
    print("internal_test_samples=0")
    print("rows_per_dataset=330000")
    print(f"STRICT_LARGE_SHA256={strict_hash}")
    print(f"MILD_LARGE_SHA256={mild_hash}")
    print(f"STRICT_OUTPUT={STRICT_OUT.relative_to(ROOT).as_posix()}")
    print(f"MILD_OUTPUT={MILD_OUT.relative_to(ROOT).as_posix()}")
    print("LARGE_CONTROL_DATASETS_OK")


if __name__ == "__main__":
    main()

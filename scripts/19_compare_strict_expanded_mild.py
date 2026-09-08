#!/usr/bin/env python3
"""Run the STRICT_BASE / STRICT_EXPANDED / MILD_EXPANDED comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = ROOT / "scripts" / "17_compare_strict_vs_mild_release.py"


def load_base_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "strict_mild_base_runner",
        BASE_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not construct the script 17 module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(base: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-base-csv",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "speedtransformer_smoke_v1"
            / "speed200_smoke_v1.csv"
        ),
    )
    parser.add_argument(
        "--strict-expanded-csv",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "speedtransformer_strict_expanded_v1"
            / "speed200_strict_expanded_v1.csv"
        ),
    )
    parser.add_argument(
        "--mild-expanded-csv",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "speedtransformer_mild_release_v1"
            / "speed200_mild_release_v1.csv"
        ),
    )
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=ROOT.parent / "SpeedTransformer-author-reference",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def policy_counts(path: Path) -> Counter[str]:
    sample_policies: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "source_policy" not in (reader.fieldnames or ()):
            return Counter()
        for row in reader:
            if row["split"] not in ("train", "validation"):
                continue
            sample_id = row["sample_id"]
            policy = row["source_policy"]
            if sample_id in sample_policies:
                assert sample_policies[sample_id] == policy
            else:
                sample_policies[sample_id] = policy
    return Counter(sample_policies.values())


def prepare_data(
    base: Any,
    strict_base_samples: dict[str, dict[str, Any]],
    strict_expanded_samples: dict[str, dict[str, Any]],
    mild_expanded_samples: dict[str, dict[str, Any]],
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    data = {
        "STRICT_BASE_TRAIN": base.split_arrays(strict_base_samples, "train"),
        "STRICT_BASE_VAL": base.split_arrays(strict_base_samples, "validation"),
        "STRICT_EXPANDED_TRAIN": base.split_arrays(
            strict_expanded_samples, "train"
        ),
        "STRICT_EXPANDED_VAL": base.split_arrays(
            strict_expanded_samples, "validation"
        ),
        "MILD_EXPANDED_TRAIN": base.split_arrays(mild_expanded_samples, "train"),
        "MILD_EXPANDED_VAL": base.split_arrays(
            mild_expanded_samples, "validation"
        ),
    }

    assert len(data["STRICT_BASE_TRAIN"][0]) == 500
    assert len(data["STRICT_EXPANDED_TRAIN"][0]) == 645
    assert len(data["MILD_EXPANDED_TRAIN"][0]) == 645
    assert all(len(data[name][0]) == 150 for name in (
        "STRICT_BASE_VAL",
        "STRICT_EXPANDED_VAL",
        "MILD_EXPANDED_VAL",
    ))
    base.assert_class_counts(data["STRICT_BASE_TRAIN"][2], 100, "STRICT_BASE Train")
    base.assert_class_counts(
        data["STRICT_EXPANDED_TRAIN"][2], 129, "STRICT_EXPANDED Train"
    )
    base.assert_class_counts(
        data["MILD_EXPANDED_TRAIN"][2], 129, "MILD_EXPANDED Train"
    )
    for name in ("STRICT_BASE_VAL", "STRICT_EXPANDED_VAL", "MILD_EXPANDED_VAL"):
        base.assert_class_counts(data[name][2], 30, f"{name} Validation")

    base_train_ids = set(data["STRICT_BASE_TRAIN"][0])
    strict_expanded_train_ids = set(data["STRICT_EXPANDED_TRAIN"][0])
    mild_expanded_train_ids = set(data["MILD_EXPANDED_TRAIN"][0])
    assert base_train_ids.issubset(strict_expanded_train_ids)
    assert base_train_ids.issubset(mild_expanded_train_ids)
    assert len(strict_expanded_train_ids - base_train_ids) == 145
    assert len(mild_expanded_train_ids - base_train_ids) == 145

    validation_names = (
        "STRICT_BASE_VAL",
        "STRICT_EXPANDED_VAL",
        "MILD_EXPANDED_VAL",
    )
    first_ids, first_features, first_labels = data[validation_names[0]]
    for name in validation_names[1:]:
        ids, features, labels = data[name]
        assert first_ids == ids
        assert np.array_equal(first_features, features)
        assert np.array_equal(first_labels, labels)

    assert base_train_ids.isdisjoint(set(first_ids))
    assert strict_expanded_train_ids.isdisjoint(set(first_ids))
    assert mild_expanded_train_ids.isdisjoint(set(first_ids))
    print("THREE_ARM_VALIDATION_IDENTICAL=True")
    print("STRICT_BASE_TRAIN_COUNTS=100,100,100,100,100")
    print("STRICT_EXPANDED_TRAIN_COUNTS=129,129,129,129,129")
    print("MILD_EXPANDED_TRAIN_COUNTS=129,129,129,129,129")
    print("VALIDATION_COUNTS=30,30,30,30,30")
    print("INTERNAL_TEST_USED=False")
    return data


def arrays(
    data: dict[str, tuple[list[str], np.ndarray, np.ndarray]],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    return data[name][1], data[name][2]


def print_delta(
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
    classes: tuple[str, ...],
) -> None:
    print(f"=== {left_name} - {right_name} (SAME VALIDATION 150) ===")
    print(f"accuracy_delta={left['accuracy'] - right['accuracy']:.6f}")
    print(f"macro_f1_delta={left['macro_f1'] - right['macro_f1']:.6f}")
    print("class,recall_delta,f1_delta")
    for index, class_name in enumerate(classes):
        print(
            f"{class_name},{left['recall'][index] - right['recall'][index]:.6f},"
            f"{left['f1'][index] - right['f1'][index]:.6f}"
        )


def main() -> None:
    base = load_base_runner()
    args = parse_args(base)
    print(f"PYTHON_VERSION={sys.version.split()[0]}")
    print(f"TORCH_VERSION={torch.__version__}")
    print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
    print(
        "GPU_NAME="
        + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    print(f"DEVICE={device}")
    print(
        "CONFIG="
        f"seed={base.SEED},window_size={base.WINDOW_SIZE},"
        f"feature_size={base.FEATURE_SIZE},num_classes={base.NUM_CLASSES},"
        f"d_model={base.D_MODEL},nhead={base.NHEAD},"
        f"num_layers={base.NUM_LAYERS},kv_heads={base.KV_HEADS},"
        f"dropout={base.DROPOUT},batch_size={base.BATCH_SIZE},"
        f"learning_rate={base.LEARNING_RATE},weight_decay={base.WEIGHT_DECAY},"
        f"gradient_clip={base.GRADIENT_CLIP},max_epochs={base.MAX_EPOCHS},"
        f"patience={base.PATIENCE},criterion=CrossEntropyLoss,"
        "optimizer=AdamW,checkpoint=Validation_loss"
    )

    model_class = base.load_reference_model(args.reference_repo)
    strict_base_samples = base.load_samples(args.strict_base_csv.resolve())
    strict_expanded_samples = base.load_samples(args.strict_expanded_csv.resolve())
    mild_expanded_samples = base.load_samples(args.mild_expanded_csv.resolve())
    data = prepare_data(
        base,
        strict_base_samples,
        strict_expanded_samples,
        mild_expanded_samples,
    )

    strict_policies = policy_counts(args.strict_expanded_csv.resolve())
    mild_policies = policy_counts(args.mild_expanded_csv.resolve())
    assert strict_policies == Counter({"strict_1s": 795})
    assert mild_policies == Counter({"strict_1s": 650, "mild_2s_le5_span205": 145})
    print(f"STRICT_BASE_SHA256={base.sha256_file(args.strict_base_csv.resolve())}")
    print(
        f"STRICT_EXPANDED_SHA256="
        f"{base.sha256_file(args.strict_expanded_csv.resolve())}"
    )
    print(f"MILD_EXPANDED_SHA256={base.sha256_file(args.mild_expanded_csv.resolve())}")
    print("STRICT_EXPANDED_MILD_RELEASE_SAMPLES=0")

    smoke_only = device.type != "cuda"
    strict_base_result = base.run_experiment(
        "STRICT_BASE",
        arrays(data, "STRICT_BASE_TRAIN"),
        arrays(data, "STRICT_BASE_VAL"),
        model_class,
        device,
        smoke_only,
    )
    strict_expanded_result = base.run_experiment(
        "STRICT_EXPANDED",
        arrays(data, "STRICT_EXPANDED_TRAIN"),
        arrays(data, "STRICT_EXPANDED_VAL"),
        model_class,
        device,
        smoke_only,
    )
    mild_expanded_result = base.run_experiment(
        "MILD_EXPANDED",
        arrays(data, "MILD_EXPANDED_TRAIN"),
        arrays(data, "MILD_EXPANDED_VAL"),
        model_class,
        device,
        smoke_only,
    )

    if smoke_only:
        print("GPU_UNAVAILABLE=True")
        print("TRAINING_STATUS=NOT_RUN_CPU_SMOKE_ONLY")
        return

    assert strict_base_result is not None
    assert strict_expanded_result is not None
    assert mild_expanded_result is not None
    base.print_result("STRICT_BASE", strict_base_result)
    base.print_result("STRICT_EXPANDED", strict_expanded_result)
    base.print_result("MILD_EXPANDED", mild_expanded_result)
    print_delta(
        "STRICT_EXPANDED",
        strict_expanded_result,
        "STRICT_BASE",
        strict_base_result,
        base.CLASSES,
    )
    print_delta(
        "MILD_EXPANDED",
        mild_expanded_result,
        "STRICT_BASE",
        strict_base_result,
        base.CLASSES,
    )
    print_delta(
        "MILD_EXPANDED",
        mild_expanded_result,
        "STRICT_EXPANDED",
        strict_expanded_result,
        base.CLASSES,
    )
    print("TRAINING_STATUS=COMPLETED")


if __name__ == "__main__":
    main()

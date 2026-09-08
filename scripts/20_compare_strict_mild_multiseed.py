#!/usr/bin/env python3
"""Run the equal-size STRICT_EXPANDED vs MILD_EXPANDED multi-seed diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = ROOT / "scripts" / "17_compare_strict_vs_mild_release.py"
SEEDS = (316, 42, 2026, 5858, 1234)
STRICT_EXPANDED_SHA256 = (
    "54CC016782D14343578E738C84B0CC2AC3CBFFE1871C3A1D005F87B272229A8B"
)
MILD_EXPANDED_SHA256 = (
    "2C3A968D209C41ED23C89935E65D441946E2FE344E862B29E4954444B82F43AC"
)


def load_base_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "strict_mild_multiseed_base_runner",
        BASE_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not construct the script 17 module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def split_and_policy_counts(path: Path) -> tuple[Counter[str], Counter[str]]:
    sample_split: dict[str, str] = {}
    sample_policy: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "source_policy" not in fields:
            raise RuntimeError("source_policy column is required")
        for row in reader:
            sample_id = row["sample_id"]
            split = row["split"]
            policy = row["source_policy"]
            if sample_id in sample_split:
                assert sample_split[sample_id] == split
                assert sample_policy[sample_id] == policy
            else:
                sample_split[sample_id] = split
                sample_policy[sample_id] = policy
    return Counter(sample_split.values()), Counter(sample_policy.values())


def prepare_data(
    base: Any,
    strict_samples: dict[str, dict[str, Any]],
    mild_samples: dict[str, dict[str, Any]],
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    data = {
        "STRICT_TRAIN": base.split_arrays(strict_samples, "train"),
        "STRICT_VAL": base.split_arrays(strict_samples, "validation"),
        "MILD_TRAIN": base.split_arrays(mild_samples, "train"),
        "MILD_VAL": base.split_arrays(mild_samples, "validation"),
    }
    assert len(data["STRICT_TRAIN"][0]) == 645
    assert len(data["MILD_TRAIN"][0]) == 645
    assert len(data["STRICT_VAL"][0]) == 150
    assert len(data["MILD_VAL"][0]) == 150
    base.assert_class_counts(data["STRICT_TRAIN"][2], 129, "STRICT_EXPANDED Train")
    base.assert_class_counts(data["MILD_TRAIN"][2], 129, "MILD_EXPANDED Train")
    base.assert_class_counts(data["STRICT_VAL"][2], 30, "STRICT Validation")
    base.assert_class_counts(data["MILD_VAL"][2], 30, "MILD Validation")

    strict_val_ids, strict_val_features, strict_val_labels = data["STRICT_VAL"]
    mild_val_ids, mild_val_features, mild_val_labels = data["MILD_VAL"]
    assert strict_val_ids == mild_val_ids
    assert np.array_equal(strict_val_features, mild_val_features)
    assert np.array_equal(strict_val_labels, mild_val_labels)
    assert set(data["STRICT_TRAIN"][0]).isdisjoint(strict_val_ids)
    assert set(data["MILD_TRAIN"][0]).isdisjoint(mild_val_ids)
    print("VALIDATION_IDENTICAL=True")
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


def print_seed_result(seed: int, arm: str, result: dict[str, Any], classes: tuple[str, ...]) -> None:
    print(f"=== SEED {seed} {arm} RESULT ===")
    print(f"best_epoch={result['best_epoch']}")
    print(f"stop_epoch={result['stop_epoch']}")
    print(f"val_loss={result['val_loss']:.6f}")
    print(f"accuracy={result['accuracy']:.6f}")
    print(f"macro_f1={result['macro_f1']:.6f}")
    print(
        f"highest_macro_f1={result['highest_macro_f1']:.6f} "
        f"epoch={result['highest_macro_f1_epoch']} diagnostic_only=True"
    )
    print("class,recall,f1")
    for index, class_name in enumerate(classes):
        print(
            f"{class_name},{result['recall'][index]:.6f},"
            f"{result['f1'][index]:.6f}"
        )


def print_aggregate(all_results: list[dict[str, Any]]) -> None:
    strict_accuracy = [row["strict"]["accuracy"] for row in all_results]
    mild_accuracy = [row["mild"]["accuracy"] for row in all_results]
    strict_macro = [row["strict"]["macro_f1"] for row in all_results]
    mild_macro = [row["mild"]["macro_f1"] for row in all_results]
    accuracy_delta = [mild - strict for mild, strict in zip(mild_accuracy, strict_accuracy)]
    macro_delta = [mild - strict for mild, strict in zip(mild_macro, strict_macro)]

    print("=== FIVE-SEED AGGREGATE ===")
    print("std_definition=population_ddof_0")
    print(f"strict_accuracy_mean={mean(strict_accuracy):.6f}")
    print(f"strict_accuracy_std={pstdev(strict_accuracy):.6f}")
    print(f"mild_accuracy_mean={mean(mild_accuracy):.6f}")
    print(f"mild_accuracy_std={pstdev(mild_accuracy):.6f}")
    print(f"strict_macro_f1_mean={mean(strict_macro):.6f}")
    print(f"strict_macro_f1_std={pstdev(strict_macro):.6f}")
    print(f"mild_macro_f1_mean={mean(mild_macro):.6f}")
    print(f"mild_macro_f1_std={pstdev(mild_macro):.6f}")
    print(f"accuracy_delta_mean={mean(accuracy_delta):.6f}")
    print(f"accuracy_delta_std={pstdev(accuracy_delta):.6f}")
    print(f"macro_f1_delta_mean={mean(macro_delta):.6f}")
    print(f"macro_f1_delta_std={pstdev(macro_delta):.6f}")
    print(
        "mild_accuracy_higher_seeds="
        f"{sum(mild > strict for mild, strict in zip(mild_accuracy, strict_accuracy))}/5"
    )
    print(
        "mild_macro_f1_higher_seeds="
        f"{sum(mild > strict for mild, strict in zip(mild_macro, strict_macro))}/5"
    )
    print("STATISTICAL_SIGNIFICANCE_CLAIMED=False")


def main() -> None:
    base = load_base_runner()
    args = parse_args()
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
    print("PREDECLARED_ROBUSTNESS_SEEDS=" + ",".join(map(str, SEEDS)))
    print("SEEDS_SELECTED_AFTER_PERFORMANCE=False")
    print(
        "CONFIG="
        f"window_size={base.WINDOW_SIZE},feature_size={base.FEATURE_SIZE},"
        f"num_classes={base.NUM_CLASSES},d_model={base.D_MODEL},"
        f"nhead={base.NHEAD},num_layers={base.NUM_LAYERS},"
        f"kv_heads={base.KV_HEADS},dropout={base.DROPOUT},"
        f"batch_size={base.BATCH_SIZE},learning_rate={base.LEARNING_RATE},"
        f"weight_decay={base.WEIGHT_DECAY},gradient_clip={base.GRADIENT_CLIP},"
        f"max_epochs={base.MAX_EPOCHS},patience={base.PATIENCE},"
        "criterion=CrossEntropyLoss,optimizer=AdamW,checkpoint=Validation_loss"
    )

    strict_path = args.strict_expanded_csv.resolve()
    mild_path = args.mild_expanded_csv.resolve()
    strict_hash = base.sha256_file(strict_path)
    mild_hash = base.sha256_file(mild_path)
    assert strict_hash == STRICT_EXPANDED_SHA256
    assert mild_hash == MILD_EXPANDED_SHA256
    print(f"STRICT_EXPANDED_SHA256={strict_hash}")
    print(f"MILD_EXPANDED_SHA256={mild_hash}")

    strict_splits, strict_policies = split_and_policy_counts(strict_path)
    mild_splits, mild_policies = split_and_policy_counts(mild_path)
    assert strict_splits == Counter({"train": 645, "validation": 150})
    assert mild_splits == Counter({"train": 645, "validation": 150})
    assert strict_policies == Counter({"strict_1s": 795})
    assert mild_policies == Counter({"strict_1s": 650, "mild_2s_le5_span205": 145})

    model_class = base.load_reference_model(args.reference_repo)
    strict_samples = base.load_samples(strict_path)
    mild_samples = base.load_samples(mild_path)
    data = prepare_data(base, strict_samples, mild_samples)

    if device.type != "cuda":
        base.SEED = SEEDS[0]
        print(f"CPU_SMOKE_SEED={SEEDS[0]}")
        base.run_experiment(
            "STRICT_EXPANDED",
            arrays(data, "STRICT_TRAIN"),
            arrays(data, "STRICT_VAL"),
            model_class,
            device,
            smoke_only=True,
        )
        base.run_experiment(
            "MILD_EXPANDED",
            arrays(data, "MILD_TRAIN"),
            arrays(data, "MILD_VAL"),
            model_class,
            device,
            smoke_only=True,
        )
        print("GPU_UNAVAILABLE=True")
        print("TRAINING_STATUS=NOT_RUN_CPU_SMOKE_ONLY")
        return

    all_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        base.SEED = seed
        print(f"=== SEED {seed} START ===")
        strict_result = base.run_experiment(
            f"SEED_{seed}_STRICT_EXPANDED",
            arrays(data, "STRICT_TRAIN"),
            arrays(data, "STRICT_VAL"),
            model_class,
            device,
            smoke_only=False,
        )
        mild_result = base.run_experiment(
            f"SEED_{seed}_MILD_EXPANDED",
            arrays(data, "MILD_TRAIN"),
            arrays(data, "MILD_VAL"),
            model_class,
            device,
            smoke_only=False,
        )
        assert strict_result is not None
        assert mild_result is not None
        print_seed_result(seed, "STRICT_EXPANDED", strict_result, base.CLASSES)
        print_seed_result(seed, "MILD_EXPANDED", mild_result, base.CLASSES)
        print(f"=== SEED {seed} MILD - STRICT ===")
        print(
            f"accuracy_delta="
            f"{mild_result['accuracy'] - strict_result['accuracy']:.6f}"
        )
        print(
            f"macro_f1_delta="
            f"{mild_result['macro_f1'] - strict_result['macro_f1']:.6f}"
        )
        all_results.append({"seed": seed, "strict": strict_result, "mild": mild_result})

    assert len(all_results) == len(SEEDS)
    print_aggregate(all_results)
    print("TRAINING_STATUS=COMPLETED")


if __name__ == "__main__":
    main()

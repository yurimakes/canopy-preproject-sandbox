#!/usr/bin/env python3
"""Train and export the STRICT_LARGE seed-316 SpeedTransformer candidate artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from canopy_ml.speedtransformer_adapter import (  # noqa: E402
    ARTIFACT_STATUS,
    CONFIDENCE_SEMANTICS,
    EXPECTED_HYPERPARAMETERS,
    LABEL_MAPPING,
    MODEL_ARCHITECTURE,
    REFERENCE_COMMIT,
    SCALER_FORMAT,
    SUPPORTED_ARTIFACT_VERSION,
    load_predictor,
    sha256_file,
    validate_artifact_files,
    validate_manifest,
)
from canopy_ml.trip_inference import (  # noqa: E402
    LocationSample,
    PredictionStatus,
    STRICT_PREPROCESSING_VERSION,
    preprocess_strict_trip,
)


BASE_RUNNER_PATH = ROOT / "scripts" / "17_compare_strict_vs_mild_release.py"
STRICT_LARGE_SHA256 = "5CC827DAB88D17F68C5EF7F1A09127A81E8B5F683990EDAB0CC7D00302C275FA"
ARTIFACT_VERSION = SUPPORTED_ARTIFACT_VERSION
MODEL_FILENAME = "model.pt"
SCALER_FILENAME = "scaler.json"
MANIFEST_FILENAME = "manifest.json"
TRAINING_LOG_FILENAME = "training.log"
SEED = 316


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-large-csv",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "speedtransformer_large_control_v1"
            / "speed200_strict_large_v1.csv"
        ),
    )
    parser.add_argument(
        "--reference-repo",
        type=Path,
        default=ROOT.parent / "SpeedTransformer-author-reference",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "outputs" / "speedtransformer_strict_v1",
    )
    parser.add_argument(
        "--artifact-zip",
        type=Path,
        default=ROOT / "speedtransformer_strict_v1_artifact.zip",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and contracts without training or creating artifacts.",
    )
    parser.add_argument(
        "--verify-artifact-dir",
        type=Path,
        help="Independently reload and verify an already exported artifact, then exit.",
    )
    return parser.parse_args()


def load_base_runner() -> Any:
    if not BASE_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Base runner is missing: {BASE_RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "strict_artifact_base_runner",
        BASE_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not construct the script 17 module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_base_protocol(base: Any) -> None:
    actual = {
        "window_size": base.WINDOW_SIZE,
        "feature_size": base.FEATURE_SIZE,
        "num_classes": base.NUM_CLASSES,
        "d_model": base.D_MODEL,
        "nhead": base.NHEAD,
        "num_layers": base.NUM_LAYERS,
        "kv_heads": base.KV_HEADS,
        "dropout": base.DROPOUT,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "weight_decay": base.WEIGHT_DECAY,
        "gradient_clip": base.GRADIENT_CLIP,
        "max_epochs": base.MAX_EPOCHS,
        "patience": base.PATIENCE,
        "criterion": "CrossEntropyLoss",
        "optimizer": "AdamW",
    }
    if actual != EXPECTED_HYPERPARAMETERS:
        raise RuntimeError(f"Script 17 protocol mismatch: {actual}")
    if tuple(base.CLASSES) != tuple(LABEL_MAPPING[str(index)] for index in range(5)):
        raise RuntimeError("Script 17 label mapping mismatch")
    if base.SEED != SEED:
        raise RuntimeError("Script 17 baseline seed mismatch")


def audit_dataset(path: Path, base: Any) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STRICT_LARGE dataset is missing: {path}")
    dataset_hash = sha256_file(path)
    if dataset_hash != STRICT_LARGE_SHA256:
        raise RuntimeError(
            f"STRICT_LARGE SHA256 mismatch: expected {STRICT_LARGE_SHA256}, got {dataset_hash}"
        )

    sample_metadata: dict[str, tuple[str, int, str]] = {}
    seen_steps: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "split",
            "class_name",
            "model_label",
            "step",
            "speed_kmh",
            "source_policy",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            split = row["split"]
            if split not in {"train", "validation"}:
                raise RuntimeError(f"Unexpected split in artifact dataset: {split!r}")
            label = int(row["model_label"])
            if label not in range(5) or row["class_name"] != LABEL_MAPPING[str(label)]:
                raise RuntimeError("Dataset class/label mismatch")
            metadata = (split, label, row["source_policy"])
            previous = sample_metadata.setdefault(row["sample_id"], metadata)
            if previous != metadata:
                raise RuntimeError("Inconsistent sample metadata")
            seen_steps[row["sample_id"]] += 1

    if any(count != 200 for count in seen_steps.values()):
        raise RuntimeError("Every artifact dataset sample must contain 200 rows")
    counts = {
        split: Counter(label for value_split, label, _ in sample_metadata.values() if value_split == split)
        for split in ("train", "validation")
    }
    if counts["train"] != Counter({index: 300 for index in range(5)}):
        raise RuntimeError(f"Train class counts mismatch: {counts['train']}")
    if counts["validation"] != Counter({index: 30 for index in range(5)}):
        raise RuntimeError(f"Validation class counts mismatch: {counts['validation']}")
    policies = Counter(policy for _, _, policy in sample_metadata.values())
    if policies != Counter({"strict_1s": 1650}):
        raise RuntimeError(f"STRICT policy mismatch: {policies}")

    samples = base.load_samples(path)
    train = base.split_arrays(samples, "train")
    validation = base.split_arrays(samples, "validation")
    if len(train[0]) != 1500 or len(validation[0]) != 150:
        raise RuntimeError("Dataset sample totals mismatch")
    if set(train[0]) & set(validation[0]):
        raise RuntimeError("Train/Validation sample overlap detected")
    return {
        "sha256": dataset_hash,
        "train": train,
        "validation": validation,
        "train_counts": {LABEL_MAPPING[str(index)]: counts["train"][index] for index in range(5)},
        "validation_counts": {
            LABEL_MAPPING[str(index)]: counts["validation"][index] for index in range(5)
        },
    }


def fit_scaler(
    train_features: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_2d = train_features.reshape(-1, 1)
    validation_2d = validation_features.reshape(-1, 1)
    scaled_train = scaler.fit_transform(train_2d).reshape(train_features.shape)
    scaled_validation = scaler.transform(validation_2d).reshape(validation_features.shape)
    return scaler, scaled_train, scaled_validation


def train(
    base: Any,
    model_class: type[nn.Module],
    dataset: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, StandardScaler, dict[str, Any], list[str]]:
    base.SEED = SEED
    base.reset_seed()
    train_ids, train_features, train_labels = dataset["train"]
    validation_ids, validation_features, validation_labels = dataset["validation"]
    del train_ids, validation_ids
    scaler, scaled_train, scaled_validation = fit_scaler(train_features, validation_features)
    train_dataset = base.SpeedDataset(scaled_train, train_labels)
    validation_dataset = base.SpeedDataset(scaled_validation, validation_labels)
    train_loader, validation_loader = base.make_loaders(train_dataset, validation_dataset, device)
    model = base.make_model(model_class, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    stop_epoch = base.MAX_EPOCHS
    log_lines = [
        "artifact_status=preproject_candidate",
        "seed=316",
        "seed_selection_status=fixed_baseline_not_selected_after_performance",
        "scaler_fit=train_only",
        "checkpoint_criterion=validation_loss",
        "internal_test_used=false",
        "epoch,train_loss,val_loss,val_accuracy,val_macro_f1",
    ]

    for epoch in range(1, base.MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, None, src_key_padding_mask=None)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), base.GRADIENT_CLIP)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total
        val_loss, accuracy, macro_f1, _, _ = base.evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        line = f"{epoch},{train_loss:.9f},{val_loss:.9f},{accuracy:.9f},{macro_f1:.9f}"
        print(line, flush=True)
        log_lines.append(line)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= base.PATIENCE:
                stop_epoch = epoch
                break

    if best_state is None:
        raise RuntimeError("No Validation-loss checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    val_loss, accuracy, macro_f1, _, _ = base.evaluate(
        model,
        validation_loader,
        criterion,
        device,
    )
    metrics = {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "validation_loss": float(val_loss),
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(macro_f1),
    }
    log_lines.extend(f"{key}={value}" for key, value in metrics.items())
    return model, scaler, metrics, log_lines


def scaler_payload(scaler: StandardScaler) -> dict[str, Any]:
    return {
        "format": SCALER_FORMAT,
        "feature_names": ["speed_kmh"],
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "var": [float(value) for value in scaler.var_],
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }


def runtime_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def write_artifact(
    artifact_dir: Path,
    model: nn.Module,
    scaler: StandardScaler,
    dataset: dict[str, Any],
    metrics: dict[str, Any],
    log_lines: list[str],
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"Artifact directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / MODEL_FILENAME
    scaler_path = artifact_dir / SCALER_FILENAME
    manifest_path = artifact_dir / MANIFEST_FILENAME

    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(cpu_state, model_path)
    scaler_path.write_text(
        json.dumps(scaler_payload(scaler), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "model_architecture": MODEL_ARCHITECTURE,
        "reference_commit": REFERENCE_COMMIT,
        "dataset_sha256": dataset["sha256"],
        "preprocessing_version": STRICT_PREPROCESSING_VERSION,
        "label_mapping": LABEL_MAPPING,
        "label_mapping_status": "candidate",
        "seed": SEED,
        "seed_selection_status": "fixed_baseline_not_selected_after_performance",
        "train_counts": dataset["train_counts"],
        "validation_counts": dataset["validation_counts"],
        "internal_test_used": False,
        "hyperparameters": EXPECTED_HYPERPARAMETERS,
        "checkpoint_criterion": "validation_loss",
        **metrics,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": MODEL_FILENAME,
        "scaler_file": SCALER_FILENAME,
        "model_sha256": sha256_file(model_path),
        "scaler_sha256": sha256_file(scaler_path),
        "confidence_semantics": CONFIDENCE_SEMANTICS,
    }
    exporter_commit = runtime_git_commit()
    if exporter_commit is not None:
        manifest["exporter_git_commit"] = exporter_commit
    validate_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / TRAINING_LOG_FILENAME).write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    validate_artifact_files(artifact_dir)
    return manifest


def synthetic_ready_fixture() -> Any:
    samples = [
        LocationSample(
            timestamp_ms=index * 1_000,
            latitude=0.0,
            longitude=index * 0.00001,
        )
        for index in range(201)
    ]
    result = preprocess_strict_trip(samples)
    if result.speed_values_kmh is None or len(result.speed_values_kmh) != 200:
        raise RuntimeError("Synthetic strict fixture did not produce 200 speeds")
    return result


def verify_artifact(artifact_dir: Path, reference_repo: Path, device: str) -> None:
    files = validate_artifact_files(artifact_dir)
    predictor = load_predictor(artifact_dir, reference_repo, device=device)
    result = predictor.predict("privacy-safe-synthetic-trip", synthetic_ready_fixture())
    if result.prediction.prediction_status is not PredictionStatus.PREDICTED:
        raise RuntimeError("Reloaded artifact did not produce a prediction")
    if len(result.logits) != 5 or not all(np.isfinite(result.logits)):
        raise RuntimeError("Reloaded artifact did not produce five finite logits")
    if predictor.model.training:
        raise RuntimeError("Reloaded model is not in eval mode")
    print(f"MODEL_SHA256={files.manifest['model_sha256']}")
    print(f"SCALER_SHA256={files.manifest['scaler_sha256']}")
    print("INDEPENDENT_RELOAD_FORWARD=PASS")
    print("LOGITS_SHAPE=(1,5)")


def independent_verify(artifact_dir: Path, reference_repo: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify-artifact-dir",
            str(artifact_dir.resolve()),
            "--reference-repo",
            str(reference_repo.resolve()),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def make_artifact_zip(artifact_dir: Path, artifact_zip: Path) -> list[str]:
    artifact_zip = artifact_zip.resolve()
    artifact_zip.parent.mkdir(parents=True, exist_ok=True)
    expected = [MODEL_FILENAME, SCALER_FILENAME, MANIFEST_FILENAME, TRAINING_LOG_FILENAME]
    with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in expected:
            archive.write(artifact_dir / name, arcname=name)
    with zipfile.ZipFile(artifact_zip) as archive:
        names = archive.namelist()
    if names != expected or any("\\" in name for name in names):
        raise RuntimeError(f"Artifact ZIP contents mismatch: {names}")
    return names


def main() -> None:
    args = parse_args()
    if args.verify_artifact_dir is not None:
        verify_artifact(args.verify_artifact_dir, args.reference_repo, args.device)
        return

    base = load_base_runner()
    assert_base_protocol(base)
    dataset = audit_dataset(args.strict_large_csv, base)
    model_class = base.load_reference_model(args.reference_repo)
    print(f"STRICT_LARGE_SHA256={dataset['sha256']}")
    print("TRAIN_COUNTS=300,300,300,300,300")
    print("VALIDATION_COUNTS=30,30,30,30,30")
    print("INTERNAL_TEST_USED=False")
    print("SEED=316")
    print("SEED_SELECTION_STATUS=fixed_baseline_not_selected_after_performance")
    print("PROTOCOL_VERIFIED_AGAINST_SCRIPT_17=True")
    if args.validate_only:
        print("VALIDATION_ONLY=PASS")
        print("TRAINING_STATUS=NOT_RUN")
        return
    if args.device != "cuda":
        raise RuntimeError("Training/export requires --device cuda; CPU training is prohibited")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    device = torch.device("cuda")
    print(f"GPU_NAME={torch.cuda.get_device_name(0)}")
    model, scaler, metrics, log_lines = train(base, model_class, dataset, device)
    manifest = write_artifact(
        args.artifact_dir,
        model,
        scaler,
        dataset,
        metrics,
        log_lines,
    )
    verification_output = independent_verify(args.artifact_dir, args.reference_repo)
    log_path = args.artifact_dir.resolve() / TRAINING_LOG_FILENAME
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(verification_output + "\n")
    names = make_artifact_zip(args.artifact_dir.resolve(), args.artifact_zip)
    print(verification_output)
    print(f"ARTIFACT_VERSION={manifest['artifact_version']}")
    print(f"ARTIFACT_ZIP={args.artifact_zip.resolve()}")
    print(f"ARTIFACT_ZIP_CONTENTS={','.join(names)}")
    print(f"ARTIFACT_ZIP_SHA256={sha256_file(args.artifact_zip.resolve())}")
    print("TRAINING_STATUS=COMPLETED")


if __name__ == "__main__":
    main()

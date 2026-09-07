#!/usr/bin/env python3
"""Controlled strict-vs-mild SpeedTransformer Validation comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959"

CLASSES = ("WALK", "BIKE", "CAR", "BUS", "SUBWAY")
WINDOW_SIZE = 200
FEATURE_SIZE = 1
NUM_CLASSES = 5

SEED = 316
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
KV_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MAX_EPOCHS = 50
PATIENCE = 7


class SpeedDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-csv",
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
        "--mild-csv",
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


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def reset_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_reference_model(reference_repo: Path) -> type[nn.Module]:
    reference_repo = reference_repo.resolve()
    model_path = reference_repo / "models" / "transformer" / "model_utils.py"
    if not model_path.is_file():
        raise FileNotFoundError("Reference model module was not found")

    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={reference_repo.as_posix()}",
            "-C",
            str(reference_repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(
            f"Reference commit mismatch: expected {REFERENCE_COMMIT}, got {commit}"
        )

    spec = importlib.util.spec_from_file_location(
        "speedtransformer_reference_model_utils",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not construct the reference model module spec")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, "TrajectoryTransformer", None)
    if model_class is None:
        raise ImportError("TrajectoryTransformer is absent from the reference module")

    print(f"REFERENCE_COMMIT={commit}")
    print("REFERENCE_MODEL=models/transformer/model_utils.py:TrajectoryTransformer")
    return model_class


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {display_path(path)}")

    samples: dict[str, dict[str, Any]] = {}
    required = {
        "sample_id",
        "split",
        "class_name",
        "model_label",
        "step",
        "speed_kmh",
    }

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            split = row["split"]
            if split not in ("train", "validation"):
                continue

            sample_id = row["sample_id"]
            label = int(row["model_label"])
            step = int(row["step"])
            speed = float(row["speed_kmh"])

            if label not in range(NUM_CLASSES):
                raise RuntimeError(f"Invalid model label for sample {sample_id}")
            if row["class_name"] != CLASSES[label]:
                raise RuntimeError(f"Class/label mismatch for sample {sample_id}")
            if not np.isfinite(speed):
                raise RuntimeError(f"Non-finite speed for sample {sample_id}")

            sample = samples.setdefault(
                sample_id,
                {
                    "split": split,
                    "label": label,
                    "steps": {},
                },
            )
            if (sample["split"], sample["label"]) != (split, label):
                raise RuntimeError(f"Inconsistent metadata for sample {sample_id}")
            if step in sample["steps"]:
                raise RuntimeError(f"Duplicate step for sample {sample_id}")
            sample["steps"][step] = speed

    for sample_id, sample in samples.items():
        if sorted(sample["steps"]) != list(range(WINDOW_SIZE)):
            raise RuntimeError(f"Sample {sample_id} does not contain steps 0..199")
        sample["values"] = np.asarray(
            [sample["steps"][step] for step in range(WINDOW_SIZE)],
            dtype=np.float64,
        )
        del sample["steps"]

    return samples


def split_arrays(
    samples: dict[str, dict[str, Any]],
    split: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(
        sample_id
        for sample_id, sample in samples.items()
        if sample["split"] == split
    )
    features = np.stack([samples[sample_id]["values"] for sample_id in ids])
    features = features[:, :, np.newaxis]
    labels = np.asarray([samples[sample_id]["label"] for sample_id in ids])
    return ids, features, labels


def assert_class_counts(labels: np.ndarray, expected: int, name: str) -> None:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if not np.array_equal(counts, np.full(NUM_CLASSES, expected)):
        raise AssertionError(f"{name} class counts mismatch: {counts.tolist()}")


def validate_controlled_datasets(
    strict_samples: dict[str, dict[str, Any]],
    mild_samples: dict[str, dict[str, Any]],
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    strict_train = split_arrays(strict_samples, "train")
    strict_val = split_arrays(strict_samples, "validation")
    mild_train = split_arrays(mild_samples, "train")
    mild_val = split_arrays(mild_samples, "validation")

    assert len(strict_train[0]) == 500
    assert len(strict_val[0]) == 150
    assert len(mild_train[0]) == 645
    assert len(mild_val[0]) == 150
    assert_class_counts(strict_train[2], 100, "STRICT Train")
    assert_class_counts(strict_val[2], 30, "STRICT Validation")
    assert_class_counts(mild_train[2], 129, "MILD Train")
    assert_class_counts(mild_val[2], 30, "MILD Validation")

    strict_train_ids = set(strict_train[0])
    strict_val_ids = set(strict_val[0])
    mild_train_ids = set(mild_train[0])
    mild_val_ids = set(mild_val[0])
    assert strict_train_ids.issubset(mild_train_ids)
    assert len(mild_train_ids - strict_train_ids) == 145
    assert strict_train_ids.isdisjoint(strict_val_ids)
    assert mild_train_ids.isdisjoint(mild_val_ids)
    assert strict_val_ids == mild_val_ids
    assert strict_val[0] == mild_val[0]
    assert np.array_equal(strict_val[1], mild_val[1])
    assert np.array_equal(strict_val[2], mild_val[2])

    print("VALIDATION_IDENTICAL=True")
    print("STRICT_TRAIN_COUNTS=100,100,100,100,100")
    print("MILD_TRAIN_COUNTS=129,129,129,129,129")
    print("VALIDATION_COUNTS=30,30,30,30,30")
    print("INTERNAL_TEST_USED=False")

    return {
        "STRICT_TRAIN": strict_train,
        "STRICT_VAL": strict_val,
        "MILD_TRAIN": mild_train,
        "MILD_VAL": mild_val,
    }


def scaled_datasets(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[SpeedDataset, SpeedDataset]:
    scaler = StandardScaler()
    train_2d = train_features.reshape(-1, FEATURE_SIZE)
    val_2d = val_features.reshape(-1, FEATURE_SIZE)
    scaled_train = scaler.fit_transform(train_2d).reshape(train_features.shape)
    scaled_val = scaler.transform(val_2d).reshape(val_features.shape)
    return (
        SpeedDataset(scaled_train, train_labels),
        SpeedDataset(scaled_val, val_labels),
    )


def make_model(model_class: type[nn.Module], device: torch.device) -> nn.Module:
    model = model_class(
        feature_size=FEATURE_SIZE,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        window_size=WINDOW_SIZE,
        dropout=DROPOUT,
        kv_heads=KV_HEADS,
    ).to(device)

    # Match TrajectoryModel.prepare_model in the reference implementation.
    for name, parameter in model.named_parameters():
        if parameter.dim() > 1 and "weight" in name:
            nn.init.xavier_uniform_(parameter)
        elif "bias" in name:
            nn.init.constant_(parameter, 0)
    return model


def make_loaders(
    train_dataset: SpeedDataset,
    val_dataset: SpeedDataset,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features, None, src_key_padding_mask=None)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).cpu().tolist())

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(predictions_all)
    return (
        total_loss / total,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        y_true,
        y_pred,
    )


def metric_details(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        ),
    }


def run_experiment(
    dataset_name: str,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    model_class: type[nn.Module],
    device: torch.device,
    smoke_only: bool,
) -> dict[str, Any] | None:
    reset_seed()
    train_dataset, val_dataset = scaled_datasets(
        train_data[0],
        train_data[1],
        val_data[0],
        val_data[1],
    )
    print(f"{dataset_name}_SCALER_FIT=TRAIN_ONLY")
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, device)
    model = make_model(model_class, device)
    print(
        f"{dataset_name}_MODEL_PARAMETERS="
        f"{sum(parameter.numel() for parameter in model.parameters())}"
    )
    criterion = nn.CrossEntropyLoss()

    if smoke_only:
        model.eval()
        features, labels = next(iter(train_loader))
        with torch.no_grad():
            logits = model(
                features.to(device),
                None,
                src_key_padding_mask=None,
            )
            loss = criterion(logits, labels.to(device))
        assert logits.shape == (len(labels), NUM_CLASSES)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)
        print(
            f"{dataset_name}_ONE_BATCH_FORWARD_OK=True "
            f"batch={len(labels)} logits_shape={tuple(logits.shape)} "
            f"loss={loss.item():.6f}"
        )
        return None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    highest_macro_f1 = float("-inf")
    highest_macro_f1_epoch = 0
    stop_epoch = MAX_EPOCHS

    print(f"=== {dataset_name} EPOCH METRICS ===")
    print("epoch,train_loss,val_loss,val_accuracy,val_macro_f1")

    for epoch in range(1, MAX_EPOCHS + 1):
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
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total
        val_loss, val_accuracy, val_macro_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        print(
            f"{epoch},{train_loss:.6f},{val_loss:.6f},"
            f"{val_accuracy:.6f},{val_macro_f1:.6f}"
        )

        if val_macro_f1 > highest_macro_f1:
            highest_macro_f1 = val_macro_f1
            highest_macro_f1_epoch = epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                stop_epoch = epoch
                break

    if best_state is None:
        raise RuntimeError("No Validation-loss checkpoint was selected")
    model.load_state_dict(best_state)
    val_loss, accuracy, macro_f1, y_true, y_pred = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )
    details = metric_details(y_true, y_pred)
    return {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "highest_macro_f1": highest_macro_f1,
        "highest_macro_f1_epoch": highest_macro_f1_epoch,
        **details,
    }


def print_result(name: str, result: dict[str, Any]) -> None:
    print(f"=== {name} BEST VALIDATION-LOSS CHECKPOINT ===")
    print(f"best_epoch={result['best_epoch']}")
    print(f"stop_epoch={result['stop_epoch']}")
    print(f"val_loss={result['val_loss']:.6f}")
    print(f"accuracy={result['accuracy']:.6f}")
    print(f"macro_f1={result['macro_f1']:.6f}")
    print(
        "highest_macro_f1="
        f"{result['highest_macro_f1']:.6f} "
        f"epoch={result['highest_macro_f1_epoch']}"
    )
    print("class,precision,recall,f1")
    for index, class_name in enumerate(CLASSES):
        print(
            f"{class_name},{result['precision'][index]:.6f},"
            f"{result['recall'][index]:.6f},{result['f1'][index]:.6f}"
        )
    print("confusion_matrix=")
    print(result["confusion_matrix"].tolist())


def print_comparison(strict: dict[str, Any], mild: dict[str, Any]) -> None:
    print("=== MILD - STRICT (SAME VALIDATION 150) ===")
    print(f"accuracy_delta={mild['accuracy'] - strict['accuracy']:.6f}")
    print(f"macro_f1_delta={mild['macro_f1'] - strict['macro_f1']:.6f}")
    print("class,recall_delta,f1_delta")
    for index, class_name in enumerate(CLASSES):
        print(
            f"{class_name},{mild['recall'][index] - strict['recall'][index]:.6f},"
            f"{mild['f1'][index] - strict['f1'][index]:.6f}"
        )


def main() -> None:
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
        if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        )
        else "cpu"
    )
    print(f"DEVICE={device}")
    print(
        "CONFIG="
        f"seed={SEED},window_size={WINDOW_SIZE},feature_size={FEATURE_SIZE},"
        f"num_classes={NUM_CLASSES},d_model={D_MODEL},nhead={NHEAD},"
        f"num_layers={NUM_LAYERS},kv_heads={KV_HEADS},dropout={DROPOUT},"
        f"batch_size={BATCH_SIZE},learning_rate={LEARNING_RATE},"
        f"weight_decay={WEIGHT_DECAY},gradient_clip={GRADIENT_CLIP},"
        f"max_epochs={MAX_EPOCHS},patience={PATIENCE},"
        "criterion=CrossEntropyLoss,optimizer=AdamW,"
        "checkpoint=Validation_loss"
    )

    model_class = load_reference_model(args.reference_repo)
    strict_samples = load_samples(args.strict_csv.resolve())
    mild_samples = load_samples(args.mild_csv.resolve())
    data = validate_controlled_datasets(strict_samples, mild_samples)
    print(f"STRICT_SHA256={sha256_file(args.strict_csv.resolve())}")
    print(f"MILD_SHA256={sha256_file(args.mild_csv.resolve())}")

    strict_train = (data["STRICT_TRAIN"][1], data["STRICT_TRAIN"][2])
    strict_val = (data["STRICT_VAL"][1], data["STRICT_VAL"][2])
    mild_train = (data["MILD_TRAIN"][1], data["MILD_TRAIN"][2])
    mild_val = (data["MILD_VAL"][1], data["MILD_VAL"][2])

    smoke_only = device.type != "cuda"
    strict_result = run_experiment(
        "STRICT",
        strict_train,
        strict_val,
        model_class,
        device,
        smoke_only,
    )
    mild_result = run_experiment(
        "MILD",
        mild_train,
        mild_val,
        model_class,
        device,
        smoke_only,
    )

    if smoke_only:
        print("GPU_UNAVAILABLE=True")
        print("TRAINING_STATUS=NOT_RUN_CPU_SMOKE_ONLY")
        return

    assert strict_result is not None
    assert mild_result is not None
    print_result("STRICT", strict_result)
    print_result("MILD", mild_result)
    print_comparison(strict_result, mild_result)
    print("TRAINING_STATUS=COMPLETED")


if __name__ == "__main__":
    main()

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "reports" / "aihub_zip_manifest.csv"

with manifest.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

gps = [r for r in rows if r["kind"] == "gps"]

train = [r for r in gps if r["split"] == "training"]
val   = [r for r in gps if r["split"] == "validation"]

train_uid = {r["uid"] for r in train}
val_uid   = {r["uid"] for r in val}

train_trip = {(r["uid"], r["tid"], r["sid"]) for r in train}
val_trip   = {(r["uid"], r["tid"], r["sid"]) for r in val}

train_full = {
    (r["class"], r["uid"], r["tid"], r["sid"])
    for r in train
}
val_full = {
    (r["class"], r["uid"], r["tid"], r["sid"])
    for r in val
}

classes = ["WALK", "BIKE", "CAR", "BUS", "SUBWAY", "ETC"]

print("=== UID CHECK ===")
print("TRAIN UID =", len(train_uid))
print("VAL UID   =", len(val_uid))
print("UID OVERLAP =", len(train_uid & val_uid))
print("VAL UID SUBSET OF TRAIN =", val_uid <= train_uid)

print()
print("=== TRAJECTORY CHECK ===")
print("TRAIN UID/TID/SID =", len(train_trip))
print("VAL UID/TID/SID   =", len(val_trip))
print("TRAJECTORY OVERLAP =", len(train_trip & val_trip))
print("CLASS+TRAJECTORY OVERLAP =", len(train_full & val_full))

print()
print("=== CLASS UID CHECK ===")
for cls in classes:
    tu = {r["uid"] for r in train if r["class"] == cls}
    vu = {r["uid"] for r in val if r["class"] == cls}

    print(
        f"{cls:7s}",
        "train_uid=", len(tu),
        "val_uid=", len(vu),
        "overlap=", len(tu & vu)
    )


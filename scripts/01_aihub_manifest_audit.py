from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

file_pattern = re.compile(
    r"^TMC-(GPS|LABEL)-([^-]+)-([^-]+)-([^-]+)-(Dataset|Label)\.csv$",
    re.IGNORECASE
)

class_pattern = re.compile(
    r"_(WALK|BIKE|CAR|BUS|SUBWAY|ETC)\.zip$",
    re.IGNORECASE
)

rows = []
bad_names = []

for package in sorted(PACKAGES.glob("*.zip")):

    name = package.name

    if name.startswith(("TS_", "TL_")):
        split = "training"
    elif name.startswith(("VS_", "VL_")):
        split = "validation"
    else:
        raise ValueError(f"Unknown split: {name}")

    kind = "gps" if "_GPS_" in name else "label"

    cm = class_pattern.search(name)
    if not cm:
        raise ValueError(f"Class parse failed: {name}")

    class_name = cm.group(1).upper()

    with ZipFile(package) as z:
        for member in z.namelist():

            if not member.lower().endswith(".csv"):
                continue

            basename = PurePosixPath(member).name
            m = file_pattern.match(basename)

            if not m:
                bad_names.append((name, basename))
                continue

            uid = m.group(2)
            tid = m.group(3)
            sid = m.group(4)

            rows.append({
                "split": split,
                "kind": kind,
                "class": class_name,
                "uid": uid,
                "tid": tid,
                "sid": sid,
                "package": name,
                "member": member,
            })

manifest_path = REPORTS / "aihub_zip_manifest.csv"

with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

gps_rows = [r for r in rows if r["kind"] == "gps"]
label_rows = [r for r in rows if r["kind"] == "label"]

gps_keys = {
    (r["split"], r["class"], r["uid"], r["tid"], r["sid"])
    for r in gps_rows
}

label_keys = {
    (r["split"], r["class"], r["uid"], r["tid"], r["sid"])
    for r in label_rows
}

train_uids = {
    r["uid"] for r in gps_rows
    if r["split"] == "training"
}

val_uids = {
    r["uid"] for r in gps_rows
    if r["split"] == "validation"
}

uid_overlap = train_uids & val_uids

print()
print("TOTAL ROWS       =", len(rows))
print("GPS ROWS         =", len(gps_rows))
print("LABEL ROWS       =", len(label_rows))
print("GPS WITHOUT LABEL=", len(gps_keys - label_keys))
print("LABEL WITHOUT GPS=", len(label_keys - gps_keys))
print("TRAIN UID COUNT  =", len(train_uids))
print("VAL UID COUNT    =", len(val_uids))
print("UID OVERLAP      =", len(uid_overlap))
print("BAD FILENAMES    =", len(bad_names))
print()
print("MANIFEST =", manifest_path)

if uid_overlap:
    print()
    print("OVERLAP UID SAMPLE:")
    for uid in sorted(uid_overlap)[:20]:
        print(uid)


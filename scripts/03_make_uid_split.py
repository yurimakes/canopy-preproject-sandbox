import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "aihub_zip_manifest.csv"
OUT = ROOT / "splits"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["WALK", "BIKE", "CAR", "BUS", "SUBWAY", "ETC"]

with MANIFEST.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# GPS만 사용. Label은 GPS와 1:1 대응이 이미 검증됨.
gps = [r for r in rows if r["kind"] == "gps"]

uid_class = defaultdict(Counter)

for r in gps:
    uid_class[r["uid"]][r["class"]] += 1

uids = sorted(uid_class)
n_uid = len(uids)

# UID 기준 80/10/10
n_train = round(n_uid * 0.80)
n_val = round(n_uid * 0.10)
n_test = n_uid - n_train - n_val

total_class = Counter()
for uid in uids:
    total_class.update(uid_class[uid])

total_rows = sum(total_class.values())

def evaluate(order):
    groups = {
        "train": order[:n_train],
        "validation": order[n_train:n_train+n_val],
        "internal_test": order[n_train+n_val:]
    }

    score = 0.0

    targets = {
        "train": 0.80,
        "validation": 0.10,
        "internal_test": 0.10
    }

    for split, split_uids in groups.items():
        counts = Counter()
        for uid in split_uids:
            counts.update(uid_class[uid])

        split_total = sum(counts.values())

        # 전체 trajectory 비율
        score += abs(split_total / total_rows - targets[split]) * 10

        # 클래스별 trajectory 비율
        for cls in CLASSES:
            if total_class[cls] == 0:
                continue

            ratio = counts[cls] / total_class[cls]
            score += abs(ratio - targets[split])

            # 클래스가 split에서 사라지면 강한 패널티
            if counts[cls] == 0:
                score += 1000

    return score, groups

best = None

# deterministic search
for seed in range(10000):
    order = uids.copy()
    random.Random(seed).shuffle(order)

    score, groups = evaluate(order)

    candidate = (score, seed, groups)

    if best is None or candidate[0] < best[0]:
        best = candidate

score, seed, groups = best

uid_split = {}

for split, split_uids in groups.items():
    for uid in split_uids:
        uid_split[uid] = split

out_csv = OUT / "uid_split_v1.csv"

with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["uid", "split"])

    for uid in sorted(uid_split):
        writer.writerow([uid, uid_split[uid]])

sha = hashlib.sha256(out_csv.read_bytes()).hexdigest()

sha_path = OUT / "uid_split_v1.sha256"
sha_path.write_text(
    f"{sha}  uid_split_v1.csv\n",
    encoding="utf-8"
)

print("SEARCH SEED =", seed)
print("SCORE       =", score)
print("TOTAL UID   =", n_uid)
print()

sets = {}

for split in ["train", "validation", "internal_test"]:
    split_uids = set(groups[split])
    sets[split] = split_uids

    counts = Counter()
    for uid in split_uids:
        counts.update(uid_class[uid])

    print("===", split.upper(), "===")
    print("UID =", len(split_uids))
    print("TRAJECTORIES =", sum(counts.values()))

    for cls in CLASSES:
        print(f"{cls:7s} = {counts[cls]}")

    print()

print("=== OVERLAP ===")
print("train ∩ validation    =", len(sets["train"] & sets["validation"]))
print("train ∩ internal_test =", len(sets["train"] & sets["internal_test"]))
print("validation ∩ test     =", len(sets["validation"] & sets["internal_test"]))

print()
print("SHA256 =", sha)
print("CSV    =", out_csv)
print("SHA    =", sha_path)


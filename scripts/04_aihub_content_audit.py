from pathlib import Path
from zipfile import ZipFile
import csv
import io
import re
import statistics
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

gps_packages = sorted(PACKAGES.glob("*_GPS_*.zip"))
label_packages = sorted(PACKAGES.glob("*_LABEL_*.zip"))

print("GPS PACKAGES   =", len(gps_packages))
print("LABEL PACKAGES =", len(label_packages))

gps_headers = Counter()
label_headers = Counter()

gps_file_count = 0
label_file_count = 0

gps_point_counts = []
label_point_counts = []

timestamp_gaps = []
accuracy_values = []

gps_empty = 0
label_empty = 0
gps_bad_rows = 0
label_bad_rows = 0

duplicate_timestamp_files = 0
nonpositive_dt_files = 0

def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")

for pkg in gps_packages:
    with ZipFile(pkg) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue

            gps_file_count += 1

            text = decode(z.read(member))
            reader = csv.DictReader(io.StringIO(text))

            fields = tuple(reader.fieldnames or [])
            gps_headers[fields] += 1

            rows = list(reader)
            gps_point_counts.append(len(rows))

            if not rows:
                gps_empty += 1
                continue

            timestamps = []

            for row in rows:
                try:
                    ts = float(row.get("timestamp", ""))
                    timestamps.append(ts)
                except:
                    gps_bad_rows += 1

                try:
                    acc = float(row.get("accuracy", ""))
                    accuracy_values.append(acc)
                except:
                    pass

            if len(timestamps) >= 2:
                dup = len(timestamps) != len(set(timestamps))
                if dup:
                    duplicate_timestamp_files += 1

                dts = [
                    timestamps[i] - timestamps[i-1]
                    for i in range(1, len(timestamps))
                ]

                if any(dt <= 0 for dt in dts):
                    nonpositive_dt_files += 1

                timestamp_gaps.extend(dt for dt in dts if dt > 0)

for pkg in label_packages:
    with ZipFile(pkg) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue

            label_file_count += 1

            text = decode(z.read(member))
            reader = csv.DictReader(io.StringIO(text))

            fields = tuple(reader.fieldnames or [])
            label_headers[fields] += 1

            rows = list(reader)
            label_point_counts.append(len(rows))

            if not rows:
                label_empty += 1

def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = int(round((len(s) - 1) * p))
    return s[idx]

print()
print("=== FILE COUNTS ===")
print("GPS CSV   =", gps_file_count)
print("LABEL CSV =", label_file_count)

print()
print("=== GPS HEADERS ===")
for h, n in gps_headers.items():
    print(n, h)

print()
print("=== LABEL HEADERS ===")
for h, n in label_headers.items():
    print(n, h)

print()
print("=== GPS POINTS PER TRAJECTORY ===")
print("min    =", min(gps_point_counts))
print("p25    =", percentile(gps_point_counts, 0.25))
print("median =", percentile(gps_point_counts, 0.50))
print("p75    =", percentile(gps_point_counts, 0.75))
print("p95    =", percentile(gps_point_counts, 0.95))
print("max    =", max(gps_point_counts))

print()
print("=== LABEL POINTS PER TRAJECTORY ===")
print("min    =", min(label_point_counts))
print("median =", percentile(label_point_counts, 0.50))
print("max    =", max(label_point_counts))

print()
print("=== TIMESTAMP GAP ===")
if timestamp_gaps:
    print("min    =", min(timestamp_gaps))
    print("p25    =", percentile(timestamp_gaps, 0.25))
    print("median =", percentile(timestamp_gaps, 0.50))
    print("p75    =", percentile(timestamp_gaps, 0.75))
    print("p95    =", percentile(timestamp_gaps, 0.95))
    print("max    =", max(timestamp_gaps))

print()
print("=== QUALITY ===")
print("GPS EMPTY FILES             =", gps_empty)
print("LABEL EMPTY FILES           =", label_empty)
print("GPS BAD TIMESTAMP ROWS      =", gps_bad_rows)
print("DUPLICATE TIMESTAMP FILES   =", duplicate_timestamp_files)
print("NONPOSITIVE DT FILES        =", nonpositive_dt_files)

print()
print("=== GPS ACCURACY ===")
if accuracy_values:
    print("count  =", len(accuracy_values))
    print("min    =", min(accuracy_values))
    print("median =", percentile(accuracy_values, 0.50))
    print("p95    =", percentile(accuracy_values, 0.95))
    print("max    =", max(accuracy_values))


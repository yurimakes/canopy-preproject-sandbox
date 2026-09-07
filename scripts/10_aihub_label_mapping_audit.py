from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv
import io
import re

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "data" / "raw" / "aihub" / "packages"

NAME_RE = re.compile(
    r"^TMC-LABEL-([^-]+)-([^-]+)-([^-]+)-Label\.csv$",
    re.I
)


def decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


row_cross = Counter()
segment_cross = Counter()

class_rows = Counter()
class_segments = Counter()

mixed_label_segments = Counter()
mixed_detail_segments = Counter()

detail_to_classes = defaultdict(set)
label_to_classes = defaultdict(set)

total_rows = 0
total_segments = 0


for pkg in sorted(PACKAGES.glob("*_LABEL_*.zip")):

    if pkg.name.startswith("TL_"):
        source_split = "training"
    elif pkg.name.startswith("VL_"):
        source_split = "validation"
    else:
        continue

    folder_class = pkg.stem.split("_")[-1].upper()

    with ZipFile(pkg) as z:

        for member in z.namelist():

            if not member.lower().endswith(".csv"):
                continue

            base = PurePosixPath(member).name
            match = NAME_RE.match(base)

            if not match:
                continue

            rows = list(
                csv.DictReader(
                    io.StringIO(
                        decode(z.read(member))
                    )
                )
            )

            total_segments += 1
            class_segments[folder_class] += 1

            labels = set()
            details = set()

            for row in rows:

                label = str(
                    row.get("label", "")
                ).strip()

                detail = str(
                    row.get("detail_label", "")
                ).strip()

                total_rows += 1
                class_rows[folder_class] += 1

                labels.add(label)
                details.add(detail)

                row_cross[
                    (
                        folder_class,
                        label,
                        detail,
                    )
                ] += 1

                label_to_classes[
                    label
                ].add(folder_class)

                detail_to_classes[
                    detail
                ].add(folder_class)

            if len(labels) > 1:
                mixed_label_segments[
                    folder_class
                ] += 1

            if len(details) > 1:
                mixed_detail_segments[
                    folder_class
                ] += 1

            segment_cross[
                (
                    folder_class,
                    tuple(sorted(labels)),
                    tuple(sorted(details)),
                )
            ] += 1


print("=== BASIC ===")
print("LABEL SEGMENTS =", total_segments)
print("LABEL ROWS     =", total_rows)


print()
print("=== FOLDER CLASS SUMMARY ===")

for cls in [
    "WALK",
    "BIKE",
    "CAR",
    "BUS",
    "SUBWAY",
    "ETC",
]:

    print()
    print(cls)
    print("segments =", class_segments[cls])
    print("rows     =", class_rows[cls])
    print(
        "mixed label segments       =",
        mixed_label_segments[cls],
    )
    print(
        "mixed detail-label segments=",
        mixed_detail_segments[cls],
    )


print()
print("=== ROW CROSS-TABLE: FOLDER / LABEL / DETAIL ===")

for key, count in sorted(
    row_cross.items(),
    key=lambda item: (
        item[0][0],
        item[0][1],
        item[0][2],
    ),
):

    folder_class, label, detail = key

    print(
        folder_class,
        "label=" + repr(label),
        "detail=" + repr(detail),
        "rows=",
        count,
    )


print()
print("=== SEGMENT LABEL-SET STRUCTURE ===")

for key, count in sorted(
    segment_cross.items(),
    key=lambda item: (
        item[0][0],
        item[0][1],
        item[0][2],
    ),
):

    folder_class, labels, details = key

    print(
        folder_class,
        "labels=",
        labels,
        "details=",
        details,
        "segments=",
        count,
    )


print()
print("=== DETAIL CODE -> FOLDER CLASSES ===")

for detail in sorted(
    detail_to_classes,
    key=lambda x: (
        int(x) if x.isdigit() else 999999,
        x,
    ),
):

    print(
        repr(detail),
        "->",
        sorted(detail_to_classes[detail]),
    )


print()
print("=== LABEL CODE -> FOLDER CLASSES ===")

for label in sorted(
    label_to_classes,
    key=lambda x: (
        int(x) if x.isdigit() else 999999,
        x,
    ),
):

    print(
        repr(label),
        "->",
        sorted(label_to_classes[label]),
    )


candidate_classes = {
    "WALK",
    "BIKE",
    "CAR",
    "BUS",
    "SUBWAY",
}

candidate_rows = sum(
    class_rows[c]
    for c in candidate_classes
)

candidate_segments = sum(
    class_segments[c]
    for c in candidate_classes
)


print()
print("=== CANDIDATE FOLDER-LEVEL 5-CLASS COVERAGE ===")
print(
    "candidate classes =",
    sorted(candidate_classes),
)
print(
    "included rows     =",
    candidate_rows,
)
print(
    "excluded ETC rows =",
    class_rows["ETC"],
)
print(
    "included segments =",
    candidate_segments,
)
print(
    "excluded ETC segments =",
    class_segments["ETC"],
)

print()
print("NOTE:")
print(
    "This audit does not adopt a final label mapping."
)
print(
    "It only reports observed folder/label/detail-label relationships."
)

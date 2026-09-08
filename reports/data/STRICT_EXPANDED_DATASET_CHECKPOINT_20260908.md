# Strict-Expanded Dataset Checkpoint - 2026-09-08

## 목적

STRICT-500과 MILD-645 비교의 Train sample 수 confound를 분리하기 위해,
기존 strict Train 500개에 strict-clean sample 145개를 추가한
STRICT-EXPANDED-645 control dataset을 생성했다.

Validation은 기존 strict Validation 150개를 그대로 유지했다.

## Dataset

- 입력: `data/processed/speedtransformer_smoke_v1/speed200_smoke_v1.csv`
- 출력: `data/processed/speedtransformer_strict_expanded_v1/speed200_strict_expanded_v1.csv`
- 생성기: `scripts/18_make_strict_expanded_dataset.py`
- seed: 316
- classes: WALK / BIKE / CAR / BUS / SUBWAY
- sample length: 200 speed values

## Strict Candidate / Selection Counts

script 12와 동일하게 class별 후보를 `sample_id`로 정렬한 후 하나의
`random.Random(316)` 인스턴스로 class 순서대로 shuffle했다.
기존 strict Train 100/class가 각 shuffled pool의 첫 100개와 정확히 일치하고,
feature 값도 재생성 결과와 일치함을 assert한 뒤 다음 29개를 선택했다.

| Class | Strict eligible | Unused after base 100 | Selected addition |
|---|---:|---:|---:|
| WALK | 925 | 825 | 29 |
| BIKE | 309 | 209 | 29 |
| CAR | 1,927 | 1,827 | 29 |
| BUS | 1,340 | 1,240 | 29 |
| SUBWAY | 489 | 389 | 29 |
| Total | 4,990 | 4,490 | 145 |

candidate 수는 하드코딩하지 않고 현재 strict eligible pool에서 계산했다.
모든 class에 unused candidate가 최소 29개 존재함을 assert했다.

## Strict-Clean Policy

추가 145개 sample은 다음 조건으로 생성했다.

- 기존 UID-disjoint split manifest의 Train UID만 사용
- 201 valid GPS points
- 200개 인접 timestamp gap이 모두 정확히 1 second
- 200 pairwise speeds
- 각 edge의 실제 timestamp 차이로 speed 계산
- interpolation / imputation 없음
- label-dependent speed cutoff 없음
- smoothing / clipping 없음
- 기존 strict Train 및 Validation sample과 중복 없음
- privacy-safe hashed `sample_id` 사용
- raw UID/TID와 GPS 좌표를 output에 기록하지 않음

## Final Counts

| Split | WALK | BIKE | CAR | BUS | SUBWAY | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 129 | 129 | 129 | 129 | 129 | 645 |
| Validation | 30 | 30 | 30 | 30 | 30 | 150 |
| Total | 159 | 159 | 159 | 159 | 159 | 795 |

- total rows: 159,000
- rows per sample: 200
- 모든 sample step: 정확히 0..199
- 기존 strict Train 500개: 모두 포함
- additional strict Train: 29/class, 총 145개
- Train / Validation `sample_id` overlap: 0
- Internal Test samples: 0
- mild-release samples: 0
- 모든 source policy: `strict_1s`
- 모든 추가 sample의 2-second edge 수: 0
- 모든 추가 sample의 wall-clock span: 200 seconds

## Validation Identity

기존 strict와 STRICT-EXPANDED의 Validation에 대해 다음을 assert했다.

- `sample_id` set equality: True
- sample ordering after ID sort: identical
- class / model label: identical
- 200 speed feature values: identical
- Validation samples: 30/class, 총 150

## Integrity Hash

- STRICT-EXPANDED CSV SHA256: `54CC016782D14343578E738C84B0CC2AC3CBFFE1871C3A1D005F87B272229A8B`

processed CSV는 `.gitignore`의 `data/processed/` 대상이며 Git에 강제로 추가하지 않는다.

## Internal Test

Internal Test는 dataset에 포함하지 않았고 후속 3-arm runner에서도 사용하지 않는다.

## 확인된 무결성

- 생성 스크립트 전체 assert 통과
- 별도 CSV 집계에서 159,000 rows / 795 samples 확인
- class별 Train 129 / Validation 30 확인
- 200 rows per sample 및 step 범위 0..199 확인
- non-strict source policy 0개 확인
- Internal Test 0개 확인
- 3-arm runner에서 세 Validation의 ID, feature, label exact equality 확인

## 제한사항

- 이 checkpoint는 dataset 구성 검증이며 GPU 성능 결과가 아니다.
- strict-expanded가 최적 Train 구성이라는 근거가 아니다.
- Validation 150 samples의 소규모 controlled diagnostic을 위한 control dataset이다.
- Internal Test, final AI Hub performance, paper reproduction, V3/RF 비교 및 CANOPY 서비스 성능은 평가하지 않았다.

# GPS 2-Second Release Window Audit - 2026-09-08

## 목적
strict 1-second 조건에서는 실패하지만 max-gap 2-second 조건에서
새로 확보되는 201-point window의 실제 시간 구조를 확인한다.

## 범위
- Train / Validation만 사용
- Internal Test 미사용
- 결측 보간 없음
- 실제 timestamp gap으로 speed 계산
- 모델 학습 없음

## Recovered TIDs
- Train: 818
- Validation: 94

앞선 release feasibility audit의 strict -> 2s 증가량과 일치함.

## Overall Window Structure

Train:
- non-1s edges median: 2
- non-1s edges p90: 26
- non-1s edges max: 72
- wall-clock span median: 202 sec
- wall-clock span p90: 226 sec
- wall-clock span max: 272 sec

Validation:
- non-1s edges median: 1.5
- non-1s edges p90: 29.1
- non-1s edges max: 61
- wall-clock span median: 201.5 sec
- wall-clock span p90: 229.1 sec
- wall-clock span max: 261 sec

## Class-specific Recovered Counts

Train:
- WALK: 194
- BIKE: 40
- CAR: 273
- BUS: 202
- SUBWAY: 109

Validation:
- WALK: 24
- BIKE: 2
- CAR: 29
- BUS: 27
- SUBWAY: 12

## Speed Outliers in Recovered Windows
>200 km/h edges:

Train:
- total: 31
- SUBWAY: 21
- CAR: 8
- WALK: 1
- BIKE: 1
- BUS: 0

Validation:
- total: 0

## Interpretation

Confirmed:
- 2-second release recovers a meaningful number of additional TIDs.
- Median recovered window differs only slightly from strict 200-second span.
- However the tail contains windows with many non-1s edges and substantially longer wall-clock spans.

Therefore max_gap <= 2 seconds alone may be too permissive as the final policy.

## Next
Audit recovered windows by:
- number of non-1s edges
- wall-clock span

Then define a mild-release candidate before model training.

Internal Test remains unused.

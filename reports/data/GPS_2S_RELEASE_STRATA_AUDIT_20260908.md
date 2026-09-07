# GPS 2-Second Release Strata Audit - 2026-09-08

## 목적
strict 1-second 조건에서 탈락하고 max-gap 2-second 조건에서
복구되는 window를 세분화하여 mild release 후보를 결정한다.

## 범위
- Train / Validation만 사용
- Internal Test 미사용
- 결측 보간 없음
- 모델 학습 없음

## Recovered
- Train: 818
- Validation: 94

## Gap Type
모든 non-1s edge는 정확히 2초였다.

Train:
- 2s edges: 6,313
- 1~2s edges: 0
- sub-1s edges: 0

Validation:
- 2s edges: 722
- 1~2s edges: 0
- sub-1s edges: 0

따라서 이번 데이터에서는
wall-clock span = 200 sec + number of 2-second edges
관계가 성립한다.

## Mild Release Candidate

Candidate:
- adjacent gap <= 2 seconds
- maximum 5 two-second gaps per 200-speed window
- wall-clock span <= 205 seconds

Recovered additions:

Train:
- total: 595
- WALK: 148
- BIKE: 29
- CAR: 199
- BUS: 134
- SUBWAY: 85

Validation:
- total: 73
- WALK: 19
- BIKE: 1
- CAR: 21
- BUS: 22
- SUBWAY: 10

Including strict-eligible TIDs:
- Train: 4,990 -> 5,585
- Validation: 631 -> 704

## Decision
`edge <= 5 / span <= 205 sec`를 첫 mild-release 실험 후보로 사용한다.

이는 최종 preprocessing 정책이 아니다.
Validation 기반 후속 모델 실험으로 효과를 확인해야 한다.

Internal Test는 정책 선택에 사용하지 않는다.

## Next
1. strict + mild-release window dataset 생성
2. 실제 timestamp 차이로 speed 계산
3. 결측 보간 및 label-dependent cutoff 사용하지 않음
4. UID-disjoint split 유지
5. 동일 reference-capacity 설정으로 Validation 비교

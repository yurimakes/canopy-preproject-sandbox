# CANOPY Daily Checkpoint — 2026-09-08

## ML milestone: GPS continuity preprocessing

### Strict/mild feasibility audit

- STRICT baseline은 201 valid GPS points와 정확한 1초 인접 gap으로 200개 actual-dt speed 값을 구성한다.
- strict-fail/mild-pass 후보는 gap 1초/2초, 2초 edge 최대 5개, wall-clock span 최대 205초 조건으로 감사했다.
- mild Train 후보는 WALK 148, BIKE 29, CAR 199, BUS 134, SUBWAY 85였고, 29/class를 결정적으로 선택했다.
- interpolation, imputation, label-dependent cutoff는 사용하지 않았다.

### 645 controlled comparison

- STRICT_EXPANDED와 MILD_EXPANDED를 같은 Train 크기 645(129/class)와 동일 Validation 150(30/class)으로 비교했다.
- seed 316 단일 실행에서 MILD_EXPANDED의 Validation-loss checkpoint Accuracy와 Macro F1이 STRICT_EXPANDED보다 높았다.
- 이 결과는 단일 seed, 소규모 동일 Validation diagnostic으로 제한했다.

### 645 five-seed robustness

- seed `316, 42, 2026, 5858, 1234`를 결과 확인 전에 고정했다.
- 동일 Validation에서 MILD_EXPANDED가 Accuracy와 Macro F1 모두 5/5 seed에서 높았다.
- population standard deviation 기준 aggregate는 Accuracy가 STRICT 0.698667, MILD 0.730667이고 Macro F1이 STRICT 0.697121, MILD 0.722761이었다.

### 1,500 LARGE controlled robustness

- 두 arm은 Train 1,500(300/class), Validation 150(30/class)으로 같았다.
- class별 271개 strict Train sample을 공유하고, 나머지 29개만 strict 또는 mild로 교체했다.
- 사전 고정한 같은 5개 seed에서 STRICT_LARGE Accuracy/Macro F1 mean은 0.741333/0.735219, MILD_LARGE는 0.734667/0.728799였다.
- MILD_LARGE가 두 metric 모두 높은 경우는 1/5 seed였다. 645 실험에서 관측된 MILD 우세 방향은 LARGE에서 재현되지 않았다.
- statistical significance는 주장하지 않는다.

## Current working decision

- 팀 공식 확정사항이 아닌 preproject working decision이다.
- STRICT를 현재 기본 preprocessing/baseline candidate로 유지한다.
- MILD는 coverage 확대 후보로 보관하되, 성능 개선 preprocessing으로 채택하지 않는다.
- 3초/5초 또는 새로운 gap threshold 탐색은 현재 preproject 단계에서 일시 freeze한다.
- Internal Test는 모든 preprocessing 선택과 위 실험에서 계속 미사용 상태다.

## Result boundaries

- 위 결과는 GPS/speed-only 개발 diagnostic이다.
- AI Hub 최종 성능, CANOPY 서비스 성능 또는 논문 재현 성능으로 해석하지 않는다.
- 실제 iPhone CANOPY 입력 domain과 동일하다고 간주하지 않는다.

## Unresolved

- 과거 reference-capacity diagnostic의 Validation Macro F1 0.7427과 현재 deterministic 실행 간 discrepancy 원인은 미확정이다.
- 과거 실행의 notebook/script/raw log provenance가 없어 현재 tracked 자료만으로 원인을 특정할 수 없다.

## Next priorities

### P0

CANOPY E2E용 ML interface와 Trip inference 설계로 이동한다.

### P1

실제 서비스 입력과 training preprocessing을 연결하기 위한 입력 계약과 조건을 확인한다.

### P2

MILD 재검토는 strict coverage 부족, 독립 Validation/Test 확보 또는 실제 iPhone GPS sampling 특성 확인 등 필요가 생길 때만 수행한다.

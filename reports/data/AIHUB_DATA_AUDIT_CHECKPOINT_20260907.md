# AI Hub Transportation Data Audit Checkpoint - 2026-09-07

## 범위
AI Hub 공개 Training + Validation GPS/Label 구조 감사.

## 확인된 사실

### 데이터 구조
- GPS packages: 12
- Label packages: 12
- GPS CSV: 111,850
- Label CSV: 111,850
- GPS-Label 대응 누락: 0
- 모든 GPS segment: 60 samples
- 관측 sampling interval: 1,000 ms
- UID/TID: 12,662

### 공식 Training / Validation 구조
- Training UID: 1,105
- Validation UID: 849
- UID overlap: 849
- UID/TID overlap: 6,683
- 이전 감사 기준 동일 UID/TID/SID trajectory overlap: 0

공개 Training/Validation은 UID-disjoint하지 않음.
동일 UID/TID의 서로 다른 SID가 양쪽 split에 존재함.

### 자체 UID-disjoint split v1
- Train UID: 884
- Validation UID: 110
- Internal Test UID: 111
- split 간 UID overlap: 0
- seed: 5858
- SHA256: dc5ed201dc5d81758994c27bba455bf7882cf8c99eca4e2e599da964851caae5

Internal Test는 AI Hub 공식 Test가 아님.

### GPS 결측
accuracy / latitude / longitude / altitude는 이번 감사에서 동일 row에 함께 결측됨.

- 전체 동시 결측 rows: 1,639,513
- WALK: 19.3055%
- BIKE: 8.0034%
- CAR: 6.6233%
- BUS: 6.5998%
- SUBWAY: 65.5724%
- ETC: 4.9685%

SUBWAY:
- total segments: 30,204
- complete: 6,984
- partial missing: 5,202
- fully missing: 18,018

결측값은 현재 보간, 0 대체, 삭제하지 않음.

### Segment 시간 구조
- negative-gap pairs: 1,238
- -59,000 ms pairs: 878
- -59,000 ms 878건 모두 동일 start timestamp
- 공식 Training/Validation 경계를 넘는 -59,000 ms pairs: 190
- duplicate-start groups: 878
- duplicate-start extra segments: 878

현재 확인된 것은 timestamp 중첩임.
실제 GPS 내용까지 동일한 중복인지 아직 확인되지 않음.

## 확인 필요
1. duplicate-start 878개 그룹의 실제 내용 동일 여부
2. -59초 이외 negative gap 원인
3. duplicate 처리 기준
4. 120-sample 연속 sequence 구조
5. 200-sample window 구성 가능 범위
6. 결측 GPS 처리 정책
7. 최종 5-class mapping

## 다음 작업
scripts/07_aihub_sequence_feasibility_audit.py

## Sequence Feasibility Audit

### Exact duplicate
- duplicate-start groups: 878
- group size: 모두 2
- exact-content groups: 878
- different-content groups: 0
- 공식 Training/Validation 경계 교차 exact duplicate groups: 190
- duplicate 그룹 간 coordinate missing count disagreement: 0

동일 start의 878개 그룹은 timestamp, accuracy, latitude,
longitude, altitude 기준 동일 내용으로 확인됨.

Raw 파일은 수정하지 않았으며 구조 분석에서만 동일 time slot으로 collapse함.

### Collapse 이후 시간 구조
- raw segments: 111,850
- unique time slots: 110,972
- remaining negative-gap pairs: 360

따라서 exact duplicate 878건을 설명한 뒤에도
부분 시간 중첩 360건이 남아 있으며 원인은 추가 감사 필요.

### 연속 sequence 구조
- total contiguous runs: 27,828
- maximum consecutive slots: 80
- runs >= 2 slots: 23,448
- runs >= 4 slots: 14,965

### Structural Window Capacity
UID/TID 기준:
- total: 12,662
- >=120 samples: 12,662
- >=200 sample capacity: 10,123

자체 UID-disjoint split:
- Train UID/TID: 10,063
  - >=120: 10,063
  - >=200: 8,098
- Validation UID/TID: 1,299
  - >=120: 1,299
  - >=200: 984
- Internal Test UID/TID: 1,300
  - >=120: 1,300
  - >=200: 1,041

주의:
위 결과는 timestamp 및 sample-count 기준 구조적 가능성임.
GPS coordinate 결측과 최종 preprocessing 조건은 아직 적용하지 않음.

## 다음 확인 필요
1. 남은 negative-gap 360건의 발생 구조
2. 결측을 반영한 실제 usable 120/200 window 수
3. SID stitching 정책
4. 최종 5-class mapping


## Usable Continuous Window Audit

### Remaining overlap
- exact-duplicate collapse 이후 negative overlap pairs: 360
- 공식 Training/Validation 경계 교차: 84
- shared timestamp 값이 모두 동일한 pairs: 360
- conflicting pairs: 0
- point-level overlapping observations: 10,767
- exact duplicate observations: 10,767
- conflicting observations: 0
- conflict TID: 0

Raw 데이터는 수정하지 않음.

### 실제 timestamp 기준 200-sample 연속성
- 전체 UID/TID: 12,662
- 200-sample structural window가 존재하는 TID: 10,126
- structural candidate windows: 2,157,978

좌표 결측을 보간하지 않은 상태:
- 100% valid-coordinate: 1,024,363 windows / 6,476 TIDs
- >=95% valid-coordinate: 1,353,849 windows / 7,632 TIDs
- >=90% valid-coordinate: 1,418,795 windows / 7,860 TIDs
- >=80% valid-coordinate: 1,502,660 windows / 8,182 TIDs

valid-coordinate는 finite latitude/longitude 기준이며
Speed feature 품질 또는 최종 모델 사용 가능성을 의미하지 않음.

### SUBWAY
200-sample structural TIDs: 2,425
- 100% valid window 보유: 595
- >=95% valid window 보유: 819
- >=90% valid window 보유: 892
- >=80% valid window 보유: 991

SUBWAY는 다른 folder class 대비 GPS 결측 영향이 큰 상태.

## 다음 작업
1. GPS에서 SpeedTransformer 입력용 speed feature 생성 가능성 감사
2. impossible speed / jump / zero-distance / missing-speed 패턴 확인
3. 최종 5-class mapping 확인
4. clean 200-point dataset 생성 조건 확정
5. SpeedTransformer smoke test

## Speed Feature Audit

### GPS -> derived speed
- unique timestamp points: 6,647,553
- valid coordinate points: 5,028,942
- invalid coordinate points: 1,618,611
- coordinate out-of-bounds: 0
- (0,0) coordinates: 0

1-second continuous edges:
- total: 6,620,085
- valid derived-speed edges: 4,965,744
- invalid derived-speed edges: 1,654,341
- zero-distance edges: 686,031

No coordinate imputation, clipping, smoothing, or speed filtering was applied.

### Extreme-speed diagnostic
- maximum observed derived speed: 4,646.963796 m/s
- maximum-speed folder class: SUBWAY
- >30 m/s: 26,716
- >50 m/s: 836
- >80 m/s: 154
- >100 m/s: 91

위 threshold들은 진단용이며 삭제 기준으로 확정하지 않음.

### Conservative 200-speed construction
200개의 pairwise speed를 만들기 위해
201개의 연속 GPS point를 요구하는 보수적 감사 기준 사용.

- structural windows: 2,142,993
- structural TIDs: 10,126
- all-valid-speed windows: 1,015,375
- all-valid-speed TIDs: 6,465

UID-disjoint split:
- Train: 815,293 windows / 5,145 TIDs
- Validation: 92,709 windows / 658 TIDs
- Internal Test: 107,373 windows / 662 TIDs

### SUBWAY
- continuous speed edges: 1,784,429
- valid speed edges: 607,161
- valid speed edge rate: 34.0255%
- structural 200-speed TIDs: 2,425
- all-valid 200-speed TIDs: 595

SUBWAY GPS missingness remains the main data-quality bottleneck.

### Important interpretation
이 결과는 AI Hub 데이터에서 speed sequence를 구성할 수 있는지에 대한
데이터 feasibility 결과임.

SpeedTransformer AI Hub 학습 성능 또는 CANOPY 서비스 성능은 아직 확인되지 않음.

또한 200은 SpeedTransformer의 고정 architecture requirement가 아니라
공식 replication/window-sweep에서 사용된 window-size 후보 중 하나임.

## 다음 작업
1. AI Hub folder class / label / detail_label 관계 감사
2. CANOPY 5-class mapping 후보 검증
3. label-independent speed preprocessing 정책 결정
4. SpeedTransformer smoke test

## Label Mapping Audit

### Observed raw folder / label structure
- WALK: raw label 0
- BIKE: raw label 1
- CAR: raw label 2
- BUS: raw label 3
- SUBWAY: raw label 5
- ETC: raw label 6

### Observed detail-label structure
- WALK: detail 2, 3
- BIKE: detail 4, 10
- CAR: detail 5, 12
- BUS: detail 6
- SUBWAY: detail 8
- ETC: detail 9, 11

이번 공개 Training + Validation에서는 detail 7은 관측되지 않음.

### Segment consistency
- total label segments: 111,850
- mixed raw-label segments: 0
- mixed detail-label segments: 0

이 결과는 데이터 내부 label 구조의 일관성을 의미하며
semantic label correctness를 별도로 검증한 것은 아님.

### Candidate CANOPY 5-class dataset
Candidate folder classes:
- WALK
- BIKE
- CAR
- BUS
- SUBWAY

Candidate included:
- rows: 6,476,400
- segments: 107,940

Candidate excluded ETC:
- rows: 234,600
- segments: 3,910

최종 mapping은 아직 프로젝트 확정사항이 아님.

Raw label/detail_label은 보존하고,
모델 학습용 model_label을 별도 생성하는 방향을 후보로 둠.

예:
- 0 = WALK
- 1 = BIKE
- 2 = CAR
- 3 = BUS
- 4 = SUBWAY/RAIL candidate

SUBWAY와 RAIL 명칭/정의는 추가 검증 후 확정.

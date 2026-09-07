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

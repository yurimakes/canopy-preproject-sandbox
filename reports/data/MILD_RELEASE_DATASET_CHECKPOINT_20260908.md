# Mild-Release Dataset Checkpoint - 2026-09-08

## 목적

기존 strict 1-second Train에 제한적인 2-second continuity window를 추가한
controlled Validation diagnostic용 데이터셋의 구성과 무결성을 기록한다.

이 checkpoint는 전처리 정책의 최적성이나 최종 모델 성능을 주장하지 않는다.

## Dataset

- strict 입력: `data/processed/speedtransformer_smoke_v1/speed200_smoke_v1.csv`
- mild-release: `data/processed/speedtransformer_mild_release_v1/speed200_mild_release_v1.csv`
- classes: WALK / BIKE / CAR / BUS / SUBWAY
- sample length: 200 speed values
- 기존 CANOPY UID-disjoint split을 유지하는 생성 절차를 사용했다.
- mild-release sample은 Train에만 추가했다.
- 기존 strict Validation sample은 변경하지 않았다.

## Release Policy

mild-release Train 추가분은 다음 조건을 모두 만족한다.

- adjacent gap: 1 second 또는 2 seconds
- 200개 edge 중 2-second edge: 최대 5개
- wall-clock span: 최대 205 seconds
- 각 edge의 실제 timestamp 차이(`actual dt`)로 speed 계산
- interpolation / missing-value imputation 없음
- label-dependent speed cutoff 없음
- strict 1-second window가 존재하는 TID는 추가 후보에서 제외

## Candidate / Selection Counts

생성 스크립트의 Audit 15 대조 assert와 실행 로그에서 확인한 Train candidate 수:

| Class | Mild candidate | Selected mild Train |
|---|---:|---:|
| WALK | 148 | 29 |
| BIKE | 29 | 29 |
| CAR | 199 | 29 |
| BUS | 134 | 29 |
| SUBWAY | 85 | 29 |
| Total | 595 | 145 |

selected mild Train은 seed 316을 포함한 privacy-safe deterministic hash key로
class별 29개를 선택했다.

## Final Counts

생성된 mild-release CSV를 생성 스크립트 출력과 별도로 집계한 결과:

| Split | WALK | BIKE | CAR | BUS | SUBWAY | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 129 | 129 | 129 | 129 | 129 | 645 |
| Validation | 30 | 30 | 30 | 30 | 30 | 150 |
| Total | 159 | 159 | 159 | 159 | 159 | 795 |

- total samples: 795
- total rows: 159,000
- rows per sample: 200
- 모든 sample의 step: 정확히 0..199
- strict source-policy samples: 650
- mild source-policy samples: 145
- Train / Validation `sample_id` overlap: 0
- mild 추가분의 관측된 최대 2-second edge 수: 5
- mild 추가분의 관측된 최대 wall-clock span: 205 seconds

## Validation Identity

strict 입력 CSV와 mild-release CSV에서 각각 Validation `sample_id` 150개를
정렬하여 대소문자를 구분한 exact set equality를 검증했다.

- strict Validation samples: 150
- mild-release Validation samples: 150
- exact Validation sample set identical: **True**

따라서 후속 strict/mild 비교에서 동일한 Validation 150 samples를 사용할 수 있다.

## Integrity Hashes

- strict CSV SHA256: `0BEE39C8C9FEAEF6E1F4C66CAAAD25A064D9CB4128C4F602FBEEF51FF021F9DC`
- mild-release CSV SHA256: `2C3A968D209C41ED23C89935E65D441946E2FE344E862B29E4954444B82F43AC`

processed CSV는 `.gitignore`의 `data/processed/` 대상이므로 Git에 강제로 추가하지 않는다.

## Internal Test

- mild-release CSV의 Internal Test rows/samples: 0
- 이 데이터셋 생성과 checkpoint의 Train/Validation 비교에는 Internal Test를 사용하지 않는다.

## 확인된 사실

- 생성 로그의 candidate 수가 Audit 15 기대값과 일치하고 관련 assert가 통과했다.
- class별 mild Train 선택 수는 모두 29개이며 총 145개다.
- mild-release CSV의 Train/Validation class 수, row/sample 수, step 연속성을 독립 집계했다.
- strict와 mild-release의 Validation `sample_id` 집합은 정확히 동일하다.
- Train과 Validation의 `sample_id`는 겹치지 않는다.
- 생성 코드에는 결측 보간, smoothing, clipping, label-dependent speed cutoff가 없다.
- mild 추가분 speed는 실제 timestamp 차이로 계산된다.
- raw UID/TID와 GPS 좌표는 processed CSV에 기록되지 않는다.

## 미확인 사항

- 이 checkpoint만으로 mild-release가 최적 전처리 정책인지는 확인되지 않았다.
- Validation 150 samples 밖의 일반화 성능은 확인되지 않았다.
- final AI Hub performance, paper reproduction, V3/RF 비교, CANOPY 서비스 성능은 확인되지 않았다.
- privacy-safe `sample_id`만으로 원래 UID를 역추적할 수 없으므로, UID-disjoint 유지는 생성 코드와 기존 split manifest 적용 절차를 근거로 한다.

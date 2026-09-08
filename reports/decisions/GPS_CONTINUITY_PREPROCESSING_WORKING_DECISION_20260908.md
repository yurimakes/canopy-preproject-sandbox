# GPS continuity preprocessing working decision

Date: 2026-09-08

## Status

이 문서는 팀 공식 확정사항이 아니다. 현재 CANOPY preproject ML 실험과 E2E 설계를 진행하기 위한 개인 작업 수준의 working decision이며, STRICT를 baseline candidate로 유지한다.

## A. STRICT baseline candidate

현재 기본 preprocessing 후보는 다음 조건을 유지한다.

- 인접한 valid GPS timestamp 간격이 모두 정확히 1초
- 201 valid GPS points로 200개 speed 값을 생성
- timestamp의 실제 시간 차이를 사용해 speed 계산
- interpolation 없음
- missing-value imputation 없음
- class-specific cutoff 없음
- clipping 없음
- smoothing 없음

이 정책을 현재 preproject의 기본 preprocessing/baseline candidate로 사용한다.

## B. MILD coverage candidate

MILD 후보 조건은 다음과 같다.

- 인접 gap은 1초 또는 2초
- 2초 edge는 최대 5개
- 201 points의 wall-clock span은 최대 205초
- timestamp의 실제 시간 차이를 사용해 speed 계산
- interpolation 및 imputation 없음
- class-specific cutoff 없음

MILD는 strict-fail 데이터를 일부 복구해 coverage를 확대할 수 있는 후보로 보관한다. 현재 근거만으로 이를 성능 개선 preprocessing으로 채택하지 않는다.

## Evidence

- 645-Train equal-size five-seed diagnostic에서는 MILD_EXPANDED가 STRICT_EXPANDED보다 높은 Validation Accuracy와 Macro F1을 5/5 seed에서 보였다.
- 1,500-Train equal-size five-seed diagnostic에서는 그 방향이 유지되지 않았다. MILD_LARGE가 두 metric 모두 높은 경우는 1/5 seed였고, logged mean delta는 Accuracy -0.006667, Macro F1 -0.006420이었다.
- 두 실험은 동일한 소규모 Validation을 사용한 개발 diagnostic이며 통계적 유의성을 주장하지 않는다.

따라서 small-pool 결과만을 근거로 MILD release를 기본 preprocessing으로 채택하지 않는다.

## Temporary preprocessing exploration freeze

현재 preproject 단계에서는 다음 탐색을 일시 freeze한다.

- 3초/5초 gap threshold 추가 탐색
- 새로운 arbitrary gap cutoff 도입
- Validation 점수를 본 뒤 threshold 조정
- Internal Test를 이용한 preprocessing 선택

이는 영구 종료가 아니다. 동일 Validation이 이미 여러 개발 실험에서 반복 사용됐고, 현재 일정에서는 CANOPY E2E 검증의 우선순위가 더 높기 때문이다.

## Conditions for reconsidering MILD

다음은 향후 validation criteria의 예이며 현재 구현 완료사항이 아니다.

- strict coverage가 서비스 또는 학습에 실질적으로 부족한 경우
- 더 크고 독립적인 Validation/Test에서 재검증할 수 있는 경우
- 실제 iPhone GPS sampling 특성이 확인된 경우

Internal Test는 현재 계속 봉인하며, preprocessing 선택에 사용하지 않는다.

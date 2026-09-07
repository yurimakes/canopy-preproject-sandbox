# SpeedTransformer Forward Smoke — 2026-09-07

## 목적
SpeedTransformer 모델 구조가 서로 다른 window size를 실제 forward에서 허용하는지 확인.

## Reference
- Repository: author-linked SpeedTransformer implementation
- Commit: 2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959

## 실행 결과
- window_size=100
  - input shape: (2, 100, 1)
  - output shape: (2, 5)
- window_size=120
  - input shape: (2, 120, 1)
  - output shape: (2, 5)
- window_size=200
  - input shape: (2, 200, 1)
  - output shape: (2, 5)

Result:
- ALL_FORWARD_SMOKE_OK

## 해석
현재 reference commit 기준으로 100/120/200 window 모두 모델 forward가 정상 동작함.
따라서 window_size=200은 모델 architecture 자체의 강제 고정값은 아님.

## 주의
- AI Hub 학습 성능 검증 아님
- 논문 재현 결과 아님
- 100/120/200 성능 비교 결과 아님
- CANOPY 서비스 성능 아님
- 최종 window size는 아직 확정하지 않음

## 다음 작업
1. data_utils.py의 실제 window 생성/padding 방식 확인
2. GPS→speed preprocessing 및 filtering 방식 확인
3. 데이터 split 방식 확인
4. CANOPY UID-disjoint split을 보존하는 adapter 설계
5. AI Hub smoke training

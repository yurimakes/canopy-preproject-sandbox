# CANOPY Daily Checkpoint — 2026-09-07

## 현재 단계
프로젝트 시작 전 AI Hub 데이터 감사 및 로컬 개발환경 초기화.

## 확인된 작업

### 개발환경
- Python 가상환경 `.venv` 생성 및 활성화
- Git 저장소 초기화
- 기본 브랜치 `main`
- `.gitignore` 구성
  - Raw AI Hub 데이터 제외
  - GPS/UID manifest CSV 제외
  - 모델 checkpoint 및 output 제외
  - 환경변수/secret 제외
- `.gitattributes` 구성
- README 생성

### Git
- 초기 커밋: `3d35ea2`
- 원격 저장소 `origin` 연결
- `main` 최초 push 성공
- `main`이 `origin/main`을 tracking
- 마지막 확인 기준 working tree clean

## AI Hub 데이터 감사 현재 상태
- GPS package 12개 확인
- Label package 12개 확인
- ZIP 손상 0건
- GPS CSV 111,850개
- Label CSV 111,850개
- GPS/Label 대응 누락 0건
- 공식 Training/Validation trajectory overlap 0
- 공식 Training/Validation UID overlap 849
- 자체 UID-disjoint split v1 생성
- Train UID 884
- Validation UID 110
- Internal Test UID 111
- UID split 간 overlap 0
- 모든 GPS segment는 60 points
- sampling interval은 관측 결과 1초
- Unique UID/TID 12,662
- 모든 TID에 2개 이상 segment 존재
- TID 내부 class change 관측 0건

## 아직 확인 필요한 사항
- GPS 4개 필드 동시 결측 1,639,513건의 발생 패턴
- segment gap `-59000 ms` 발생 원인
- 연속 SID 연결 가능 조건
- 120초 / 200-sample window 구성 기준
- 최종 5-class label mapping
- GitHub 저장소 visibility는 터미널 로그만으로 확인하지 않음

## 다음 작업
1. segment gap 및 SID 순서 구조 감사
2. 결측 GPS segment 패턴 분석
3. 연속 sequence 생성 가능성 검증
4. window 정책 확정
5. 이후 V3 / SpeedTransformer 동일 UID split 실험 준비

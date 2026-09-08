# Strict Run Discrepancy Audit - 2026-09-08

## 목적

과거 수동 reference-capacity diagnostic의 STRICT 결과와 현재 deterministic
script 17 Colab 결과가 다른 원인을 저장소 코드와 기록으로 확인 가능한 범위에서 감사한다.

이 감사는 현재 deterministic runner의 결과나 설정을 변경하지 않는다.

## 관측 결과

| Result | Past manual diagnostic | Current script 17 Colab run | Current - Past |
|---|---:|---:|---:|
| Best Validation-loss epoch | 31 | 14 | -17 |
| Validation loss | 0.667003 | 0.933921 | +0.266918 |
| Validation accuracy | 0.740000 | 0.693333 | -0.046667 |
| Validation Macro F1 | 0.742700 | 0.684480 | -0.058220 |

- 과거 값의 tracked 근거: `reports/ml/SPEEDTRANSFORMER_REFERENCE_CAPACITY_DIAGNOSTIC_20260908.md`
- 과거 보고서 commit: `848353c`
- 현재 runner: `scripts/17_compare_strict_vs_mild_release.py`
- 현재 runner commit: `37fe6b3`
- 현재 Colab 결과는 이 작업에 제공된 별도 실행 결과이며 저장소에 tracked raw log는 없다.
- 제공된 정보에 따르면 현재 script 17 STRICT 결과는 두 번 완전히 동일하게 재현됐다.

## 공통으로 기록된 조건

과거 보고서와 현재 script 17에서 다음 조건은 동일하게 확인된다.

- strict dataset: Train 500, Validation 150
- UID-disjoint split 유지 및 Internal Test 미사용
- seed 316
- window size 200, feature size 1, classes 5
- d_model 128, nhead 8, layers 4, kv_heads 4, dropout 0.1
- batch size 64
- learning rate 2e-4, weight decay 1e-4, gradient clip 1.0
- max epochs 50, patience 7
- checkpoint criterion: Validation loss
- reference commit: `2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959`

동일하게 기록된 항목만으로 두 실행의 전체 실행 경로가 동일했다고 판단할 수는 없다.

## 현재 script 17에서 확인된 실행 방식

- CSV sample을 `sample_id` 오름차순으로 재정렬하여 dataset index를 구성한다.
- strict CSV의 원래 Train sample 순서는 `sample_id` 오름차순이 아니다.
- 각 arm 시작 시 Python, NumPy, PyTorch 및 CUDA seed를 316으로 reset한다.
- `torch.use_deterministic_algorithms(True)`를 사용한다.
- cuDNN deterministic=True, benchmark=False를 사용한다.
- Train DataLoader 전용 `torch.Generator`를 seed 316으로 초기화한다.
- Train loader는 shuffle=True, workers=0이다.
- StandardScaler는 정렬된 해당 arm Train feature에만 `fit_transform`한다.
- Validation에는 같은 arm scaler의 `transform`만 적용한다.
- reference `TrajectoryTransformer`를 import한 뒤 reference wrapper와 같은 Xavier weight / zero bias 초기화를 명시적으로 수행한다.
- model 생성은 각 arm의 seed reset 이후 수행된다.
- CrossEntropyLoss와 AdamW를 사용한다.
- AdamW에는 learning rate와 weight decay만 명시하며 나머지는 설치된 PyTorch 기본값을 사용한다.
- 매 batch `optimizer.zero_grad(set_to_none=True)`를 수행한다.
- backward 후 gradient norm을 1.0으로 clip하고 optimizer step을 수행한다.
- AMP/autocast는 사용하지 않는다.
- Validation loss가 strictly 낮아질 때 checkpoint state를 갱신한다.

## 과거 실행에서 확인되지 않는 항목

commit `848353c`에는 결과 Markdown 한 파일만 추가되어 있다. Git history 검색에서도
0.7427 값을 생성한 notebook, training script 또는 raw log는 발견되지 않았다.
따라서 과거 실행에 대해 다음은 확인할 수 없다.

- CSV 원래 순서를 유지했는지, `sample_id`로 정렬했는지
- Train DataLoader의 실제 index 순서와 batch permutation
- 별도 DataLoader generator 사용 여부 및 generator seed
- seed reset의 정확한 위치와 model initialization 시점
- strict CSV의 당시 byte hash와 현재 입력 content의 exact equality
- StandardScaler에 전달한 sample/row ordering과 실제 fit content
- optimizer의 전체 argument 및 당시 PyTorch 기본값/버전
- `zero_grad(set_to_none=True)` 사용 여부
- gradient clipping 호출 위치
- deterministic algorithm 및 cuDNN flag 설정
- AMP/autocast 사용 여부
- dependency 및 CUDA/cuDNN 버전
- Xavier 초기화를 같은 시점과 같은 RNG state에서 수행했는지
- epoch별 batch 순서 및 전체 epoch log

reference repository의 일반 학습 코드는 이 항목 중 일부를 구현하지만, 과거 수동
diagnostic이 그 코드 경로를 그대로 사용했다는 tracked 증거는 없다.

## 결론

### 확인된 차이

- 두 결과의 best epoch, Validation loss, accuracy, Macro F1이 다르다.
- 현재 runner는 strict CSV의 원래 sample 순서를 사용하지 않고 `sample_id`로 정렬한다.
- 현재 runner의 ordering, RNG reset, DataLoader generator, initialization, optimizer step,
  deterministic flag는 코드로 감사할 수 있다.
- 과거 결과 commit에는 결과 보고서만 있고 동일 수준의 실행 provenance가 없다.

### 원인 미확정

sample ordering, batch ordering, RNG consumption, initialization timing, AMP 및 library
version 중 어느 하나가 차이를 만들었는지 현재 tracked 자료만으로 특정할 수 없다.
과거 notebook/script/raw log가 없으므로 0.7427과 0.684480 차이의 원인은 미확정이다.

현재 deterministic runner가 두 번 동일한 결과를 냈다는 사실은 현재 실행의 반복성을
지지하지만, 과거 결과와의 차이 원인을 자동으로 설명하지는 않는다.

## 다음 확인 방법

과거 실행 환경 또는 notebook을 확보할 수 있다면 다음을 동일 순서로 기록해 재실행한다.

1. CSV SHA256 및 sample index 순서 hash
2. epoch별 DataLoader permutation hash
3. scaler mean/scale
4. model initial state hash
5. Python/PyTorch/CUDA/cuDNN/scikit-learn 버전과 AMP 상태
6. optimizer 전체 defaults와 seed reset 위치

Internal Test는 이 discrepancy 확인에도 사용하지 않는다.

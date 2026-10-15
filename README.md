# InfGen: 논문 근거 기반 학습·추론 재현

이 저장소는 Xiuyu Yang, Shuhan Tan, Philipp Krähenbühl의 **Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation (ICCV 2025)** 구현입니다. UCLA의 SceneStreamer와는 별개 연구입니다.

- 출발점: [공식 OrangeSodahub/InfGen](https://github.com/OrangeSodahub/InfGen), 커밋 `68f643c0cd464e908fac0ef79f82d55b5e6edc90`.
- 포크: [seulbinHwang/InfGen](https://github.com/seulbinHwang/InfGen), 작업 브랜치 `infgen`.
- 기준 자료: 사용자가 지정한 [논문 PDF 및 부록](https://orangesodahub.github.io/InfGen/assets/infgen_preprint.pdf).
- 원본 README와 인용 정보는 [UPSTREAM_README.md](docs/UPSTREAM_README.md)에 보존했습니다. 원저자 코드와 MIT 라이선스를 유지합니다.

**목표는 논문과 근거를 추적할 수 있는 실행 가능한 재현입니다.** 무오류, 논문 수치 달성, 미실행 환경의 동작을 보증하지 않습니다. 합성 데이터 테스트는 실제 WOMD 학습·평가를 대체하지 않습니다.

## 무엇을 구현하고 검증하는가

공개된 약 1,140만 파라미터 모델을 출발점으로 사용합니다. 0.5초 움직임 토큰과 객체 상태 토큰을 이용하여 기존 객체의 움직임·유지·제거를 예측하고, 새 객체를 순차적으로 생성하다가 BEGIN MOTION에서 다음 시간 단계로 넘어갑니다. 지도 인코더, 상대 위치 어텐션, 점유 격자, 위치→방향 생성, 객체 크기·종류 예측을 포함합니다.

기존 구현의 경로·설정·빈 손실 처리·배치 인덱스·마스킹·추론 수명주기 문제를 보완했습니다. `configs/paper_reproduction.yaml`의 `Model.paper_objective: true`는 논문 식 8의 손실 계수, 제어 클래스 가중치, 거리순 생성, 명시된 추론 규칙을 적용합니다. 기존 설정 파일은 비교용으로 남아 있으며, 논문 설정과 동일하다고 주장하지 않습니다.

근거와 판단은 다음 문서에서 계속 기록합니다.

| 기록 | 내용 |
| --- | --- |
| [논문 대조표·판단 기록](docs/paper_audit.md) | 절·페이지·수식별 요구사항, 논문/공개 코드의 불일치, 미명시 설정과 선택 이유 |
| [모델 수정 기록](docs/model_audit.md) | 인과 마스크, 빈 학습 대상, 객체 유지·제거, 샘플링과 손실 처리 |
| [데이터 수정 기록](docs/data_audit.md) | 전처리·유효성·제어 토큰, 과거/미래 분리, 데이터 경로·메타데이터 처리 |
| [검증 기록](docs/verification.md) | 실행 명령·결과·검증 환경과 미검증 범위 |

## 설치

Python 3.10을 권장합니다. 전처리된 시나리오의 학습·NPZ 추론에는 TensorFlow/Waymo 평가 패키지가 필요하지 않습니다. 아래 설치는 현재 시스템의 PyTorch 빌드를 사용하며, NVIDIA 학습 환경에서는 먼저 필요한 CUDA용 PyTorch를 선택하십시오.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.5.1 setuptools wheel
uv pip install --python .venv/bin/python --no-build-isolation -r requirements-dev.txt
source .venv/bin/activate
```

`torch-cluster`/`torch-scatter`는 PyTorch 버전과 맞는 확장 모듈이 필요합니다. [PyG 공식 설치 안내](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)에 맞는 휠을 사용하거나 C++ 도구가 있는 환경에서 빌드하십시오. 이 Mac에서 빌드할 때는 다음 SDK 설정이 필요했습니다.

```bash
export SDKROOT="$(xcrun --show-sdk-path)"
export CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
uv pip install --python .venv/bin/python --no-build-isolation torch-cluster==1.6.3 torch-scatter==2.1.2
```

원본 Linux/CUDA/TensorFlow/Waymo 전체 환경은 `environment.yml`에 있습니다. 실제 raw TFRecord 전처리와 공식 평가에는 별도 전체 환경이 필요합니다. 모든 패키지가 들어 있는 원본 환경을 이 Mac에서 실행 검증했다는 뜻은 아닙니다.

## 데이터 준비

논문 데이터는 WOMD입니다. 데이터 이용 권한에 따라 [Waymo에서 scenario protocol 데이터를 확보](https://waymo.com/open/download/)하고 원본 전처리를 사용하십시오. 원본 README는 WOMD v1.2.1을 명시합니다. `python data_preprocess.py --help`와 `scripts/data_preprocess.sh`에서 입력·출력·분할 옵션을 확인할 수 있습니다.

기본 디렉터리 구조:

```text
data/waymo_processed/
  training/*.pkl
  validation/*.pkl
  meta_infos.json             # 선택 사항
```

`--data_root`로 다른 위치를 지정할 수 있습니다. 메타데이터가 없으면 모든 `.pkl`을 읽고 경고를 남깁니다. 메타데이터가 있으면 원본 객체 수 필터를 적용합니다. 원본 전처리가 객체 크기를 미래 프레임까지 평균하여 저장한 데이터는 새 전처리로 다시 만드는 것이 안전합니다. 생성된 객체의 학습 표적에는 미래 관측을 사용할 수 있지만, 초기 추론 입력에는 사용할 수 없습니다.

## 학습

논문은 총 배치 8, A5000 8개, AdamW, 학습률 0.0005, cosine schedule을 사용합니다. 기본 설정은 GPU당 배치 1 × 8개입니다.

```bash
python train.py --config configs/paper_reproduction.yaml \
  --data_root /path/to/waymo_processed --devices 8 \
  --save_ckpt_path output/infgen
```

저장 위치에는 체크포인트, CSV 로그, 실제 적용된 `resolved_config.yaml`이 생성됩니다. 외부 로깅은 기본으로 사용하지 않습니다. `--wandb`를 명시한 경우만 선택적으로 활성화됩니다.

```bash
# 옵티마이저·스케줄러·학습 위치까지 재개
python train.py --data_root /path/to/waymo_processed --devices 8 \
  --ckpt_path output/infgen/last.ckpt --save_ckpt_path output/infgen

# 명시적인 부분 가중치 초기화; 학습 재개와 다름
python train.py --data_root /path/to/waymo_processed --devices 8 \
  --pretrain_ckpt /path/to/pretrained.ckpt --save_ckpt_path output/initialized
```

`--accelerator cpu --devices 1`, `--num_workers`, `--batch_size`, `--max_epochs`, `--max_steps`, `--limit_train_batches`, `--limit_val_batches`로 검증용 실행을 제한할 수 있습니다. 자원이나 배치를 줄인 실행을 논문의 원래 실험과 동일하게 간주하지 않습니다.

## 추론

학습 체크포인트에 저장된 모델 설정을 사용하고 가중치를 엄격하게 로딩합니다. 입력은 전처리된 시나리오 하나입니다. 현재 공개 decoder의 제약에 맞추어 한 번에 한 시나리오를 처리합니다.

```bash
python infer.py --checkpoint output/infgen/last.ckpt \
  --input /path/to/waymo_processed/validation/SCENARIO_ID.pkl \
  --output output/rollout.npz --future_steps 300 --device cuda --seed 2024
```

`--future_steps`는 10Hz 미래 프레임 수이며 5의 배수입니다. 300이면 과거 11프레임을 포함한 총 311프레임, 80이면 총 91프레임을 출력합니다. `--device cpu`도 지원합니다. `--disable_insertion`은 신규 추가를 끄는 진단 옵션입니다. 이 옵션만으로 WOSAC의 전체 평가 프로토콜을 구현했다고 간주하지 않습니다.

NPZ에는 `pred_traj` `[객체, 프레임, 2]`, `pred_head`, `pred_valid`, `pred_state`, `pred_shape`, `pred_type`, 객체 ID·토큰·시드 등이 들어 있습니다. `numpy.load(..., allow_pickle=False)`로 열 수 있습니다. 같은 출력 파일을 덮어쓰지 않습니다.

전체 평가 환경에서는 `run.py --validate --ckpt_path ...`가 원본 평가 경로를 사용합니다. `--test`는 지표 계산 없이 원본 형식의 rollout을 저장합니다. 실제 WOMD 및 TensorFlow 기반 지표 계산은 별도 검증 대상입니다.

## 재현 검증

```bash
python -m pytest -q
ruff check .
python -m compileall -q infgen run.py train.py val.py infer.py data_preprocess.py
python scripts/verify_reproduction.py
```

마지막 명령은 별도 임시 디렉터리에 합성 데이터를 만들고 **실제 전체 모델 학습 → 체크포인트 저장 → 재개 → 엄격한 재로딩 → 8초/30초 추론 → 미래 입력 변경 불변성**을 검사합니다. 실제 데이터나 학습된 공개 체크포인트 없이 코드 경로를 검증하기 위한 절차입니다. 결과와 한계는 [검증 기록](docs/verification.md)에 명시합니다.

## 논문에 없어서 선택한 요소와 알려진 차이

| 요소 | 구현 결정과 이유 |
| --- | --- |
| 위치 토큰 수 | 논문 표 4의 1,849개와 공개 코드의 1,961개가 충돌합니다. 반경 75m·간격 3m의 공개 원형 격자와 파라미터 호환성을 유지했습니다. 임의의 112개 셀 제거에는 근거가 없습니다. |
| 학습 기간·dropout·seed | 논문에 확정값이 없습니다. 원본의 32 epochs·dropout 0.1·seed 2024를 초기 선택으로 유지하고 설정으로 공개합니다. |
| optimizer 세부값·shape loss | 미명시된 AdamW 세부값은 PyTorch 기본값, shape 회귀는 원본 MSE를 사용합니다. |
| 샘플링 세부값 | 미명시된 위치 K=10·움직임 K=5·온도 1은 원본 선택을 유지합니다. 제어 토큰은 모드에 허용된 선택지 안에서 샘플링합니다. |
| MLP 층수·가중치 호환성 | 논문 모드에서는 부록의 단일 선형 fusion과 3개 선형층 shape head를 적용합니다. 공개 코드의 3층 fusion·2층 shape head와 달라, 원본 체크포인트는 명시적인 부분 초기화 후 바뀐 부분을 다시 학습해야 합니다. |
| 점유 격자 표현 | 논문 텍스트의 셀별 특징 행렬과 공개 코드의 query별 압축 토큰 표현이 다릅니다. 식 6의 효율화 변환 Γ에 대한 구현 해석으로 공개 압축 표현을 유지하고, 동일한 텐서 구조라고 주장하지 않습니다. |
| 추가 보조 손실·offset | 논문 식 8에 없는 map/occupancy/offset 보조 손실과 학습되지 않는 offset 보정을 논문 모드에서 제외합니다. 점유 격자 입력 자체는 유지합니다. |
| 과거 입력의 경계 | 과거 1.1초 이후를 관측하지 않은 것으로 처리해 마지막 과거 토큰에 미래 REMOVE 정보가 섞이지 않도록 합니다. 미래에 처음 나타나는 객체가 역외삽으로 과거에 생기는 것도 막습니다. |
| 단일 토큰 수명·중간 가림 | 논문이 모든 경계 사례를 정하지 않습니다. 단일 유효 토큰은 ADD를 우선하고, 중간 가림 등 남은 원본 가정은 데이터 감사 문서에 명시합니다. |
| 수치 재현 | 실제 WOMD 전체 학습·다중 GPU·공식 지표 점수는 이 로컬 검증과 별개입니다. 논문의 realism 점수를 재현했다고 주장하지 않습니다. |

변경할 때는 관련 감사 문서에 **논문 근거 → 코드 선택 → 선택 이유 → 검증 결과**를 함께 기록합니다. 알려진 차이를 숨기거나 테스트 통과를 무오류의 증명으로 취급하지 않습니다.

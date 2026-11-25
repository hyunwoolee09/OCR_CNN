# OCR CNN

MathWriting 손글씨 기호 92종을 분류하는 Keras CNN 프로젝트다. 수식 OCR의 기호 분할 결과를 입력받아 LSTM에 전달할 기호 확률을 계산한다.

## 설치와 추론

Python 3.11 이상에서 다음 명령으로 설치한다.

```bash
pip install -r requirements.txt
```

```python
import numpy as np
from PIL import Image
from cnn import crop, load_cnn, predict_probs
from symbols import CLASSES

model = load_cnn()
images = np.stack([crop(Image.open("symbol.png"))])
probs, features = predict_probs(model, images, with_features=True)
best = int(probs[0].argmax())
print(CLASSES[best], float(probs[0, best]))
```

`crop()`은 흰 배경의 개별 기호 이미지를 받는다. CNN 입력은 `(N, 1, 48, 48)` float32이며 배경은 0, 획은 1이다. `predict_probs()`는 입력 순서대로 `(N, 92)` float32 확률을 반환한다. 각 행의 확률 합은 1이다. `with_features=True`이면 `(N, 256)` 특징도 함께 반환한다. 클래스 순서는 `symbols.py`에서 관리하며 저장 모델을 읽을 때 검사한다. 위치 정보는 분할 결과에서 별도로 계산해 함께 전달한다.

기본으로 불러오는 파일은 `results/physics_cnn_92_48_shift_elastic.keras`다. 1픽셀 이동 모델을 불러오려면 `load_cnn("results/physics_cnn_92_48_shift.keras")`를 사용한다.

## 학습

```bash
python train_shift_augmented.py
python train_elastic_augmented.py
```

첫 명령은 1픽셀 이동, 둘째 명령은 1픽셀 이동과 약한 탄성 변형을 적용한다. 학습 데이터에만 증강을 적용한다. 탄성 변형은 기호마다 50% 확률로 가로와 세로 각각 최대 1.5픽셀 이동하는 부드러운 변형이다.

`data/physics_symbols_92_48.npz`에 이미지, 라벨, 샘플 ID, 클래스 순서와 학습·검증·시험 인덱스가 들어 있다. 학습 1,856장, 검증 462장, 시험 574장이다. 포함된 캐시로 바로 학습할 수 있다. 원본에서 캐시를 다시 만들려면 `data/mathwriting-2024.tgz`가 추가로 필요하다. 원본 아카이브는 MathWriting 배포처에서 받는다. 데이터의 출처와 이용 조건은 [MathWriting 공식 저장소](https://github.com/google-research-datasets/mathwriting)를 참고한다.

## 확인한 성능

동일한 손글씨 시험 이미지 574장으로 평가했다. 20에포크를 학습하고 검증 정확도가 가장 높은 가중치를 선택했다.

| 학습 방법 | 검증 정확도 | 시험 정확도 |
|---|---:|---:|
| 1픽셀 이동 | 70.13% | 70.38% (404/574) |
| 1픽셀 이동과 탄성 변형 | 70.35% | 70.91% (407/574) |

상세 이력은 `results/physics_results_92_48_shift_keras.json`과 `results/physics_results_92_48_shift_elastic_keras.json`에 기록했다.

## 파일 구성

- `cnn.py`와 `symbols.py`에 CNN 구조, 전처리, 추론 함수와 92개 클래스 순서가 있다.
- `train_physics.py`에 데이터 준비와 공통 학습·평가 과정이 있다.
- `train_shift_augmented.py`와 `train_elastic_augmented.py`에 증강 실험 실행부가 있다.
- `prepare_symbols.py`에 MathWriting 획을 이미지로 만드는 함수가 있다.
- `prepare_rendered_symbols.py`와 이름에 `rendered`가 들어간 파일은 인쇄체 추가 실험 자료다. 위 두 손글씨 실험은 인쇄체 자료 없이 학습했다.
- `data/`와 `results/`에는 현재 데이터와 실험 산출물이 있다. 32픽셀·95종 데이터 및 `.pt` 파일은 이전 실험 산출물이다.

작은 점의 펜 움직임이 전처리 과정에서 크게 확대되는 사례가 있다. 확대 배율 상한을 적용하는 개선은 아직 진행하지 않았다.

## 수식 한 개의 기호별 출력

`symbol_relations.formula_rows(cnn, relations, crops, boxes)`는 기호를 중심 x, 다음 y 순으로 정렬한다. 반환값 `rows`는 float32 배열이며 크기는 (기호 수, 388)이다. 0~255열은 CNN 특징, 256~347열은 기호 확률, 348~355열은 수식 경계 기준 위치값, 356~387열은 들어오는 관계 16개와 나가는 관계 16개다. 기호가 하나면 관계 메시지는 모두 0이다. 반환값 `order`로 출력 행에 대응하는 원래 기호를 찾는다.

```python
from cnn import load_cnn
from symbol_relations import formula_rows, load_relations

result = formula_rows(load_cnn(), load_relations(), crops, boxes)
rows = result["rows"]
```

`crops`는 (N, 1, 48, 48), `boxes`는 같은 순서의 (왼쪽, 위, 오른쪽, 아래) 좌표다. 수식 전체의 대표 기호 높이로 CROHME 획을 렌더링해 작은 점의 크기를 보존한다.

`python train_relations.py`는 학습된 CNN을 고정하고 CROHME 2013의 주석 달린 기호 쌍으로 방향 있는 7개 관계와 메시지 임베딩을 학습한다. 메시지가 실제 관계 분포를 나타내도록 보조 손실을 사용한다. 검증 macro-F1으로 가중치를 선택하고 공식 시험 분할에서 지원되는 기호와 정답 상자를 사용해 평가한다. GitHub Actions의 **Train symbol relations** 실행 결과에서 가중치와 JSON 성능 보고서를 내려받아 `results/`에 둘 수 있다.

### 서버 시험 결과

기존 CNN 가중치를 고정하고 CROHME 2013에서 지원되는 수식 4,500개로 관계 분류기를 12에포크 학습했다. 검증 수식 843개의 관계 macro-F1으로 선택한 12번째 가중치가 저장소의 `results/symbol_relations.weights.h5`다.

정답 기호 영역을 사용한 독립 시험 수식 407개에서 방향 있는 기호 쌍 43,028개를 평가했다. 관계 정확도는 84.52%, 관계 7종 macro-F1은 0.4093이다. 시험 쌍의 93.20%가 관계 없음이므로 정확도만으로 성능을 판단하기 어렵다. 위첨자 재현율은 74/215, 아래첨자 재현율은 77/277이었다. 기호 분류와 관계 분류의 지표는 각각 `results/symbol_relations_metrics.json`에 있다.

[서버 학습 및 출력 검사 기록](https://github.com/hyunwoolee09/OCR_CNN/actions/runs/45530929446)

### 가상 수식으로 관계 분류 학습하기

`python train_relations.py --synthetic pairs`는 같은 손글씨 특징 쌍에 실제 학습 수식에서 수집한 나란한 배치, 위첨자 배치, 아래첨자 배치를 적용한다. `--synthetic context`는 추가로 합성 세트의 약 절반에 +와 다음 문자를 배치해 가까운 기호 사이의 관계 없음도 연습한다. CNN을 고정하므로 이미 계산된 손글씨 특징을 재사용한다. 각 실험은 매 에포크 실제 학습 수식 3,375개와 합성 수식 1,125개를 사용한다. 검증과 시험은 실제 CROHME 수식이다.

| 학습 방식 | 검증 macro-F1 | 시험 macro-F1 | 시험 위첨자 정밀도 / 재현율 | 시험 아래첨자 정밀도 / 재현율 |
|---|---:|---:|---:|---:|
| 실제 수식 | 0.394 | 0.409 | 14.1% / 34.4% | 14.8% / 27.8% |
| 실제 + 기호 쌍 | 0.371 | 0.384 | 11.0% / 49.3% | 13.0% / 38.6% |
| 실제 + 기호 쌍과 주변 기호 | 0.376 | 0.383 | 10.9% / 33.5% | 14.2% / 31.4% |

가상 수식 실험 가중치는 `load_relations("results/symbol_relations_pairs.weights.h5")` 또는 `load_relations("results/symbol_relations_context.weights.h5")`로 선택한다. [서버 비교 실행](https://github.com/hyunwoolee09/OCR_CNN/actions/runs/45533932593)

### 관계 없음 샘플링과 기호 쌍의 상대 위치 실험

1. `python train_relations.py --variant hard`는 각 학습 배치에서 뽑던 관계 없음 쌍의 절반을 기존 모델이 위·아래첨자로 혼동할 확률이 높은 쌍으로 채운다. 나머지는 무작위로 뽑는다.
2. `python train_relations.py --variant hard_geometry`는 같은 샘플링을 사용하고, 기준 기호 오른쪽 끝과 상대 기호 중심 사이의 가로 거리(기준 높이 단위), 두 기호의 높이 비율, 작은 기호 높이 대비 세로로 겹치는 비율을 관계 분류기에 추가한다.

두 모델은 같은 CROHME 학습 수식으로 각각 12에포크 학습했다. 검증 수식 843개의 macro-F1으로 가중치를 고른 뒤, 같은 시험 수식 407개의 정답 영역에서 방향 있는 기호 쌍 43,028개를 평가했다.

| 모델 | 검증 macro-F1 | 시험 정확도 | 시험 macro-F1 | 위첨자 정밀도 / 재현율 | 아래첨자 정밀도 / 재현율 |
|---|---:|---:|---:|---:|---:|
| 기존 | 0.3941 | 84.52% | 0.4093 | 14.1% / 34.4% | 14.8% / 27.8% |
| 1단계 | 0.3363 | 82.20% | 0.3635 | 23.1% / 2.8% | 30.7% / 9.7% |
| 1+2단계 | 0.6485 | 94.15% | 0.6301 | 70.1% / 51.2% | 63.5% / 38.3% |

`load_relations()`는 검증 macro-F1이 가장 높은 1+2단계 가중치를 불러온다. 세 모델 모두 기호당 관계 메시지 32개와 결합 출력 388개를 제공한다. [서버 실행과 시험 보고서](https://github.com/hyunwoolee09/OCR_CNN/actions/runs/45553328021)

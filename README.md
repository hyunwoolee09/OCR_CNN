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

## 수식 인코더

`python train_formula_encoder.py`는 CROHME 2013의 주석 달린 손글씨 수식으로 관계 분류기와 2층 양방향 LSTM 인코더를 학습한다. 기존 기호 CNN의 가중치를 고정하고 기호 특징 256개와 확률 92개를 사용한다. 수식 단위 배율로 기호를 그리고, 중심 x와 y 순으로 정렬해 수식 경계 기준 위치값 8개를 계산한다. 기호 쌍의 7가지 관계 확률을 들어오는 방향과 나가는 방향으로 각각 16개씩 요약한다. 각 행의 388개 값을 256개로 투영한 뒤 양방향 인코더에 입력한다.

학습 정답은 CROHME의 기호 라벨과 MathML의 방향 있는 관계다. 디코더 없이 인코더를 학습하기 위해 각 위치에서 문맥을 반영한 기호를 예측하는 보조 분류기를 사용한다. 기호 정답 상자가 주어지는 조건에서 평가하며, 검증 기호 정확도와 관계 macro-F1의 평균으로 가중치를 선택한다. 테스트는 공식 시험 분할 중 지원하는 기호·수식 표본을 사용한다.

서버 학습은 GitHub Actions의 **Train formula encoder**에서 실행할 수 있다. 완료 후 `trained-formula-encoder` 결과물에 있는 두 파일을 `results/`에 두면 아래처럼 사용할 수 있다. `crops`는 기호별 CNN 입력이고, `boxes`는 같은 순서의 원본 이미지 좌표 (왼쪽, 위, 오른쪽, 아래)다.

```python
from cnn import load_cnn
from formula_encoder import encode_formula, load_encoder

out = encode_formula(load_cnn(), load_encoder(), crops, boxes)
memory = out["memory"]  # (N, 512), 각 방향 256개
rows = out["rows"]      # (N, 388)
order = out["order"]    # 각 출력 행에 대응하는 원래 기호 번호
```

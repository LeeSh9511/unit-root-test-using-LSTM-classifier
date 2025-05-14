##LSTM 분류기를 이용한 단위근 개수 예측과 비이론적 시계열에의 적용 평가
(Predicting the Number of Unit Roots Using LSTM Classifiers and Evaluating Applicability to Non-Theoretical Time Series)

- 본 연구에서는 대표적인 단위근 검정인 ADF 검정을 딥러닝 방법론인 LSTM 구조를 통해 구현함으로써 거기서 통계적 인사이트를 얻고자함.

### 1️⃣ simulation data 생성 구조.
- **정상 시계열**: `Z_t = φ₁Z_{t−1} + φ₂Z_{t−2} + ε_t`, `ε_t ~ WN(0,1)`,  φ₁, φ₂~ i.i.d U(-0.9, 0.9) under stationarity condition.
- **단위근 시계열**: 위 정상 시계열에 대해 누적합 연산  
  - 1차 누적합 → 단위근 1개  
  - 2차 누적합 → 단위근 2개
- **혼합 시계열 (비이론적)**: 정상+단위근을 앞/뒤 절반으로 결합하여 생성  
  - 앞 50% 정상 + 뒤 50% 단위근 (또는 그 반대)

총 100길이 시계열로, 각 실험 세트에 대해 10,000개 이상의 샘플을 생성

---

### 2️⃣ 분류기 구조 및 학습 설정

- 입력 벡터: 길이 100의 시계열
- 모델 구조: LSTM(hidden size=30) → Dense Layer(Softmax)
- 분류 방식:
  - **Binary**: 정상 vs. 단위근 여부
  - **3-Class**: 단위근 개수 (0, 1, 2)
- 손실 함수: Categorical Cross-Entropy
- 최적화 알고리즘: Adam
- Batch size: 1000
- 최대 Epoch: 200
- Early Stopping: Validation loss 10회 미개선 시 학습 종료

---

### 3️⃣ 성능 비교 대상 및 지표

- 비교 대상: ADF (Augmented Dickey-Fuller) 검정
- 평가 지표:
  - **정확도 (Accuracy)**
  - **경험적 사이즈 (Empirical Size)**: 단위근 시계열을 정상으로 오분류한 비율
  - **경험적 검정력 (Empirical Power)**: 정상 시계열을 정상으로 올바르게 분류한 비율

---

### 4️⃣ 실험 조건 다양화

- 단위근 시계열 비율 `p`를 변화시켜 혼합 테스트셋 구성  
  - `p ∈ {0.15, 0.3, 0.45, 0.6, 0.75, 0.9}`
- 각 `p`에 대해 10,000개 샘플 생성
- ADF는 유의수준 1%, 5%, 10%에서 각각 평가

---

### 📌 참고 이미지

- 학습 손실/정확도 그래프: `figures/loss_acc.png`
- 단위근 비율별 정확도 비교: `figures/accuracy_p.png`
- 3-Class 분류 혼동 행렬: `figures/confusion_matrix.png`

> 📁 모든 결과 그래프는 `figures/` 폴더에 포함되어 있습니다.


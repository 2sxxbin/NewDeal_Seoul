# 1. ML 평가
## Regression
* y(x) = w_0*x + w_1
* 직선이 가장 데이터에 부합하는 w를 찾아야 한다

## MSE(Mean Square Error)
* w_0과 w_1을 결정하면 제곱 오차 계산이 가능
* 오차가 최소가 되도록 반복계산으로 최적의 w를 구해야 함
  * 경사 하강법(Gradient Descent)이라 함

## Gradient Descent
* 기울기가 양수: w는 지금보다 작은 값을 사용
* 기울기가 음수: w는 큰 값을 사용
* 0이면 최적 값

## Regression Assessment
* 회귀는 얼마나 비슷한 값으로 예측 했는 지가 중요함(정확하게 맞추기는 어려움)
  * MSE(Mean Squared Error)
  * RMSE(Root Mean Squared Error)
  * MAE(Mean Absolute Error)
  * R^2(R Squared) : 0 ~ 1 사이 값을 가지며 클 수록 잘 맞췄음을 의미
<br><br>
* 모델이 예측한 분류 결과는 Confusion Matrix라고 불리는 테이블을 통해서 정리될 수 있다.
* 이는 실제 데이터와 예측 데이터에 대한 분포를 표현한 Matrix

* |   |   ||prediction|prediction|
  |---|---|---|---|---|
  |   |   ||Negative|Positive|
  |True|Negative|:|True Negative|False Positive|
  |True|Positive|:|False Negative|True Positive|

* True Positive
  * 관심 범주를 정확하게 분류한 값
* False Negative
  * 관심 범주가 아닌 것으로 잘못 분류함
* False Positive
  * 관심 범주라고 잘못 분류함
* True Negative
  * 관심 범주가 아닌 것을 정확하게 분류

## accuracy
* 전체 예측 값에서 예측을 맞추거나 예측을 안 맞춘 것에 대한 비율
* 암 예측 같이 data가 imbalance한 경우에는 accuracy가 학습 데이터에 따라 정확하지 않을 확률이 높음
  * precision, recall이 더 선호 될 수 있음

* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TP + TN<br>
    accuracy = &nbsp;----------------------<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TP + TN + FP + FN

## precision
* 예측된 관측치 중 제대로 예측된 비중
  * Negative 한 데이터 예측을 Positive로 잘못 판단하게 되면 큰 영향이 발생하는 경우
    * 스팸메일
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TP<br>
    precision = &nbsp;---------<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TP + FP

## Recall(Sensitivity, TPR : True Positive Rate)
* Positive한 대상 중에서 실제 값이 Positive와 일치한 비율
* 암환자를 정상으로 분류 시 치명적, 사기 거래를 정상으로 분류 시 치명적
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TP<br>
    &nbsp;---------<br>
    &nbsp;&nbsp;TP + FN

## Specificity(Negative 재현율, TNR : True Negative Rate)
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TN<br>
    &nbsp;---------<br>
    &nbsp;&nbsp;TN + FP

## 좋은 평가 모델
* precision, Recall이 동시에 나오는 것, But trade-off
* precision이 100% : 정말 확실한 환자 1명만 예측
* Recall이 100% : 모든 환자를 positive로 예측

## F-1 score
* 정밀도와 재현율을 결합한 지표
* 정밀도와 재현율이 상대적으로 한쪽으로 치우치지 않은 수치를 나타내면 높은 값
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;precision * recall<br>
    F1 = 2&nbsp;*&nbsp;-----------------------<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;precision + recall

## ROC Curve
* AUC(Area Under Curve)는 ROC Curve 아래 면적을 의미
* 값이 클수록 좋은 값 (최대1)

## 비지도 학습
* 클러스터 간 거리 또는 클러스터 내 분산을 고려함

## Dunn Index
* 군집 간 거리의 최소값을 분자, 군집내 요소 간 거리의 최대값을 분모
* 군집 간 거리가 멀수록, 군집 내 분산이 작을수록 좋은 군집화

## Silhouette Index
* a(i) : i번째 개체와 같은 클러스터에 속한 요소들 간의 거리 평균
* b(i) : i번째 개체와 다른 클러스터에 속한 요소들 간의 각각 거리 평균
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b(i) - a(i)<br>
    s(i) = &nbsp;&nbsp;-----------------<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max {a(i), b(i)}
---
# 2.ML 성능 향상 기법
## Overfitting, Underfitting
* Overfitting
  * 주어진 데이터에만 특화되어 이전에 보지 못한 데이터에 대해서는 성능이 떨어지는 형상
  * Low Bias, High Variance
    * 학습 데이터의 모든 샘플에 대해 편향없이 잘 추정
    * 학습 데이터의 작은 변화에도 민감하게 반응하기 때문에 추정 값 변화가 큼
    * 성능이 일반화 되어 있지 않음

* Underfitting
  * 학습 데이터나 테스트 데이터에도 작동이 안됨
  * 학습이 잘 안되어있는 상태
  * High Bias, Low Variance
    * 모델이 편향성을 가지고 있어 모든 학습 데이터를 다 고려해 주지 않음
    * 테스트 데이터 셋에서 잘 동작하지 않음

* 최종 목표는 Low Bias, Low Variance를 달성하는 것이 목표
* Bias는 가정(추정)이 얼마나 편향되어 있는가(잘못 되었는가)
* Variance는 입력 데이터 변동에 따른 변화

## 일반적인 ML에서 데이터 셋을 활용하는 방법
* training
  * 모델 학습
* validation
  * 모델 학습 후 검증, 재 학습을 통해 모델 성능 향상
* Test
  * 모델학습과는 상관은 없이 모델의 성능을 평가하기 위한 데이터

## Cross - validation
* 여러 hyper parameter가 한 가지 training, validation set에만 특화될 수 있어서 여러 validation set을 이용한 모델 학습 방법

## Cross - validation의 종류
1) Leave-one-out cross validation(LOOCV)
   - 전체 데이터에서 샘플 하나씩 돌아가면서 검증 데이터로 사용
   - 나머지는 학습 데이터로 사용 -> 시간이 오래 걸림

2) K-fold cross validation
   - 전체 데이터를 k개 그룹으로 나누고 한 그룹씩 돌아가면서 검증 데이터로 사용
   - 나머지 그룹은 학습 데이터로 사용

3) K-Holdout Cross validation
   - K번 성능을 평가하되 한번 평가할때마다 전체 데이터에서 무작위로 학습 데이터와 검증 데이터를 만들어서 사용
   - 중복되거나 제외되는 데이터 발생 가능

## Overfitting 방지방법
* 정규화(Regularization)
  * 모델이 복잡해 질수록 불이익을 받도록 하는 방법
  * 복잡해 지면 성능이 안좋아 지도록 에러함수에 정규화 변수를 추가함
  * 정규화 파라미터 값은 조절이 잘 되어야 함(작으면 오버피팅되고 크면 언더피팅 됨)

* 조기 중단(Early Stopping)
  * 모델 학습을 위해 소요되는 시간에 제약을 두는 방법
  * 학습 중단 조건을 설정

* Feature Selection
  * 데이터 내 존재하는 특성 중 일부만 선택해서 사용하는 방법
  1) 후진 방법
       - 처음에는 전체 다 사용 후 특성을 제거
  2) 전진 방법
       - 가장 최소한의 특성 집합으로 시작해서 반복 시행 마다 특성을 늘려나감

* **Question) Feature가 많으면 좋은것 아닌가요?**
  * 값의 조합이 엄청나게 늘어난다
  * 모델 복잡성, 컴퓨팅 파워, 더 많은 데이터가 필요하다
  * 오버피팅의 위험도
  * 차원이 데이터에 따라 다르다
  * 적당한 차원에서 문제를 푸는것이 좋다

* 차원 축소(Demension Reduction)
  * 고차원, 저차원 데이터로 변환
  * 차원을 변환하기 때문에 특성의 종류와 같이 변화, 손실 정보를 최소한으로 유지

## 차원의 저주(Curse of Dimensionality)
* 특성이 늘어날 수록 특성들이 가질 수 있는 값의 조합이 늘어남
* 오버피팅의 위험도가 증가
* 경우의 수가 늘어나는 것에 비해 실제 값들이 분포하는 곳은 많지 않을 수 있음
* 적당한 차원에서 문제를 풀어 최적의 결과를 내는 것이 중요

## Feature Selection
* 데이터 내 존재하는 모든 특성들 중 일부만 선택해 사용하는 방법
* 후진 방법
  * 특성을 처음에는 전체 다 사용하고 반복 시행단계마다 특성을 제거함
* 전진 방법
  * 가장 최소한의 특성집합으로 시작해 반복 시행 단계마다 특성을 늘려나감

## Model Ensemble
* Model ensemble은 여러 모델을 함께 사용하여 기존보다 성능을 더 올리는 방법을 말함
  
## Bagging(Bootstrap Aggregation)
* Model ensemble은 여러 모델을 함께 사용하여 기존보다 성능을 더 올리는 방법을 말함
* Categorical Data는 투표방식, Continuous Data는 평균으로 집계
* Random Forest가 대표적

## Bagging(Bootstrap Aggregation) 단계
* Replace 가능한 샘플링 방법으로 원본 학습 데이터에서 새로운 학습 데이터를 생성
* 만들어진 학습 데이터를 각각에 모델에 학습
* 테스트 데이터에 대해서 학습된 모델들이 만들어낸 결과를 평균화 or 최다득표보팅을 최종 결과 생성

## Boosting
* 가중치를 활용하여 약 분류기를 강 분류기로 만드는 방법
* Adaboost, Gradient Boost, XGBoost, Light GBM

## 스태킹(Stacking), 블렌딩(Blending)
* 부스팅처럼 학습 모델이 순차적으로 실행
* 실행 단계별로 학습 모델의 역할과 기능을 다르게 설계

---
# 3. Deep Learning
## Logistic Regression
* 로지스틱 함수(시그모이드 함수)를 사용하는 분류기
* Sigmoid function : 입력된 값에 0과 1 사이 값을 할당하는 함수
* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1<br>
    y(z) = &nbsp;&nbsp;--------------<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1 + exp(-z)
* z는 함수의 입력인 feature x의 가중치 합
* 실제로 이를 완전히 매핑하기에는 어려움이 있어 bias를 추가해서 사용
* z = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n*x_n = w^T*x

## Neurons
* 뇌를 이루고 있는 기본 구성요소
* Cell body : main process unit
* Dendrite : input gate
* Axon : output gate
* 각 Neuron은 1000 ~ 10000개 Neuron과 연결됨

## Simple Model
* x : dendrites
* w : 연결 강도
* F : Cell body
* y : axon


## AND, OR, NOT 문제 해결
* |x_1|x_2|sigma|y|
  |:---:|:---:|:---:|:---:|
  |0|0|-1.5|0|
  |0|1|-0.5|0|
  |1|0|-0.5|0|
  |1|1|0.5|1|

## XOR은 해결 안됨
## Multi layer perceptron으로 해결
## Neural Network
* 여러 개의 perceptron으로 이루어져 Multi-Layer Perceptron이라고도 부름
* Input, Output Layer는 오직 하나지만 Hidden Layer는 여러개가 될 수 있음
* Hidden Layer가 많아질수록 그만큼 더 복잡한 feature공간을 표현 가능

## Feed Forward
* 입력 Feature를 받아서 출력 결과를 받는 것을 말함

## Back Propagation
* 주어진 입력 feature 값에 대해 순전파를 예측 값 계산
* 예측 값과 target의 차이, cost 함수 값 계산
* 뒤의 가중치를 미분을 통해서 업데이트 함

## Deep Neural network
* 장점
  * 높은 성능, 무한한 가능성
* 단점
  * 최적화가 어렵고 오버피팅 위험, Covariate shift 가능성

## Solution
* 높은 복잡성 : 상대적으로 simple한 structure 채택, 변환
* 최적화의 어려움
  * 좋은 시작점을 선택함
  * Activation function 선택
  * Mini - batch 그레디언트 디센트
* 오버피팅
  * 거대한 학습 데이터 양
  * 다양한 Regulariztion Method (Drop out 등)
* Internal Covariate shift
  * Batch Normalization 등 (배치별 정규화)

## CNN(Convolutional Neural Network)
* image의 특징을 인식 하는 것이 CNN의 기본 원리
* CNN architecture
* Filter(Kernel)로 Feature Map을 생성함
* Convolution layer를 통과한 특징을 pooling을 통해서 특징을 추출함
* filter의 개수만큼 output layer가 생성됨
* 크게 바라본 아키텍처 1, subsampling이 pooling
* 크게 바라본 아키텍처 2

## 순차적인 데이터 모델
* 다음 상태 예측
  * ABCDE -> Sequence Model -> F
* 분류
  * ABCDE -> Sequence Model -> Good / Bad
* 순서 생성
  * ABCDE -> Sequence Model -> 1234567

## 순차적 생성
* Machine Translation
  * This is a very good wine -> 이건 매우 좋은 와인이에요
* Speech Recognition
  * 음성 -> this is a very good wine
* Image Caption Generation
  * img -> A bird is flying

## RNN(Recurrent Neural Network)
* 이전 상태 값이 이후 상태로 입력이 되는 형태

## RNN(Recurrent Neural Network)의 문제
* 값이 길어질 수록 이전의 값의 영향력이 줄어드는 문제가 발생
  
## LSTM(Long Short Term Memory)
* 각 layer의 상태값을 passing하지 않고 저장하여 이후 layer의 입력 값으로 활용
  

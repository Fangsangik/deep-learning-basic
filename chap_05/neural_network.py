"""
# 100. Boston House Price Regression

- 보스턴 주택가격 예측

### 13 개의 종속변수와 1 개의 독립변수 (주택가격 중앙값) 으로 구성

#### 독립변수 (13 개)
CRIM 자치시(town) 별 1인당 범죄율
ZN 25,000 평방피트를 초과하는 거주지역의 비율
INDUS 비소매상업지역이 점유하고 있는 토지의 비율
CHAS 찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0)
NOX 10ppm 당 농축 일산화질소
RM 주택 1가구당 평균 방의 개수
AGE 1940년 이전에 건축된 소유주택의 비율
DIS 5개의 보스턴 직업센터까지의 접근성 지수
RAD 방사형 도로까지의 접근성 지수
TAX 10,000 달러 당 재산세율
PTRATIO 자치시(town)별 학생/교사 비율
B 1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함
LSTAT 모집단의 하위계층의 비율(%)

#### 종속변수 (1 개)
MEDV 본인 소유의 주택가격(중앙값) (단위: $1,000)
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from tensorboard import summary
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os

df_boston = pd.read_csv("/Users/hwangsang-ik/Desktop/머신러닝/infran_ML_DL-main/boston_house.csv", index_col=0)
boston = df_boston.drop("MEDV", axis=1)
target = df_boston["MEDV"]

x = boston.values # boston data
y = target.values # boston target

print(y[:10])
print(x.shape) # (506, 13)
print(y.shape) # (506,)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
# 0 ~ 1 사이의 값으로 스케일링
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# model 구성
"""
sequential model => add layer
input_shape > tuple 형태로 입력
activation X => linear
activation ='relu' => 현대작러 믾이 사용 / 활성화 함수로 사용

완전 연결: Dense 레이어의 모든 뉴런은 이전 레이어의 모든 뉴런과 연결
파라미터: 각 연결에는 가중치(weight)가 있으며, 각 뉴런에는 바이어스(bias)
Dense 레이어는 다양한 활성화 함수와 함께 사용될

Hidden layer의 output이 두번쨰 layer의 input이 됨
keras가 나머지는 알아서 해줌

지금 만든 model은 두개의 입력 층을 갖고 있는 model 
- dense 1 : 64개의 뉴런, input shape 13
- dense 2 : 32개의 뉴런
- dense 3 : 1개의 뉴런 (output layer)
"""
model = Sequential()
model.add(Dense(64, input_shape=(13,), activation ='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

"""
손실 함수 
- model이 예측한 값과 우리가 알고 있는 지도 학습 레이블 값과 얼만큼 차이가 있는지 
- 손실을 최소화 할수록 model이 좋아짐
- metrics[mae, mse] -> 중간 중간 출력 
"""

model.compile(loss = 'mean_squared_error', optimizer='adam', metrics = ['mae', 'mse'])
model.summary()

"""
history에 model이 훈련되면서 얻어진 정보들이 저장
batch_size는 한 번의 학습(Forward + Backward Propagation)에서 모델에 넣는 데이터 샘플의 개수
batch_size가 크면 메모리를 많이 사용하지만, 학습 속도가 빨라질 수 있음 / 작으면 메모리 사용량이 적지만, 학습 속도가 느려질 수 있음
epoch는 data를 몇번 보여줄 것인가 
validation_data => epoch가 끝날 때 마다 검증
verbose => 출력되는 로그의 양을 설정 / 0: 출력 없음, 1: 진행률 표시줄, 2: epoch 당 한 줄
"""
history = model.fit(x_train, y_train, batch_size = 32, epochs=500, validation_data=(x_test, y_test), verbose=1)
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

# MSE = mean squared error 계산
print('mean squared_error : {:.2f}'.format(mean_squared_error(y_test, y_pred)))
# R2 score 계산
print('R2 score : {:.2f}'.format(r2_score(y_test, y_pred)))

plt.scatter(y_test, y_test, label='true')
plt.scatter(y_test, y_pred, label='predict')
plt.xlabel('y_test')
plt.ylabel('y')
plt.legend()
plt.title('Boston House Price Prediction ($1,000)')
plt.show()

"""
history 객체에 저장된 훈련 과정 시각화 
-> history.history
- loss, mae, mse, val_loss, val_mae, val_mse
"""
plt.plot(history.history['mse'], label = 'Train error')
plt.plot(history.history['val_mse'], label = 'Test error')
plt.ylim([0, 50])
plt.legend()
plt.show()

# sklearn LinearRegression 비교
regr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)
print('Coefficients: \n', regr.coef_)
print('Intercept : \n', regr.intercept_)

# MSE(mean squared error) 계산
print("Mean squared error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
# R2 계산
print("R2 score: {:.2f}".format(r2_score(y_test, y_pred)))
print(x_test.shape, y_pred.shape)

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', c = "r")
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Boston House Price Prediction ($1,000)')
plt.show()
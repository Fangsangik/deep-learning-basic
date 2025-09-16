"""
dataset : sklearn.datasets.load_diabets
Feature : 나이 , 성별, 체질량지수, 혈압, 6가지 혈청 수치 -> scaling
target : 1년 후 당뇨병 진행 정도
model : linear_model.LinearRegression
평가기준 : RMSE(Root Mean Squared Error) => 평균제곱 오차 + root

sklearn : data = x, target = y

datasets.load_diabetes() 를 호출하면 Bunch 객체라는 걸 반환
파이썬의 Dict와 유사한 자료구조 / 주요 키 정해져 있음
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 단변수 선형회기
# 나이, 성별, 체질량지수, 혈압, 6가지 혈청 수치 => already scaled
diabetes = datasets.load_diabetes()
print(diabetes.DESCR)  # -> 데이터셋 설명
print(diabetes.feature_names)
print(diabetes.data.shape)  # (442 record, 10 features)
print(diabetes.target)  # 442
print()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(df.head())
print()

# -1은 numpy에게 알아서 행의 개수를 맞추라는 뜻
dia_x = df[['bmi']].values.reshape(-1, 1) # 442, 1
print(dia_x)
print()

# 442개의 훈련 set / 20개의 검증 set
dia_x_train = dia_x[:-20]
dia_x_test = dia_x[-20:]
print(dia_x_train.shape, dia_x_test.shape)
print()

dia_y_train = diabetes.target[:-20]
dia_y_test = diabetes.target[-20:]
print(dia_y_train.shape, dia_y_test.shape)
print()

# Sklearn의 선형회귀 모델 객체 생성
# regr을 보면 기울기와 절편이 만들어져 있음
regr = linear_model.LinearRegression()
regr.fit(dia_x_train, dia_y_train)
print(regr.coef_) # 기울기
print(regr.intercept_) # 절편
print()

y_pred = regr.predict(dia_x_test)
print("y_pred", y_pred)
print()

# 산점도
plt.scatter(dia_x_test, dia_y_test, label = "True value", color = 'b') # 실제값
plt.plot(dia_x_test, y_pred, color = 'r') # 예측값
plt.title("Diabetes Using bmi progression prediction")
plt.xlabel("bmi")
plt.ylabel("disease progression")
plt.show()

# R2 계산
r2_score(dia_y_test, y_pred)
print("R2:", r2_score(dia_y_test, y_pred))
mean_squared_error(dia_y_test, y_pred)
print("MSE:", mean_squared_error(dia_y_test, y_pred))
print()

# 다변수 선형회귀
# dia_x1 = df[["bmi", "bp"]].values # 두개 자체가 metrix -> reshape 필요 없음
dia_x1 = df.values # 10개 전체
print(dia_x1)

dia_x1_train = dia_x1[:-20]
dia_x1_test = dia_x1[-20:]
print(dia_x1_train.shape, dia_x1_test.shape)
print()

dia_y1_train = diabetes.target[:-20]
dia_y1_test = diabetes.target[-20:]
print(dia_y1_train.shape, dia_y1_test.shape)
print()

regr_1 = linear_model.LinearRegression()
regr_1.fit(dia_x1_train, dia_y1_train)
print(regr_1.coef_)
print(regr_1.intercept_)
print()

y1_pred = regr_1.predict(dia_x1_test)
print("y1_pred", y1_pred)
r2_score(dia_y1_test, y1_pred)
print("R2:", r2_score(dia_y1_test, y1_pred))
print()


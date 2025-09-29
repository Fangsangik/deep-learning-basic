"""
LSTM을 이용한 주가 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Lambda

appl = yf.download("AAPL", start='2018-01-01', end='2023-12-31', progress=False)
print(appl.head())

print(appl.Close.plot())

"""
LSTM layer로 하려고 하는 것은 지난 window-size 일(즉, t-window에서 t-1까지)의 역사적 가격을 사용하여 
시간 t의 가격을 예측하는 것입니다. 정확한 가격이 아닌 추세를 파악하기 위해 노력할 것
"""
print(appl.shape)

hist = []
target = []
window = 3
# 알아 맞출 data
close = appl['Close'].values

"""
close : 시계열 data 
window : 몇 개의 시점을 입력으로 쓸지 
hist : 입력 data 집합 x 
target : 출력 data 집합 y
x : window 개의 timestep
y : x 다음의 값
"""
# Window를 하나씩 moving 하면서 hist, target 생성
# 깂이 자기 지산으로 부터 나옴
for i in range(len(close) - window):
    x = close[i:i + window]
    y = close[i + window]
    hist.append(x)
    target.append(y)

print(close[:10])
print(hist[:5])
print(target[:5])

"""
"hist"의 각 요소는 window개 timestep의 list
1씩 증가하기 때문에 "hist"의 두 번째 요소의 마지막 항목은 "target"의 첫 번째 요소와 같아야 합니다. 
또한 마지막 숫자가 같아야 합니다.
"""
print(hist[1][-1] == target[0])

hist = np.array(hist)
target = np.array(target)
# target의 shape이 metrix 형태로 변경
target = target.reshape(-1, 1)
# data를 받아서 그 다음 종과를 알아 맞추는 모양
print(hist.shape, target.shape)

print(len(hist) - 100)

"""
일반 data 셋은 다 shuffle을 일부로 했지만, 시계열 data는 순서가 중요하기 때문에 shuffle을 하지 않습니다.
1098일의 데이터로 모델을 학습시키고 다음 100일의 데이터로 테스트하는 방식으로 데이터를 분할합니다.
"""
spilt = len(hist) - 100
x_train = hist[:spilt]
x_test = hist[spilt:]
y_train = target[:spilt]
y_test = target[spilt:]

# LSTM 데이터는 3차원이므로 2차원으로 변환 후 정규화
sc1 = MinMaxScaler()
# (samples, timesteps) 형태로 reshape
x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
x_train_scaled = sc1.fit_transform(x_train_reshaped)
# 다시 3차원으로 변환
x_train_scaled = x_train_scaled.reshape(x_train.shape)

# 테스트 데이터도 동일하게 처리
x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])
x_test_scaled = sc1.transform(x_test_reshaped)
x_test_scaled = x_test_scaled.reshape(x_test.shape)

sc2 = MinMaxScaler()
y_train_scaled = sc2.fit_transform(y_train)
y_test_scaled = sc2.transform(y_test)
print(x_train_scaled.shape, x_test_scaled.shape)
print(y_train_scaled.shape, y_test_scaled.shape)

# LSTM을 위해 3차원으로 reshape (samples, timesteps, features)
x_train_scaled = x_train_scaled.reshape(-1, window, 1)
x_test_scaled = x_test_scaled.reshape(-1, window, 1)
print("After reshape for LSTM:")
print(x_train_scaled.shape, x_test_scaled.shape)
print(y_train_scaled.shape, y_test_scaled.shape)

# Model 생성 fitting
# LSTM layer 3개, Dense layer 1개 -> stack을 쌓음
# return_sequences -> 다음번 LSTM이 올때
model = tf.keras.Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(window, 1), dropout=0.2))
model.add(LSTM(32, return_sequences=True, dropout=0.2))
model.add(LSTM(16, dropout=0.2))
# 숫자 하나 맞추는 것
model.add(Dense(units=1))
# Lambda 레이어 제거 - 정규화를 사용하므로 불필요
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
# 정규화된 데이터로 훈련
history = model.fit(x_train_scaled, y_train_scaled, epochs=100, batch_size=32)

plt.plot(history.history['loss'])
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 정규화된 테스트 데이터로 예측
pred_scaled = model.predict(x_test_scaled)
# 예측 결과를 원래 스케일로 역변환
pred = sc2.inverse_transform(pred_scaled)

plt.figure(figsize=(12,6))
plt.plot(np.concatenate((y_train_scaled.flatten(), y_test_scaled)), label='True')
plt.plot(np.concatenate((y_train_scaled.flatten(), pred.flatten)), label='Predicted')
plt.title('Apple Stock Price Prediction')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(y_test, label='True')
plt.plot( pred, label='Predicted')
plt.title('Apple Stock Price Prediction')
plt.legend()
plt.show()

"""
Simple LSTM
1) data 는 0 ~ 99 까지의 연속된 숫자이고, target 은 (1 ~ 101) * 2 으로 구성한다.
입력  data 에 대응하는 출력 data 를 예측하는 model 을 LSTM 으로 작성
연속된 5 개의 숫자를 보고 다음 숫자를 알아맞추도록 LSTM 을 이용한 model 작성
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Training Data
numbers = [[i] for i in range(105)]
print(numbers[:5])

data = []
target = []
for i in range(5, len(numbers)):
    data.append(numbers[i - 5:i])  # 5개의 연속된 숫자
    target.append([numbers[i][0] * 2])  # 다음 숫자의 2배

print(data[5])
print(target[5])

# List -> numpy array
data = np.array(data, dtype=np.float32)
target = np.array(target, dtype=np.float32)

# Normalize
# (100, 5, 1) (100, 1)
# 100 data, 5 timestep, 1 feature
data = data / 100
target = target / 100
print(data.shape, target.shape)  # (100, 5, 1) (100, 1)

model = Sequential()
model.add(LSTM(16, input_shape=(5, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics= ['mae'])
model.summary()

# validation_split=0.2 -> 80% for training, 20% for validation
history = model.fit(data, target, epochs=500, validation_split=0.2)

test_data = [[35]], [[36]], [[37]], [[38]], [[39]]
x = np.array(test_data, dtype=np.float32) / 100
model.predict(x.reshape(1, 5, 1)) * 100

test_data = [[95]], [[96]], [[97]], [[98]], [[99]]
x = np.array(test_data, dtype=np.float32) / 100
model.predict(x.reshape(1, 5, 1)) * 100

test_data = [[100]], [[101]], [[102]], [[103]], [[104]]
x = np.array(test_data, dtype=np.float32) / 100
model.predict(x.reshape(1, 5, 1)) * 100


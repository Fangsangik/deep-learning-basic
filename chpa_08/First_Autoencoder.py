"""
First Autoencoder
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) # 생성되는 난수 고정
tf.random.set_seed(42) # 텐서플로우에서 생성되는 난수 고정

"""
Data 셍성
3차원 data 생성 
"""

m = 100 # data 개수
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5 # 0 ~ 1 사이에 난수 * 3 * pi / 2 - 0.5
data = np.empty((m, 3)) # 특별한 값 없이 자리만 잡아 놓는 것
data[:, 0] = np.cos(angles) + np.sin(angles) / 2 + 0.1 * np.random.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + 0.1 * np.random.randn(m) / 2
data[:, 2] = data[:, 0] * 0.1 + data[:, 1] * 0.3 + 0.1 * np.random.randn(m)
print(data.shape)

"""
3차원 data 시각화 
"""
# data.mean(axis=0, keepdims=0)
x_train = data

# preview data
ax = plt.axes(projection='3d') # 3차원 그래프
ax.scatter3D(x_train[:, 0], x_train[:, 1], # x, y, z 좌표
             x_train[:, 2], c=x_train[:, 0], cmap='Reds') # c : 색깔, cmap : 색깔 종류

"""
Autoencoder Model
"""
# 3차원 data를 2차원으로 압축
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# encoder, decoder 생성
# autoencoder에서 중요한 것은 encoder / decoder의 경우 encoder를 학습 시키는 용도
encoder = Sequential([Dense(2, input_shape=(3,))]) # input 3차원, output 2차원
decoder = Sequential([Dense(3, input_shape=(2,))]) # input 2차원, output 3차원

# autoencoder 생성 (encoder + decoder)
# 역전파가 되면서 encoder, decoder가 같이 학습
autoencoder = Sequential([encoder, decoder])
autoencoder.summary()
# autoencoder 학습
# MSE 손실함수, SGD 최적화 함수, learning rate 0.1
autoencoder.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=0.1))
# autoencoder의 특징은 자기 자신이 정답 -> 지도학습
history = autoencoder.fit(x_train, x_train, epochs=200)

"""
encoder output 시각화 
"""
encodings = encoder.predict(x_train)
print(encodings.shape)

# encoder output 시각화
fig = plt.figure(figsize=(4, 3))
plt.plot(encodings[:, 0], encodings[:, 1], 'b.')
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)

"""
Decoder를 이용한 data 복원
"""
# decoder output 시각화
decodings = decoder.predict(encodings)
print(decodings.shape)

# decoder output 시각화
ax = plt.axes(projection='3d')
ax.scatter3D(decodings[:, 0], decodings[:, 1],
                decodings[:, 2], c=decodings[:, 0], cmap='Reds')
plt.show()



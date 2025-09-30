"""
Simple and Stacked Autoencoder with MNIST - Dense
- fashion_mnist dataset 을 이용한 deep autoencoder 생성
- Mnist dataset 의 손글씨체를 encoding 후 decoding 하여 복원
"""

import numpy as np
import tensorflow as tf
from IPython.core.pylabtools import figsize
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model  # 그림이 어떻게 모델이 구성 되는지 시각화
import matplotlib.pyplot as plt

# fashion_mnist dataset load
# 비지도 학습이기에 y값 필요 없음
(x_train, _), (x_test, _) = fashion_mnist.load_data()
print(x_train.shape)

# sample image 시각화
fig, ax = plt.subplots(1, 10, figsize=(20, 4))

# 처음 10개 이미지 출력
for i in range(10):
    # imshow() : 이미지 출력 -> 숫자로 되어 있는 값을 이미지로 변환
    ax[i].imshow(x_test[i], cmap='gray')
    ax[i].set_xticks([])  # x축 눈금 제거
    ax[i].set_yticks([])  # y축 눈금 제거

# 데이터 전처리
# 0~255 -> 0~1 0 : 검정, 1 : 흰색
# StandardScaling과 비슷
x_train = x_train / 255.
x_test = x_test / 255.

# 2차원 데이터를 handling -> convd network 사용
# 하지만 간단한 것은 dense network 사용
# -1 : 행의 개수에 따라 자동 설정, 784 : 28*28
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
print(x_train.shape, x_test.shape)

# stacked autoencoder 작성
# 784개르 feature로 하는 input layer
input = Input(shape=(784,))

# stacked autoencoder
# hidden layer가 여러개 있는 autoencoder
# units : 뉴런의 개수, activation : 활성화 함수, 전단계를 input으로 받음
# relu : max(0,x)
# 갈수록 차원이 줄어든다
x = Dense(units=128, activation='relu')(input)  # 첫번째 은닉층
x = Dense(units=64, activation='relu')(x)  # 중간층
encoder = Dense(units=32, activation='relu')(x)  # encoded representation

# decoder
# 차원을 다시 늘려감
x = Dense(units=64, activation='relu')(encoder) # encoder를 input으로 받음
x = Dense(units=128, activation='relu')(x)
# sigmoid : 0~1 사이의 값으로 압축
decoder = Dense(units=784, activation='sigmoid')(x)  # decoded representation

# autoencoder model
# 하나의 model로 합침 -> 역전파 가능
encoder_model = Model(inputs=input, outputs=encoder) # encoder model
autoencoder_model = Model(inputs=input, outputs=decoder) # autoencoder model -> backpropagation
autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy') # optimizer : 경사하강법, loss : 손실함수

# model summary, 시각화
autoencoder_model.summary()
# plotting -> 시각화
plot_model(autoencoder_model, show_shapes=True)
# fitting the model
history = autoencoder_model.fit(x_train, x_train, epochs=50, batch_size=256, validation_data=(x_test, x_test))

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# 784로 푼 것을 다시 28*28로 복원
fig, ax = plt.subplots(3, 10, figsize=(20, 8))
for i in range(10):
    # 2차원
    ax[0, i].imshow(x_test[i].reshape(28, 28), cmap='gray')

    img = np.expand_dims(x_test[i], axis=0)

    # encoder model에 predict -> 32차원 -> 8*4
    ax[1, i].imshow(encoder_model.predict(img, verbose=0).reshape(8, 4), cmap='gray')
    # autoencoder model에 predict -> 784차원 -> 28*28
    ax[2, i].imshow(autoencoder_model.predict(img, verbose=0).reshape(28, 28), cmap='gray')

    ax[0, i].axis('off')
    ax[1, i].axis('off')
    ax[2, i].axis('off')
plt.show()

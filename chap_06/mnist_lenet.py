import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Conv2D -> image 처리하는 함수
# MaxPooling2D -> 이미지의 크기를 줄여주는 함수
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Flatten -> 다차원 데이터를 1차원으로 펴주는 함수
from tensorflow.keras.layers import Dense, Flatten, Activation

# mnist dataset
from tensorflow.keras.datasets import mnist

# random seed 고정
np.random.seed(101)

# Mnist는 data 자체를 줄떄 spilt 해서 준다
# x, y로 짝을 이뤄서 준다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.figure(figsize=(5, 5))

for i in range(9):
    # 3행 3열
    # matplotlib -> 0이라는 인덱스가 없음 -> 1부터 시작
    plt.subplot(3, 3, i + 1)
    # 이미지 표시 하는 것
    plt.imshow(x_train[i], cmap='gray', interpolation='none')
    plt.title('Class {}'.format(y_train[i]))
    # xticks, yticks 눈금
    plt.xticks([])
    plt.yticks([])
# 적당히 간격을 띄어줌
plt.tight_layout()
plt.show()

# Scaling
# 0 ~ 1 사이의 값으로 스케일링
x_train_scaled = x_train / 255.
x_test_scaled = x_test / 255.
print(x_train_scaled.shape)
print(x_test_scaled.shape)

# Conv2D layer (칼라 이미지) 의  입력 사양에 맞추어 3 dimension 으로 차원 증가
x_train_scaled = np.expand_dims(x_train_scaled, axis=3)
x_test_scaled = np.expand_dims(x_test_scaled, axis=3)
print(x_train_scaled.shape)
print(x_test_scaled.shape)

# label 을 one-hot-encoding
# categorical cross entropy -> one hot encoding되어 있어야 True value & 예측한 값 하고의
# 손실 / 로스 계산 가능
# 0, 1, 2, .... -> 6번째만 1이고 나머지는 0
"""
y_train[0] = 5
y_train[0:10] = [5 0 4 1 9 2 1 3 1 4]
보시다시피 y_train[0] = 5
그래서 to_categorical(y_train)[0]이 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]가 되는 것
5번째 인덱스(0부터 시작하므로 실제로는 6번째 위치)에 1이 있고 나머지는 0
"""
y_train_onehot = tf.keras.utils.to_categorical(y_train)
print(y_train_onehot[0])
y_test_onehot = tf.keras.utils.to_categorical(y_test)
print(y_test_onehot[0])

# tf.data를 이용한 shuffling and batch
# tenser type으로 바꿔줌으로써 GPU 빠르게 사용 가능
# numpy type -> CPU에서는 돌아가지만, GPU에서는 안돌아감 -> 내부적으로 tensorflow가 numpy data 들을
# -> GPU에서 돌아갈 수 있는 -> tensor type으로 바꿔줌
# shuffle -> 섞어줌 (buffer)
# batch -> 몇개씩 묶어서 처리할지
# 훈련 set은 같은 data batch로 데이터 묶음을 계속 주게 되면, 과적합(overfitting)이 발생할 수 있음
train_ds = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train_onehot)) \
    .shuffle(10000).batch(128)

# test쪽은 훈련용이 아니기 때문에 무관함
test_ds = tf.data.Dataset.from_tensor_slices((x_test_scaled, y_test_onehot)).batch(128)

# Lenet model
model = tf.keras.Sequential()
# filter 6개, kernel size 5x5, padding='same' -> input과 output의 크기가 같음, input shape 28x28x1
# color -> 28x28x3
# +a -> 흑백 이미지에서도 dropout을 사용 할때가 있음
# - X-ray, MRI / 큰 흑백 모델
model.add(Conv2D(6, kernel_size=5, padding='same', input_shape=(28, 28, 1)))
# 활성화 함수로 relu 사용
model.add(Activation('relu'))
# pooling -> 이미지의 크기를 줄여줌
# pool_size -> 2x2
# strides -> 2x2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(16, kernel_size=5, padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 1차원으로 펴줌 400개의 1차원 데이터로 변환
model.add(Flatten())
# 120개의 Dense layer와 연결
model.add(Dense(120, activation='relu'))
# 84개의 Dense layer와 연결
model.add(Dense(84, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# epochs는 전체 훈련 데이터셋을 몇 번 반복해서 학습할지를 결정하는 매개변수
history = model.fit(train_ds, epochs=5, validation_data=test_ds)
"""
- verbose=0: 조용한 모드 - 아무것도 출력하지 않음
- verbose=1: 진행률 표시 - 진행률 바와 상세 정보 출력 (기본값)
- verbose=2: 요약 정보만 - 각 epoch마다 한 줄씩만 출력
"""
score = model.evaluate(test_ds, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
격자(grid) : 서브플롯들이 배치되는 표 형태의 레이아웃입니다.

  subplot(행, 열, 위치)에서:
  - 행: 세로로 몇 개의 줄
  - 열: 가로로 몇 개의 칸
  - 위치: 그 격자 안에서 몇 번째 칸

  예시:
  1행 2열 격자 = 가로로 2개 칸이 나란히
  ┌─────┬─────┐
  │  1  │  2  │
  └─────┴─────┘

  2행 1열 격자 = 세로로 2개 칸이 위아래로
  ┌─────┐
  │  1  │
  ├─────┤
  │  2  │
  └─────┘

  2행 2열 격자 = 2x2 표 형태
  ┌─────┬─────┐
  │  1  │  2  │
  ├─────┼─────┤
  │  3  │  4  │
  └─────┴─────┘
"""

# 시각화
# 훈랸이 잘 되었는지 확인
plt.figure(figsize=(12, 4))
# subplot(행, 열, 위치)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])

# loss
# subplot(행, 열, 위치)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

"""
Accuracy: 어느 순간 교차가 되면 검증 셋의 정확도는 떨어진다. (overfitting) -> 일반화 능력 감소 
"""
y_pred = model.predict(x_test_scaled).argmax(axis=1)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))

import seaborn as sns

plt.figure(figsize=(7, 6))
plt.xticks(np.arange(10), list(range(10)), rotation=45, fontsize=12)
plt.yticks(np.arange(10), list(range(10)), rotation=45, fontsize=12)
plt.xlabel("Predicted label", fontsize=15)
plt.ylabel("True label", fontsize=15)
plt.title("Confusion Matrix", fontsize=15)
print("test accuracy : ", accuracy_score(y_test, y_pred))

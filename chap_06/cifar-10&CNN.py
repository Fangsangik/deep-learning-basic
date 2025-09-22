from tensorflow.python.client import device_lib
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

"""
 1. 데이터 복잡성

  - CIFAR-10: 실제 사진 (비행기, 자동차 등) - 복잡한 패턴
  - MNIST: 단순한 손글씨 숫자 - 단순한 패턴

  2. 모델 크기

  - CIFAR-10: Conv2D(16→32→64) + Dense(256) - 더 많은 파라미터
  - MNIST: Conv2D(6→16) + Dense(120→84) - 적은 파라미터

  3. 과적합 위험도

  - 복잡한 데이터 + 큰 모델 = 과적합 위험 높음
  - 단순한 데이터 + 작은 모델 = 과적합 위험 낮음

  컬러 vs 흑백은 부차적

  - 컬러(3채널) vs 흑백(1채널)은 단순히 입력 채널 수 차이
  - 진짜 중요한 건: 데이터의 복잡성과 모델 크기

  결론: Dropout은 "컬러라서"가 아니라 "더 복잡하고 큰 모델이라서" 사용
"""
# CIFAR-10 전용 one-hot encoding 생성

print(device_lib.list_local_devices())
"""
[name: "/device:CPU:0"
 device_type: "CPU"
 memory_limit: 268435456
 locality {
 }
 incarnation: 1821960915115514637
 xla_global_id: -1]
"""
print(tf.__version__)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

cifa10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(2, 8, figsize=(15, 4))
axes = axes.ravel()
for i in range(16):
    idx = np.random.randint(0, len(x_train))
    axes[i].imshow(x_train[idx, :]) # 이미지 표시
    axes[i].set_xticks([]) # 눈금 제거
    axes[i].set_yticks([]) # 눈금 제거
    axes[i].set_title(cifa10_classes[y_train[idx, 0]])
plt.show()

x_train_scaled = x_train / 255.
x_test_scaled = x_test / 255.

x_train_onehot = utils.to_categorical(y_train)
y_test_onehot = utils.to_categorical(y_test)
print(x_train_onehot.shape)
print(y_test_onehot.shape)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_scaled, x_train_onehot)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test_scaled, y_test_onehot)).batch(32)

"""
# Dropout
  1. 복잡도 감소

  # Dropout 없을 때
  Dense(256) → 모든 256개 뉴런 사용 (복잡)

  # Dropout(0.5) 있을 때
  Dense(256) → 128개만 랜덤 사용 (단순)

  2. 의존성 차단

  - 문제: 특정 뉴런에 과도하게 의존
  - 해결: 랜덤하게 끄면서 다양한 뉴런 조합 학습

  3. 일반화 강제

  - 훈련 시: 매번 다른 서브네트워크로 학습
  - 예측 시: 모든 뉴런 사용하되 앙상블 효과
"""
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(train_ds, epochs=20, validation_data=test_ds, verbose=0)
model.evaluate(test_ds, verbose=0)

# argmax => 가장 큰 값의 index를 반환
y_pred = model.predict(x_test_scaled).argmax(axis=1)
print(y_pred.shape)
print(y_pred)

y_true = y_test.ravel()
print(y_true.shape)
print(y_true)

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True)

plt.xticks(np.arange(10), cifa10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifa10_classes, rotation=45, fontsize=12)
plt.xlabel("true class")
plt.ylabel("predicted class")
plt.title('Confusion Matrix')
plt.show()
print('Test Accuracy :', accuracy_score(y_true, y_pred))
"""
실습 : fashion MNIST 를 이용하여 위와 동일한 작업

Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D

np.random.seed(102)

# train, test spilt
(x_train_images, y_train_labels), (x_test_images, y_test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train_images)
print(y_train_labels.shape)
print(x_test_images.shape)
print(y_test_labels.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train_labels[i]])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

x_train_scaled = x_train_images / 255.0
x_test_scaled = x_test_images / 255.0
print(x_train_scaled.shape)
print(x_test_scaled.shape)

# Conv2D layer 의  입력 사양에 맞추어 3 dimension 으로 차원 증가
x_train_scaled = np.expand_dims(x_train_scaled, axis=3)
x_test_scaled = np.expand_dims(x_test_scaled, axis=3)

print(x_train_scaled, x_test_scaled.shape)

# label 을 one-hot-encoding
x_train_onehot = tf.keras.utils.to_categorical(y_train_labels)
x_test_onehot = tf.keras.utils.to_categorical(y_test_labels)
print(x_train_onehot[0])

# tf.data 를 이용한 shuffling and batch 구성
train_ds = tf.data.Dataset.from_tensor_slices((x_train_scaled, x_train_onehot)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test_scaled, x_test_onehot)).batch(32)

# LeNet model 구성
# image size는 줄여가고, filter는 늘려감
# kernel_size를 줄여서 적용하면 이미지가 줄어드는 정도가 더 작아짐
# 레이어가 많다고 정확한 모델이 되는 것은 아님 / 오히려 과적합 될 수 있음
model = tf.keras.Sequential()

model.add(Conv2D(6, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(16, kernel_size=3, padding='valid', activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, kernel_size=3, padding='valid', activation='relu'))
model.add(Activation('relu'))

"""# 4번째 Conv2D 레이어 추가
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(Activation('relu'))

# 5번째 Conv2D 레이어 추가
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Activation('relu'))

# 6번째 Conv2D 레이어 추가
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(Activation('relu'))"""

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu')) # 다중 분류 문제이므로 softmax 활성화 함수 사용
model.add(Dense(10, activation='softmax'))

model.summary()

# model compile and predict
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train history 시각화
history = model.fit(train_ds, epochs=5, validation_data=test_ds)
score = model.evaluate(test_ds)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# confusion matrix 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])

plt.show()

# predict
y_pred = model.predict(x_test_scaled).argmax(axis=1)
print(y_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test_labels, y_pred)
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()
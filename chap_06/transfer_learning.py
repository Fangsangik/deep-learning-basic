import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.mobilenet import decode_predictions

import numpy as np
import matplotlib.pyplot as plt


# from chap_05.neural_network import history  # 주석 처리하여 다른 파일 실행 방지

"""
Tensorflow Hub 에서 PRE-TRAINED MOBILENET 의 WEIGHT 를 가져옴  
- Fine Tuning 없이 사용하기 위해 Full Model download
"""
Trained_MobileNet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
# Trained_Mobilenet.add(Dense(100) -> add를 활용하는 방식이 있음
# Sequential 활용 하는 방법
# input_shape=(224, 224, 3) -> height, width, color_channel
Trained_Mobilenet = tf.keras.Sequential([
    hub.KerasLayer(Trained_MobileNet_url, input_shape=(224, 224, 3), trainable=False)])
print(Trained_Mobilenet.input, Trained_Mobilenet.output)

"""
PRE-TRAINED MOBILENET 평가
Fine Tuning 없이 사용
"""
from PIL import Image  # 파이썬 image 처리 하는 라이브러리
from urllib import request  # url 에서 image 가져오는 라이브러리
from io import BytesIO  # byte 형태의 데이터를 file 객체로 변환 (ASCICode로 변환)

url = "https://github.com/ironmanciti/MachineLearningBasic/blob/master/datasets/TransferLearningData/watch.jpg?raw=true"
res = request.urlopen(url).read()  # url 에서 image 데이터를 읽어옴 (byte 형태)
sample_image = Image.open(BytesIO(res)).resize((224, 224))  # open -> res data를 읽어오고, resize로 image사이즈 맞춰서 가져옴
plt.imshow(sample_image)
plt.title('Sample Image')
# plt.show()

# Trained_MobileNet이 분류하는 법을 확인
# image를 np.array로 변환후, preprocess_input이라는 메소드에 입력으로, 전처리된 이미지가 출력
# 입력된 이미지 크기는 224, 224 / 모델을 전처리가 일치해야 한다.
x = tf.keras.applications.mobilenet.preprocess_input(np.array(sample_image))
print(x.shape)

# expand_dims (차원 확장)
# image는 한건이지만 Mobilenet이 batch로 훈련 된 것 처럼 하기 위해 -> 1 x 224 x 224 x 3 (dimension을 변경해서 사용)
predicted_class = Trained_Mobilenet.predict(np.expand_dims(x, axis=0))
print(predicted_class)

# 어떤 값이 큰지 알 수 없기 때문에 argmax를 활용해서 가장 큰 값을 가진 index를 출력
# 1000개의 class 중에서 가장 높은 확률을 가진 class index를 반환
print(predicted_class.argmax(axis=-1))

# decode_predictions : 모델의 출력을 사람이 읽을 수 있는 클래스 라벨과 Mapping
print(decode_predictions(predicted_class[:, 1:]))

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = open(labels_path).read().splitlines()
print(imagenet_labels[827])  # 827번째 index가 watch
print(imagenet_labels[1:20])

"""
특정 domain 의 Batch Image 에 대한 MobileNet 평가 - No Fine Tuning
MobileNet 은 Flower 에 특화된 model 이 아니므로 정확도 낮을 것 예상
"""

flowers_data_path = tf.keras.utils.get_file(
    'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)  # untar=True : 압축을 풀어줌
print(flowers_data_path)

# flow from directory : 폴더에 모여 있는 데이터를 전부 폴더 이름에 레이블 명으로 인식해서 자동으로 가져옴
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    # 전처리 모듈
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

flower_data = image_generator.flow_from_directory(flowers_data_path,
                                                  target_size=(224, 224),
                                                  batch_size=64,
                                                  shuffle=True)
print(flower_data.class_indices)

# generator 함수는 return을 사용하는 것이 아닌 yield를 사용
# next() 함수를 사용해서 다음 데이터를 가져옴
input_batch, label_batch = next(flower_data)
print(input_batch.shape, label_batch.shape)  # (64, 224, 224, 3) (64, 5)
print(label_batch[0])  # one-hot encoding

print(flower_data.num_classes)
# class_indices : class name과 index를 dictionary 형태로 반환
print(flower_data.class_indices)

# key, value를 바꿔서 index, class name 형태로 변경
class_names = {v: k for k, v in flower_data.class_indices.items()}
print(class_names)

# data 시각화
# 전처리를 진행한 image (-1 ~ 1) -> (0 ~ 255)
# (x + 1) * 127.5
plt.figure(figsize=(16, 8))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    # 8bit unsigned integer (0~255)
    img = ((input_batch[i] + 1) * 127.5).astype(np.uint8)
    idx = np.argmax(label_batch[i])
    plt.imshow(img)
    plt.title(class_names[idx])
    plt.axis('off')

# plt.show()

# 5가지를 정확히 분류만 할 수 있다면 최적화 된 model
# mobilenet은 꽃에 대한 학습이 X -> mobilenet이 예측했을때 가장 높은 확률을 가진 class를 출력
prediction = Trained_Mobilenet.predict(input_batch[2:3])
print(prediction.shape)  # (64, 1001)
print(decode_predictions(prediction[:, 1:]))

"""
전이학습 model을 flower 분류에 적합한 model로 retrain
fine-tuning을 위해 head가 제거된 model을 가져옴

mobilenet의 경우 top layer까지 포함됨 
그래서 toplayer를 제거한 feature vector model을 가져옴 (= image feature만 쭉 뽑은 것) 
"""
extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
extractor_layer = hub.KerasLayer(extractor_url, input_shape=(224, 224, 3))
feature_batch = extractor_layer(input_batch)  # toplayer 제거된 model
print(feature_batch.shape)  # (64, 1280)

# MobileNet 의 pre-trained weight 는 update 않음
# Top layer 에 Dense layer 추가
# feature vector model은 trainable이 False -> 역전파가 더이상 발생 X
extractor_layer.trainable = False

model = tf.keras.Sequential([
    extractor_layer,
    tf.keras.layers.Dense(flower_data.num_classes, activation='softmax')
])
print(model.summary())
print(model.input, model.output)

# 64개의 image를 batch로 넣었을 때 5개의 class로 분류
prediction = model(input_batch)
print(prediction.shape)  # (64, 5)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(flower_data, epochs=5)

# Flower 분류 전문으로 fine-tuning된 model 평가
y_pred = model.predict(input_batch)
y_pred = np.argmax(y_pred, axis=-1)
print(y_pred)

y_true = np.argmax(label_batch, axis=-1)
print(y_true)

# accuracy score
print(f"{sum(y_pred == y_true) / len(y_true) * 100:.2f}%")

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for i in range(64):
  plt.subplot(8, 8, i+1)
  img = ((input_batch[i]+1)*127.5).astype(np.uint8)
  plt.imshow(img)
  color = "green" if y_pred[i] == y_true[i] else "red"
  plt.title(class_names[y_pred[i]], color=color)
  plt.axis('off')

plt.show()



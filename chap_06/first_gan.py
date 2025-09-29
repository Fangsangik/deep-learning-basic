# First GAN with MNIST - MNIST dataset

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import matplotlib.pyplot as plt
from IPython import display  # Jupyter Notebook 에서 그래프를 그릴 때 사용

"""
Utilities
가짜 이미지를 그릴 수 있는 도우미 기능
훈련되는 동안 Gan의 Sample 출력을 시각화 하는데 사용 
"""
def plot_multiple_images(images, n_cols=None):
    """
Utilities
가짜 이미지를 그릴 수 있는 도우미 기능
훈련되는 동안 Gan의 Sample 출력을 시각화 하는데 사용
"""
    # 원래 그렸던 이미지에다가 새로 그리겠다.
    display.clear_output(wait=False)
    # 격자형으로 그리기 위함
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    # batch 차원 제거
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    plt.figure(figsize=(n_cols, n_rows))

    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')


"""
Download and Prepare the Dataset 
- MNIST dataset load 
- 픽셀 값을 정규화하여 이를 전처리 
"""
# gan을 훈련하기 위해 MNIST 데이터셋을 사용
# 원래는 x_train, y_train, x_test, y_test로 나누지만, gan은 하나만 사용
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255
print(x_train[0])

# 훈련하는 동안 모델에 공급 할 수 있도록 훈련 이미지의 배치를 생성
batch_size = 256

# from_tensor_slices : numpy array를 dataset으로 변환, shuffle : 데이터를 섞음
# dataset.batch : batch size 만큼 데이터를 묶음, drop_remainder : batch size가 안되는 데이터는 버림, prefetch : 데이터를 미리 가져옴 (메모리상에 미리 올려놔라)
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
print(dataset)

"""
Build the Model 
- 생성기 : 가짜 데이터 셍성 
- 판별자 : 이미자가 가짜인지 실제인지 결정 

Sequential API를 사용하여, Dense Layer를 쌓아 이러한 하위 네트워크 구축 

Generator 
임의의 노이즈를 받아 가짜 이미지를 만드는데 사용. 이르 위해 이 모델은 랜덤 노이즈 형태로 받아서 MNIST dataset과 동일한 크기의 이미지를 출력 
SELU는 GAN에 적합한 활성화 함수로 확인, 처음 두개의 Dense network에서 이를 사용 
최종 Dense Network는 0과 1서아의 pixel 값을 생성하기 원함 -> sigmoid로 활성화 
그 다음 MNIST 데이터 set 차원에 맞게 reshape
"""
random_normal_dimension = 32

generator = keras.models.Sequential([
    # 64개의 neuron을 갖고 있는, activation function은 selu, input shape는 random normal dimension (= 32)
    keras.layers.Dense(64, activation='selu', input_shape=[random_normal_dimension]),
    # hidden layer
    keras.layers.Dense(128, activation='selu'),
    # output layer
    keras.layers.Dense(28 * 28, activation='sigmoid'),
    keras.layers.Reshape([28, 28])
])

# 훈련되지 않은 generator의 샘플 출력. -> random point
# 훈련 후 MNIST dataset의 숫자를 닮게 된다 .
# batch size는 16으로 지정, (난수를 생성)
test_noise = tf.random.normal(shape=[16, random_normal_dimension])
# 28 by 28 이미지 16개 생성
test_image = generator(test_noise)
plot_multiple_images(test_image, n_cols=4)
plt.show()

"""
Discriminator 
판별자는 입력 이미지를 가져와 가짜인지 진짜인지를 판별 
따라서 input shape은 훈련 이미지의 모양이 된다. 
이것은 flatten이 되어 dense network에 공급 될 수 있으며, 최종 출력은 0과 1 사이에 값이 된다 

generator와 마찬가지로 처음 두개의 dense network에서 SELU 활성화 / sigmoid로 final network 활성화 
"""
# Flatten : 2차원 이미지를 1차원으로 변환
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(128, activation='selu'),
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# 두 모델을 추가하여 GAN 모델을 만듦
gan = keras.models.Sequential([generator, discriminator])
print(gan.summary())

"""
Configure Training Parameters 
라벨이 0(=가짜) 또는 1(=진짜)가 될 것을 예상 -> binary_crossentropy loss function 사용
"""
# discriminator 모델 컴파일
# binary_crossentropy : 0과 1사이의 값을 예측하는데 사용되는 loss function
# rmsprop : GAN에 적합한 optimizer
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

# trainable = False : backpropagation이 발생하지 않음
discriminator.trainable = False

# gan 모델 컴파일
# binary_crossentropy : 0과 1사이의 값을 예측하는데 사용되는 loss function
# rmsprop : GAN에 적합한 optimizer
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

"""
Train the Model
 1. 가짜 데이터와 실제 데이터를 구분하도록 판별자를 훈련 
 2. 판별자를 속이는 이미지를 생성하도록 훈련 
 
 각 epoch마다 생성자에 의해 생성되는 가짜 이미지를 확인하기 위해 이미지 샘플 갤러리를 표시 
"""
def train_gan(gan, dataset, random_normal_dimension, n_epochs=10):
    """
    Defines the two-phase training loop of the GAN
    Args:
      gan -- the GAN model which has the generator and discriminator
      dataset -- the training set of real images
      random_normal_dimensions -- dimensionality of the input to the generator
      n_epochs -- number of epochs
    """

    # Gan 모델에서 generator와 discriminator를 추출
    generator, discriminator = gan.layers

    # loop start
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))

        for real_images in dataset:
            # 훈련 배치에서 배치 크기 추론
            batch_size = real_images.shape[0]

            # 1. 판별자 훈련
            # 무작위 노이즈 생성
            noise = tf.random.normal(shape=[batch_size, random_normal_dimension])

            # noise를 사용하여 가짜 이미지 생성
            fake_images = generator(noise)

            # 가짜 이미지와 실제 이미지를 연결하여 list
            mixed_image = tf.concat([fake_images, real_images], axis=0)

            # Discriminator에 대한 라벨 생성
            # 0 : 가짜, 1 : 진짜
            discriminator_labels = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

            # train_on_batch를 사용하여 mixed_images 와 discriminator_labels로 판별자를 훈련합니다.
            discriminator.trainable = True  # 역전파가 발생하도록 설정
            discriminator.train_on_batch(mixed_image, discriminator_labels)

            # Train the generator
            # GAN 공급할 노이즈 입력 배치 생성
            noise = tf.random.normal(shape=[batch_size, random_normal_dimension])

            # 생성된 모든 image에 real 레이블 지정
            generator_label = tf.constant([[1.]] * batch_size)

            # Freeze the discriminator
            discriminator.trainable = False  # 역전파가 발생하지 않도록 설정

            # 레이블이 모두 True로 설정된 노이즈에 대한 GAN 훈련
            gan.train_on_batch(noise, generator_label)
            # gan.fit(noise, generator_label)

        plot_multiple_images(fake_images, 8)
        plt.show()


def visualize_discriminator_performance(generator, discriminator, dataset, random_normal_dimension, n_sample=64):
    """
    Discriminator의 성능을 시각화
    - real image와 fake image를 구분하는 능력 확인
    - Discriminator의 예측 확률 분포 시각화
    """

    real_batch = next(iter(dataset))
    real_images = real_batch[:n_sample]

    noise = tf.random.normal(shape=[n_sample, random_normal_dimension])
    fake_images = generator(noise)

    real_predictions = discriminator(real_images)
    fake_predictions = discriminator(fake_images)

    plt.figure(figsize=(16, 8))

    # Real 이미지들 (상위 8개)
    for i in range(8):
        # plt.subplot(nrows, ncols, index) -> row, column, index
        plt.subplot(2, 8, i + 1)
        plt.imshow(real_images[i], cmap='gray')
        plt.title(f'Real\nP={real_predictions[i][0]:.3f}', color='green')
        plt.axis('off')

    # Fake 이미지들 (하위 8개)
    for i in range(8):
        plt.subplot(2, 8, i + 9)
        plt.imshow(fake_images[i], cmap='gray')
        plt.title(f'Fake\nP={fake_predictions[i][0]:.3f}', color='red')
        plt.axis('off')

    plt.suptitle('Discriminator Performance on Real and Fake Images')
    plt.tight_layout()
    plt.show()

    # 2. 확률 분포 히스토그램
    plt.figure(figsize=(12, 5))

    """
    real_predictions : Tensor 
    .numpy() : Tensor를 numpy array로 변환
    discriminator가 real image에 대해 출력한 확률 값들이 들어 있을 가능성 
    bins = 히스토그램을 그릴 때 구간을 20개로 쪼갬 
    alpha = 투명도
    label = 범례에 표시될 텍스트
    color = 막대 색상
    """
    plt.subplot(1, 2, 1)
    plt.hist(real_predictions.numpy(), bins=20, alpha=0.7, label='Real Images', color='green')
    plt.hist(fake_predictions.numpy(), bins=20, alpha=0.7, label='Fake Images', color='red')
    plt.xlabel('Discriminator Output (Probability of being Real)')
    plt.ylabel('Count')
    plt.title('Discriminator Predictions Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 성능 지표
    plt.subplot(1, 2, 2)

    # 정확도 계산 (0.5를 기준으로)
    real_accuracy = np.mean(real_predictions > 0.5)
    fake_accuracy = np.mean(fake_predictions < 0.5)
    overall_accuracy = (real_accuracy + fake_accuracy) / 2

    # 평균 확률
    real_mean = np.mean(real_predictions)
    fake_mean = np.mean(fake_predictions)

    metrics = ['Real Acc', 'Fake Acc', 'Overall Acc', 'Real Mean', 'Fake Mean']
    values = [real_accuracy, fake_accuracy, overall_accuracy, real_mean, fake_mean]
    colors = ['green', 'red', 'blue', 'lightgreen', 'lightcoral']

    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylim(0, 1)
    plt.title('Discriminator Performance Metrics')
    plt.ylabel('Score')

    # 막대 위에 값 표시
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. 통계 출력
    print("=== Discriminator Performance Analysis ===")
    print(f"Real Images - Mean: {real_mean:.3f}, Std: {np.std(real_predictions):.3f}")
    print(f"Fake Images - Mean: {fake_mean:.3f}, Std: {np.std(fake_predictions):.3f}")
    print(f"Real Accuracy (>0.5): {real_accuracy:.3f}")
    print(f"Fake Accuracy (<0.5): {fake_accuracy:.3f}")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")

    # Discriminator가 너무 강하거나 약한지 판단
    if overall_accuracy > 0.9:
        print("⚠️  Discriminator가 너무 강합니다! Generator 학습이 어려울 수 있습니다.")
    elif overall_accuracy < 0.6:
        print("⚠️  Discriminator가 너무 약합니다! 더 훈련이 필요합니다.")
    else:
        print("✅ Discriminator 성능이 적절합니다!")


def train_gan_with_monitoring(gan, dataset, random_normal_dimension, n_epochs=50):
    """
    모니터링 기능이 추가된 GAN 훈련 함수
    """
    generator, discriminator = gan.layers

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for real_images in dataset:
            batch_size = real_images.shape[0]

            # 1. 판별자 훈련
            noise = tf.random.normal(shape=[batch_size, random_normal_dimension])
            fake_images = generator(noise)
            mixed_image = tf.concat([fake_images, real_images], axis=0)
            discriminator_labels = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(mixed_image, discriminator_labels)

            # 2. 생성자 훈련
            noise = tf.random.normal(shape=[batch_size, random_normal_dimension])
            generator_label = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, generator_label)

        # 매 5 에포크마다 성능 분석
        if (epoch + 1) % 5 == 0:
            print(f"\n--- Epoch {epoch + 1} Analysis ---")
            print(f"Discriminator Loss: {d_loss:.4f}")
            print(f"Generator Loss: {g_loss:.4f}")

            visualize_discriminator_performance(generator, discriminator, dataset, random_normal_dimension)

            # 생성된 이미지도 보여주기
            plot_multiple_images(fake_images[:16], 4)
            plt.title(f'Generated Images - Epoch {epoch + 1}')
            plt.show()

# GAN 훈련 실행
train_gan(gan, dataset, random_normal_dimension, n_epochs=10)

# Discriminator 성능 분석
print("\n=== GAN 훈련 완료 후 Discriminator 성능 분석 ===")
visualize_discriminator_performance(generator, discriminator, dataset, random_normal_dimension)

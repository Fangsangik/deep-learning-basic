"""
Feature Scaling
- 특정 feature 의 value 가 다른 feature 들 보다 훨씬 크면, 그 값이 목적함수를 지배하게 되므로 정확한 학습이 되지 않음
- sklearn 의 preprocessing module 은 scale, minmax_scale 함수와 이에 대응하는 StandardScaler, MinMaxScaler class 및 fit(), transform() method 를 제공하여 쉽게 scaling 을 할 수 있도록 지원

1) Simple Feature Scaling
X_{new} = {X_{old}}/{X_{max}}

2) Min-Max Scaling
- 최대/최소값이 1, 0 이 되도록 scaling
- x=min 이면 y=0, x=max 이면 y=1.

X_{new} = {X_{old} - X_{min}/{X_{max} - X_{min}

3) Standard Scaling (Z-score)
- 평균과 표준편차를 이용하여 scaling
- \mu : 평균, \sigma : 표준편차

X_{new} = {X_{old}} - mu/sigma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# -3 ~ 5 까지의 값을 가지는 1차원 배열 생성 + reshape => matrix 로 변환
x = np.arange(-3, 6).astype('float32').reshape(-1, 1)
print(x.shape)
# vsta stack -> 수직으로 쌓기 (가상의 outlier 추가)
x = np.vstack([x, [20]])
print()

# simple scaling
# image 처리 할때 간단하게 할 수 있음
# 픽셀 값이 0~255 사이이므로 255로 나누면 0~1 사이로 바뀜
x_simple_scaled = x / np.max(x)
print(x_simple_scaled)
print()

# min-max scaling
X_minmax = (x - x.min()) / (x.max() - x.min())
print(X_minmax)

# sklearn
# fit = x의 최대값, 최소값 미리 계산 -> transform = 계산된 값으로 scaling
scaler = MinMaxScaler()
x_minmax = scaler.fit_transform(x)
print(x_minmax)
print()

# standard scaling
X_standard = (x - x.mean()) / x.std()
print(X_standard)
print()

sc = StandardScaler()
x_standard = sc.fit_transform(x)
print(x_standard)
print()


# 시각화
# subplot(행, 열, 위치)
plt.figure(figsize=(14, 4))
plt.subplot(1, 4, 1)
plt.hist(x, bins=30)
plt.title('Original')

plt.subplot(1, 4, 2)
plt.hist(x_simple_scaled, bins=30)
plt.title('Simple Scaled')

plt.subplot(1, 4, 3)
plt.hist(x_minmax, bins=30)
plt.title('standard Scaled')

plt.subplot(1, 4, 4)
plt.hist(x_standard, bins=30)
plt.title('Min-Max Scaled')

plt.show()
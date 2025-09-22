"""
## 실습

### UCI Machine Learning Repository 의 Auto MPG dataset 을 사용하여 자동차 연비 예측 Regression model 작성

- auto-mpg.data - data file


- auto-mpg.names - data 설명 file

    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous (배기량)
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete, 1 - USA, 2 - Europe, 3 - Japan
    9. car name:      string (unique for each instance)

Missing Attribute Values:  horsepower has 6 missing values  ==> "?" 로 들어 있으므로 read_csv 시 nan 으로 변환
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Data load 및 Preprocessing
data_path = tf.keras.utils.get_file("auto-mpg.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']

# null data drop
rawdata = pd.read_csv(data_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
rawdata.dropna(inplace=True)

data = rawdata.copy()

# 범주형 데이터를 one-hot encoding
data = pd.get_dummies(data, columns=['cylinders', 'origin'])

# mpg 컬럼을 label로 분리
label = data.pop('mpg')
X_train, X_test, y_train, y_test = train_test_split(data.values, label.values)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(X_test.shape)

# Regression Model Build
model = Sequential()
model.add(Dense(64, input_shape = (13,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) # output layer

model.compile(loss = 'mse', optimizer='adam', metrics=['mae', 'mse'])
model.summary()

# train
history = model.fit(X_train, y_train, batch_size = 64, epochs = 300, validation_data=(X_test, y_test), verbose=1)
model.evaluate(X_test, y_test, verbose= 0)

# predict
y_pred = model.predict(X_test)

### $r^2$ 계산
print("Mean Squared Error : {:.2f}".format(mean_squared_error(y_test, y_pred)))
print("R2 score : {:.2f}".format(r2_score(y_test, y_pred)))


### 시각화
plt.scatter(X_test[:, 0], y_test, label = 'true')
plt.scatter(X_test[:, 0], y_pred, label = 'predict')
plt.xlabel('displace')
plt.ylabel('mpg')
plt.legend()
plt.show()

plt.scatter(X_test[:, 2], y_test, label = 'true')
plt.scatter(X_test[:, 2], y_pred, label = 'predict')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.legend()
plt.show()

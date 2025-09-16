"""
### iris dataset

iris.feature_names :

Sepal Length : 꽃받침 길이
Sepal Width  : 꽃받침 너비
Petal Length  : 꽃잎 길이
Petal Width   :  꽃잎 너비

Species (꽃의 종류) :  setosa / versicolor / virginica 의 3종류로 구분된다.

**꽃받침 길이, 너비 두가지 feature 를 가지고 KNN 알고리즘을 이용하여 꽃의 종류 분류**

**neighbors.KNeighborsClassifier(n_neighbors, weights=weights)**

- weights : (가중치)
    - uniform : uniform weights. 모든 neighbor 의 가중치를 동일하게 취급
    - distance : neighbor 의 거리에 반비례하여 가중치 조정

두개의 변수가 x축, y축에 표현된 2차원 평면에 각 데이터가 점으로 표현된다. / 정답은 z 축에 표현된다.
즉 2차원 공간 안에서 3차원 데이터가 표현
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# iris dataset load
iris = load_iris()
print(iris.DESCR)
print(iris.feature_names)
print(iris.data.shape)  # (150, 4)
print(iris.target_names)

x = iris.data[:, :2]  # sepal length, sepal width
y = iris.target  # 150
print(x[:5])
print()
print(y[:5])

# 80 훈련 set / 20 검증 set
# random_state을 통해 shuffle -> 8 : 2로 분할
# test_size : 검증 set의 비율을 더 주고 안주고에 따라 정확도가 달라진다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train.shape, x_test.shape)  # (120, 2) (30, 2)
print(y_train.shape, y_test.shape)  # (120,) (30,)

# KNN 모델 객체 생성
# clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')
clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("y_pred", y_pred)
print()

# 예측 정확도 평가
print(accuracy_score(y_test, y_pred))

# x_train x축에 앞에 2개 -> x_train[:0] / y축에 뒤에 2개 x_train[:1]
"""
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1])
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1])
plt.scatter(x_train[y_train == 2, 0], x_train[y_train == 2, 1])
"""
for i in range(3) :
    plt.scatter(x_train[y_train == i, 0], x_train[y_train == i, 1], label=i)
# 데이터 하나 선택 했는데 model이 제대로 분류를 하는가?
plt.plot(x_test[20, 0], x_test[20, 1], c='r', marker= 'x', markersize = 20)
plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')

print(clf.predict(x_test[20:21]))
plt.show()

# 어떤 거를 잘 맞추고 잘 못맞췄는지
cm = confusion_matrix(y_test, y_pred)
print(cm)

# seaborn
# matplotlib 기반의 시각화 패키지
plt.figure(figsize= (5,4))
ax = sns.heatmap(cm, annot = True, fmt = 'd')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()
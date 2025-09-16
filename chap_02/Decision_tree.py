"""
### iris dataset

iris.feature_names :

Sepal Length : 꽃받침 길이
Sepal Width  : 꽃받침 너비
Petal Length  : 꽃잎 길이
Petal Width   :  꽃잎 너비

Species (꽃의 종류) :  setosa / versicolor / virginica 의 3종류로 구분된다.

**위 feature 를 모두 가지고 Decision Tree 알고리즘을 이용하여 꽃의 종류 분류**

model만 KNN -> DecisionTreeClassifier 로 변경 flow는 KNN과 동일
명확한 기준으로 인해 설명할 수 있기 떄문에 white box model
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
print(iris.data.shape)
print(iris.feature_names)
print(iris.target_names)

# train / test split & accuracy check
"""
criterion : 불순도 측정 방법
    - gini : Gini impurity (default) = cart 알고리즘
    - entropy : Information gain = id3 알고리즘
    - max_depth : 트리의 최대 깊이 (깊이가 깊어질수록 복잡한 모델) / 과적합 주의
    - random_state : 랜덤 시드 고정 (성능 개선) 
"""
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70%는 훈련  / 30% 검증 set
clf = tree.DecisionTreeClassifier(max_depth= 2, criterion='entropy', random_state=0)

clf.fit(x_train, y_train)
# 예측
y_pred = clf.predict(x_test)
print("y_pred", y_pred)
acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc)

# visualization with matplotlib
fig = plt.figure(figsize=(25, 20))
_= tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
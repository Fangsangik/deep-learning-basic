"""
Random Forest and Gradient Boosting
Random Forest Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

dataset = pd.read_csv('C:/Users/p/Desktop/infran_ML_DL-main/datasets/Social_Network_Ads.csv')
dataset.head()
print()

x = dataset.iloc[:, [2, 3]].values.astype(np.float32)
y = dataset.iloc[:, 4].values.astype(np.float32)
print(x.shape)
print(y.shape)
print()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print()

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""
n_estimators=100 -> 몇개의 트리 만들 것인지 
criterion='gini' -> 불순도 측정 방법 (gini, entropy) / gini -> cart / entropy -> id3
"""
rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf.fit(x_train, y_train)
y_predict = rf.predict(x_test)
print(y_predict)
print(accuracy_score(y_test, y_predict))

"""
Gradient Boosting Classifier
앞에 트리가 만든 오류를 예측 
"""
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0, max_depth=5)
gb.fit(x_train, y_train)
y_predict = gb.predict(x_test)
print(y_predict)
print(accuracy_score(y_test, y_predict))
print()

# visualization
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# 가상의 data point 생성
"""
x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
전체 record에 대해서 0번째의 min -> x 값의 최소값 / x_test[:, 0].max() -> 첫 컬럼의 최대값  
min() - 1 -> 간격 넉넉하게 해주기 위함 

meshgrid -> 가상의 데이터에서 교차 되는 점을 하나씩 만들어 줌 
"""
x1_min, x1_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
x2_min, x2_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))

# ravel -> N차원 배열을 1차원 배열로 변환
# 가상의 data를 생성
xx = np.column_stack([x1.ravel(), x2.ravel()]) # x1 => x / x2 => y
print(x1.shape)
print(x2.shape)
print(xx.shape)  # (58800,)
print()

y_rf = rf.predict(xx)  # 가상의 data에 대해서 예측
print(y_rf)
y_gb = gb.predict(xx)
print(y_gb)

# figure -> 새로운 창을 띄움
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True) # sharey=True -> y축을 공유
ax1.pcolormesh(x1, x2, y_rf.reshape(x1.shape), cmap = cmap_light, shading = 'auto') # 3차원으로 들어감
for i in range(2) :
    ax1.scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], color = cmap_bold(i), label = i, s = 30, edgecolor = 'k')
ax1.set_title('Random Forest Classifier')

ax2.pcolormesh(x1, x2, y_gb.reshape(x1.shape), cmap = cmap_light, shading = 'auto') # 3차원으로 들어감
for i in range(2) :
    ax2.scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], color = cmap_bold(i), label = i, s = 30, edgecolor = 'k')
ax2.set_title('Gradient Boosting Classifier')
ax1.legend()
ax2.legend()
plt.show()

print(gb.feature_importances_) # [0.12345678 0.87654321] -> 두 feature의 중요도 = 둘다 중요함
feature_imp = pd.Series(gb.feature_importances_, index=['Age', 'EstimatedSalary'])
print(feature_imp)

feature_imp.plot(kind='bar')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()


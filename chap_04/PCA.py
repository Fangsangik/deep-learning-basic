"""
PCA (Principal Component Analysis)

- 매트릭스 형태의 데이터를 3개의 메트릭스로 구분하면서 매트릭스를 구성하고 있는 잠재된 성분을 찾아내는 기법
- PCA는 Karl Pearson이 1901 년에 발명한 통계 기법으로 직교 변환을 사용하여 변수 집합을 주성분이라고 하는 선형적으로 상관되지 않은 변수 집합으로 매핑.
- PCA는 원래 데이터세트의 공분산 행렬의 SVD(Singular Value Decomposition)를 기반으로 한다.
  이러한 분해(decomposition)의 고유 벡터는 회전변환 행렬(rotation matrix)로 사용된다.
  고유 벡터는 설명된 분산에 따라 내림차순으로 회전변환 행렬에 배열.
- PCA는 단순한 공간 변환, 차원 감소 및 스펙트럼 정보로부터의 혼합물 분리에 이르기까지 다양한 응용 분야에서 강력한 기술로 사용
- "ChurnData.csv"의 각 행은 고객을 나타내고 각 열은 고객의 속성 표시
- 데이터 세트에는 지난달에 탈회한 고객에 대한 정보가 포함 (Churn 1.0 - 탈회, 0.0 - 유지)

27개의 차원 -> 2차원으로 줄이는 것이 목표
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cdf = pd.read_csv('/Users/hwangsang-ik/Documents/머신러닝/infran_ML_DL-main/datasets/ChurnData.csv')
print(cdf.head())
print(cdf.shape)

# churn 컬럼을 제외한 나머지 feature 들 -> churn 예측 값으로 사용 할 예정
columns = cdf.columns[:-1]
print(columns.size)

# loc => 컬럼 이름을 직접 정해준다
# lioc => 전체 record에 대해서 index로 접근
x = cdf.loc[:, columns]
y = cdf['churn']

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape)
print(x_test.shape)

# PCA
# 정보 손실은 최소화 -> 차원은 축소
"""
### PCA 적용

- 27 개의 feature 를 2 개로 차원 축소  


- components_
    - array, shape (n_components, n_features)
    - n_feature dimension 공간에서의 주성분 축  
    - data 의 분산을 최대로 보존하는 방향
    - explained_variance_ 에 순서대로 정렬되어 있음
    
- explained_variance_  
    - shape (n_components,)  
    - 선택한 각 구성 요소에서 설명하는 분산의 양  
    
- explained_variance_ratio_   
    - shape (n_components,)
    - 선택한 각 구성 요소가 설명하는 분산의 백분율입니다.
"""

clf = LogisticRegression()
clf.fit(x_train, y_train)  # 모델이 X와 y의 관계를 학습
y_pred = clf.predict(x_test)  # 훈련된 모델이 X_test로 예측

# y값(레이블)을 사용하는 이유는 분류의 목표가 y값을 맞히는 것
accuracy = accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy)

# 차원 축소된 churn data 시각화
# 27 개의 feature 가 2 개의 PCA 로 차원 축소 되었으므로 평면상의 시각화 가능
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # 2차원으로 축소
x_train_pca = pca.fit_transform(x_train)  # 27의 차원을 분석 -> 축을 표현 / fit이라는 단계 -> transform -> 선형대수 SVD -> 차원 축소
x_test_pca = pca.transform(x_test)  # test와 train은 동일한 분산을 갖고 있어야 함 / 별도로 fit을 할 경우 다른 분산을 갖을 수 있음
print(x_train_pca.shape)
print(x_test_pca.shape)

print(pca.components_, pca.components_.shape)
print(pca.explained_variance_ratio_)

# 2개의 차원으로 분산된 -> logistic regression
clf2 = LogisticRegression()
clf2.fit(x_train_pca, y_train)

y_pred2 = clf2.predict(x_test_pca)
accuracy2 = accuracy_score(y_test, y_pred2)
print("accuracy2 : ", accuracy2)

# 시각화
# 차원을 축소 -> 왼쪽 x, 오른쪽 y
x1, x2 = x_train_pca[y_train == 0, 0], x_train_pca[y_train == 0, 1]
plt.scatter(x1, x2, c='r', label='churm - 0', marker='o')

x1, x2 = x_train_pca[y_train == 1, 0], x_train_pca[y_train == 1, 1]
plt.scatter(x1, x2, c='blue', label='churm - 1', marker='o')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

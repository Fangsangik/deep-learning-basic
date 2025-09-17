"""
실습 - Titanic 호 data 를 이용한 Feature Engineering 과 Modeling

Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)  $\rightarrow$ 객실 등급
survival -  Survival (0 = 사망; 1 = 생존)  $\rightarrow$ 생존여부  -> 정답
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard  $\rightarrow$ 함께 탑승한 형제 또는 배우자 수
parch - Number of Parents/Children Aboard  $\rightarrow$ 함께 탑승한 부모 또는 자녀 수
ticket - Ticket Number
fare - Passenger Fare (British pound)
cabin - Cabin  $\rightarrow$ 선실번호
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)  $\rightarrow$ 탑승한 항구(얼마나 부유한지와 관련)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')  # 경고 무시

df_titanic = pd.read_csv('C:/Users/p/Desktop/infran_ML_DL-main/datasets/titanic.csv')
print(df_titanic.head())
print()

print(df_titanic.shape)
print(df_titanic.columns)
# 불필요한 열 제거
print(df_titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True))
print(df_titanic.head())
print()

print(df_titanic.info())  # null 값 확인 -> age, embarked
print(df_titanic.isnull().sum())  # null 값 확인
print(df_titanic.describe())
print()

# Feature간 상관관계
# corr -> 각 feature 간의 상관관계가 얼마나 높은지 낮은지
# 1일 경우 perfect 하게 동일 / -1일 경우 완전히 반대
print(df_titanic.corr(numeric_only=True))

import seaborn as sns

g = sns.heatmap(df_titanic.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# Missing Data 처리
print(df_titanic.isnull().sum())
print(df_titanic.shape)

# age -> 평균값으로 대체 / embarked -> drop
print(df_titanic['Age'].fillna(df_titanic['Age'].median(), inplace=True))
print(df_titanic.isna().sum())

# Data 의 skewness check
# bin -> 구간 / figsize -> 그래프 크기
# pandas -> matplotlib을 이용해서 시각화
# fare를 보면 너무 많은 연속된 수로 구성 -> log 변환
df_titanic.hist(bins=30, figsize=(8, 8));
plt.show()

df_titanic['Fare'] = df_titanic['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
df_titanic.hist(bins=30, figsize=(8, 8));
plt.show()

"""
Category 변수 처리
Category column 들을 one-hot-encoding 으로 변환한다.

숫자가 아닌 것들을 -> 새로운 컬럼으로 변환
"""
df_titanic = pd.get_dummies(df_titanic, columns=['Sex','Embarked'], drop_first=True)
print(df_titanic.head())
print(df_titanic.shape)
print()

"""
Train / Test split
"""
y = df_titanic['Survived'].values
x= df_titanic.drop('Survived', axis=1).values
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print()

# Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape)
print()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy_score)  # 0.8156424581005587
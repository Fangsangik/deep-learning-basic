"""
Logistic Regression

data: 성별 소득 data 에 따라 특정 구매자의 구매할지 여부를 예측

data -> age, estimatedSalary, purchased(정답 data) / gender(string -> 적용 하지 않음)
1. scaling
2. model instance
3. model training
4. model prediction / recall
5. model evaluation

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# 입력 data 정규화 -> preprocessing
from sklearn.preprocessing import StandardScaler

# linear model을 변형 시킨 것 -> logistic regression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, f1_score  # 평가 지표
# precision : 정밀도, recall : 재현율 (정답 case를 놓지고 싶지 않을때)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
import seaborn as sns

dataset = pd.read_csv('C:/Users/p/Desktop/infran_ML_DL-main/datasets/Social_Network_Ads.csv')
dataset.tail()
# 산사람 안산사람
data = dataset['Purchased'].value_counts()  # 0 : 257, 1 : 143
print(data)
print()

# Age , EstimatedSalary 변수
# index slicing => iloc[행, 열]
x = dataset.iloc[:, [2, 3]].values.astype(np.float32)  # [:, [2, 3]] -> 2, 3번째 열 모두 (age, estimatedSalary)
y = dataset.iloc[:, 4].values.astype(np.float32)  # [:, 4] -> 4번째 열 모두 (purchased)
print(x.shape)
print(y.shape)
print()

# dataset -> train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # 80% 훈련 / 20% 검증
print(x_train.shape, x_test.shape)  # (320, 2) (80, 2)
print(y_train.shape, y_test.shape)  # (320,) (80,)
print()

"""
train set과 test set은 애초에 shuffle을 해서 섞어서 나눴다는 이야기 -> train set과 test set의 분포가 동일하다. 라는 전제 하 
만약 data의 분포가 동일하다면 평균 == 표준편차 

x_test라는 것은 미래에 발생 할 것 / 아직 발생하지 않은 것 -> 미리 가정을 해서 사용 / 따라서 표준편차를 구할 수 없음 
"""
# Feature Scaling -> 데이터가 같은 크기로 표현되어야 함
# train data 로 fit 하고, test data 는 train data 의 분포에 맞추어 transform
sc = StandardScaler()
x_train = sc.fit_transform(x_train)  # x_train에 있는 값으로 평균과 표준편차 계산 -> x_train을 표준화
# 실제로 적용해서 작은 수로 적용
# if fit을 해줄 경우 별도로 -> 평균과 표준편차를 다시 계산
x_test = sc.transform(x_test)  # 실제로 적용해서 작은 수로 적용
print(x_train.shape)
print()

# Training set 에 대해 Logistic Regression model 을 fitting
lr_classifier = LogisticRegression(solver='lbfgs', random_state=0)
lr_classifier.fit(x_train, y_train)
print(lr_classifier)
print()

# predict
y_pred = lr_classifier.predict(x_test)
print("y_pred", y_pred)
accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy_score(y_test, y_pred))  # 0.925

# precision, recall
# precision : 정밀도 -> 예측한 것 중에 실제로 맞는 것
# recall : 재현율 -> 전체 True 중에서 맞춘 것 중 얼마나 맞췄는지 / recall을 올려야 하는 상황 -> 스레시 홀드를 조정
# recall을 올려야 하는 상황 -> 암 진단
print("precision : ", precision_score(y_test, y_pred))  # 0.903
print("recall : ", recall_score(y_test, y_pred))  # 0.7727
print()

# 스레스 홀드 조정
# precision을 높이고 싶다면 threshold를 올리면 됨
threshold = 0.8
y_pred_proba = lr_classifier.predict_proba(x_test)
y_pred_proba1 = y_pred_proba[:, 1] > threshold
print(y_pred_proba)  # 0일 확률 / 1일 확률
print("y_pred_proba1", y_pred_proba1)
print(sum((y_pred_proba[:, 1] > 0.5) == y_test) / len(y_test))  # 0.5 이상이면 1로 예측 -> 전체 개수 중 몇개를 맞췄는지
print(sum(y_pred_proba1 == y_test) / len(y_test))  # 0.5 이상이면 1로 예측 -> 전체 개수 중 몇개를 맞췄는지
print("precision_score", precision_score(y_test, y_pred_proba1))  # 엄격하게 맞추기 때문에 precision이 올라감
print("recall_score", recall_score(y_test, y_pred_proba1))  # 엄격하게 맞추기 때문에 recall이 떨어짐
print()

# confusion matrix
"""
True positive | False positive 
False negative | True negative
--------------------------------------------------
classification rate = Accuracy = (TP + TN) / (TP + TN + FP + FN)
TP = 실제 1인 것 중에 1로 맞춘 것
FP = 실제 0인 것 중에 1로 틀린 것
FN = 실제 1인 것 중에 0으로 틀린 것
TN = 실제 0인 것 중에 0으로 맞춘 것
--------------------------------------------------
"""
cm = confusion_matrix(y_test, y_pred)
print(cm)
ax = sns.heatmap(cm, annot=True, fmt='d') # annot=True 숫자 표시 / fmt='d' 정수형태로 / labels = ['0', '1'] -> 1, 0로 표기시 x축과 y축이 바뀜
ax.set_title("0.1")
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
plt.show()
print()

# ROC curve / AUC
# ROC curve : FPR(위양성률) / TPR(재현율)
# AUC : ROC curve 아래 면적 -> 1에 가까울수록 좋은 모델
# FPR : 실제 0인 것 중에 1로 잘못 예측한 것 / TPR : 실제 1인 것 중에 1로 맞춘 것
# 면적이 작을 수록 좋지 않은 모델 / 면적이 클수록 좋은 모델
y_prob = lr_classifier.predict_proba(x_test)
y_scores = y_prob[:, 1]  # 양성 클래스에 대한 확률
fpr, tpr, _ = roc_curve(y_test, y_scores)  # fpr : x축 / tpr : y축
auc = roc_auc_score(y_test, y_scores)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC & AUC curve')
plt.show()
print()
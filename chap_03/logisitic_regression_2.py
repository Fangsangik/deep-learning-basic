"""
실습
통신회사의 고객 이탈 여부를 logistic regression 으로 예측
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
import seaborn as sns

from chap_03.logisitic_regression_1 import lr_classifier

churnf_df = pd.read_csv('C:/Users/p/Desktop/infran_ML_DL-main/datasets/ChurnData.csv')
print(churnf_df.head())

"""
Data pre-processing and feature selection
개별 feature 에 대한 분석 후(본 과정에서는 생략)에 Logistic Regression 에 사용할 feature 선정
tenure(가입기간), age (연령), income(수입) 으로 예측하고 churn (이탈여부) 은 integer 로 type 변경
"""
churnf_df = churnf_df[['tenure', 'age', 'income', 'churn']]
churnf_df['churn'] = churnf_df['churn'].astype('int')
print(churnf_df.head())
print()

# x, y 분리
x = churnf_df[['tenure', 'age', 'income']]
y = churnf_df['churn']


# dataset -> train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # 80% 훈련
print(x_train.shape, x_test.shape)  # (160, 3) (40, 3)
print(y_train.shape, y_test.shape)  # (160,) (40,)
print()

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape)
print()

# Logistic Regression model fitting
log_reg = LogisticRegression(solver= 'lbfgs', random_state=0)
log_reg.fit(x_train, y_train)
print(log_reg)
print()

# predict
"""
accuracy :  0.775
precision :  0.5
recall :  0.4444444444444444
"""
y_pred = log_reg.predict(x_test)
print("y_pred", y_pred)
accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy_score(y_test, y_pred))
print('precision : ', precision_score(y_test, y_pred))
print('recall : ', recall_score(y_test, y_pred))
print()

# threshold 조정
threshold = 0.3
y_pred_proba = log_reg.predict_proba(x_test)
y_pred_proba_1 = (y_pred_proba[:, 1] >= threshold).astype('int')
print(y_pred_proba)
print(y_pred_proba_1)
print(sum(y_pred_proba[:, 1] > 0.5) == y_test / len(y_pred_proba[:, 1]))
print(sum(y_pred_proba_1) == y_test / len(y_test))
print('accuracy : ', accuracy_score(y_test, y_pred_proba_1))
print('precision : ', precision_score(y_test, y_pred_proba_1))
print('recall : ', recall_score(y_test, y_pred_proba_1))
print()

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
y_prob = log_reg.predict_proba(x_test)
y_score = y_prob[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
auc = roc_auc_score(y_test, y_score)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (area = {:.3f})'.format(auc))
plt.show()

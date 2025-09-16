"""
Pandas Crash - Basic Pandas
Pandas는 Series data type 과 DataFrame data type 으로 구성된다.

Series (1 차원) : numpy array 와 유사.
차이점 - numpy 와 달리 Series 는 axis (행, 열)에 label 을 부여할 수 있다.
즉, numpy 와 같이 숫자로만 indexing 하는 것이 아니라 label 명으로 indexing 을 할 수 있다.
또한 숫자 뿐 아니라 임의의 Python object(String dataType) 를 모두 element 로 가질 수 있다.

DataFrame (2차원, table)
- Python program 안의 Excel

Series vs DataFrame
Series(Excel의 column) -> column 단위로 관리하면 memory 효율적
DataFrame(Excel의 sheet)
"""

import numpy as np
import pandas as pd

# Data Frame : 여러개의 Series를 같은 index 기준으로 모아 table로 만든 것
# rand = 균일 난수를 생성
# randn = 정규분포 난수를 생성 (평균 = 0 / 표준편차 = 1, -1)
np.random.seed(101)  # 랜덤 시드 고정
data = np.random.randn(5, 4)
print(data)

df = pd.DataFrame(data, columns = ['A', 'B', 'C', 'D'])
print(df.columns) # 컬럼명
print(df.info())
print(df.describe()) # 통계량 (25, 50, 75 1사 분위수, 2사 분위수, 3사 분위수, 4사 분위수)
print(df['A']) # A 컬럼
print(type(df['A']))

# 컬럼 추가
df['new'] = df['A'] * 2
print(df)

# 컬럼 삭제
# 그냥 삭제하면 안됨 (axis=1 컬럼 삭제, axis=0 행 삭제)
# 메모리 상에서는 삭제, 원본은 유지
print(df.drop('new', axis=1))
print(df)

# 원본에 반영하려면 inplace=True
df.drop('new', axis=1, inplace=True)
print(df)
print(df, df.shape)
print()

"""
Misssing data가 있는 row 혹은 column 삭제
"""
df = pd.DataFrame({'A': [1, 2, np.nan],
                               'B': [5, np.nan, np.nan],
                               'C': [1, 2, 3]})
print(df)
print()

# na 값이 하나라도 있는 row 삭제
print(df.dropna())
print()

# missing value 대체
df.fillna(value=0)
df.fillna(df['A'].mean())
print(df)
print()

# missing value 를 포함하고 있는 모든 column 삭제
print(df.dropna(axis=1))

# CSV 파일 읽기
# default : , (comma) / sep = '구분자'
# df = pd.read_csv('파일경로', sep='구분자')
# df.head() # 앞에서 5개 출력
# df.tail() # 뒤에서 5개 출력
# print(df)
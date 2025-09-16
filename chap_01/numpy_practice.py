# 1. Numpy and Linear Algebra (선형대수)

"""
- 수치 data 를 다루는데 효율적이고 높은 성능 제공
- 각종 수학적 함수 제공
- Python scientific library 들이 Numpy 기반으로 구축

ndarray
- n-dimensional array (다차원 배열 객체) 로 구성 """

import sys
import numpy as np

# 스칼라
x1 = 6
print(x1)

# 벡터
# 원소값이 하나인 1차원 배열
x2 = np.array([1])
print(x2, x2.shape)
print()

# 1차원 배열
x3 = np.array([1, 2, 3])
print(x3, x3.shape)
# 큰 값 호출
np.argmax(x3)
print()

# 2차원 배열
y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y, y.shape)
print()

# 3차원 배열 (= tensor)
# element 걸러내는 법
# vector 함수 생각 하면 됨
# image를 사용 할 때 많이 사용
z = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(z, z.shape)
print()

# ex
# 앞 부분이 행, 뒤가 열
# 파이썬의 list 슬라이싱 문법과 동일
"""
Expression   Shape
arr[:2, 1:]   (2, 2)
arr[2]        (3,)
arr[2, :]     (3,)
arr[2:, :]    (1, 3) 행이 하나인 matrix 
arr[:, :2]    (3, 2)
arr[1, :2]    (2,)
arr[1:2, :2]  (1, 2)
"""

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr, arr.shape)
print(arr[:2, 1:])
print(arr[2])
print(arr[2, :])
print(arr[2:, :])
print(arr[:, :2])
print(arr[1, :2])
print(arr[1:2, :2])
print()

"""
Vector의 내적(inner product)
내적이 되려면 두 백터의 dimension 이 같아야 함
내적은 각 element 끼리 곱한 후 모두 더하는 것
inner product = dot product

ab^t = [2,5,1] [4 = [2*4 + 5*3 + 1*5] = 8 + 15 + 6 = 28
                3
                5]
matrix 곱셈
두 행렬 A, B는 A의 열 갯수가 B의 행 갯수가 같을 때 곱 가능 
result = A의 row x B의 column
[a b  [e f 
 c d] g h] = [ae + bg  af + bh
                ce + dg  cf + dh]
"""
x = np.array([1, 2, 3])
print(x, x.shape)
y = np.array([4, 5, 6])
print(y, y.shape)
print(x.dot(y)) # 1*4 + 2*5 + 3*6 = 32

a = np.array([[2,1], [1,4]])
print(a, a.shape)
b = np.array([[4,3], [5,6]])
print(b, b.shape)
print(a.dot(b)) # [[2*4 + 1*5, 2*3 + 1*6], [1*4 + 4*5, 1*3 + 4*6]]
print()

# 전치행렬
# 행과 열을 바꿈
# (m,n) -> (n,m)
a = list(range(1, 10))
print(a)
a = np.arange(1, 10)
print(a) # array([0 1 2 3 4 5 6 7 8 9])

b = a.reshape(3,3)
print(b, b.shape) # (3,3)
# 행과 열 위치 변경
print(b.T, b.T.shape) # (3,3)
c = np.transpose(b)
print(c, c.shape) # (3,3)
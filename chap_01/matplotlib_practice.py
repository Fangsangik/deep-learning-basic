"""
Matplotlib Crash - Basic Matplotlib
matplotlib 은 두가지 style 로 사용 가능

1. Functional Programing Style
```
import matplotlib.pyplot as plt
plt.plot(x, y)
```

2. OOP stype
```
fig, ax = plt.subplots()
ax.plot(x, y)
```

matlab style (pylab) 은 더이상 사용하지 않음
pyplot 의 object 구성
가장 큰 object : Figure
그 안에 여러개의 Axes (subplot) 이 들어감
Axes 안에 여러개의 axis (xaxis, yaxis) 가 들어감
"""

import numpy as np
from matplotlib import pyplot as plt

# 한글 폰트 사용
import platform
from matplotlib import font_manager

if platform.system() == "Darwin":  # Mac
    plt.rc('font', family='AppleGothic')
else:
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # For Windows
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)

plt.rcParams['axes.unicode_minus'] = False  # 한글사용시 마이너스 사인 깨짐 방지

# 점 찍기
plt.plot(3, 2, 'o')
plt.show()

# 선 긋기
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

x = [5, 8, 10]
y = [12, 16, 6]
x2 = [6, 9, 11]
y2 = [6, 15, 7]
plt.plot(x, y)
plt.plot(x2, y2)
plt.title("test plot")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

# bar
plt.bar(x, y, label="bar plot")
plt.bar(x2, y2, label="bar2 plot")
plt.title("test bar")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend() # 범례
plt.show()

# scatter plot(산점도)
plt.scatter(x, y, label='x')
plt.scatter(x2, y2, label='x2')
plt.title('Scatter plot')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
plt.legend()
plt.show()

# histogram
np.random.seed(0)
x = np.random.randn(1000)
plt.hist(x, bins=30) # bins = 계급 구간
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# OOP style
x = np.random.randint(low = 1, high = 11, size = 50)
y = x + np.random.randint(1,5, size = x.size)
data = x + y
print(data[:6])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8,4)) # 1행 2열
ax1.scatter(x, y, color = 'r', marker = 'o', edgecolors = 'k')
ax1.set_title('Scatter plot')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.grid(True)
ax2.hist(data, bins = 30)
ax2.set_title('Histogram')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('X axis')
plt.show()

# Imshow
# image data 처럼 행과 열을 가진 행렬 형태의 2차원 데이터는 imshow 으로 표현
from sklearn.datasets import load_digits
digits = load_digits()
x= digits.images[0] # 파이썬의 dict 형태
print(x, x.shape)
plt.imshow(x, cmap = 'gray') # 흑백 이미지
plt.xticks([])
plt.yticks([])
plt.show()
"""
# K-Means Clustering

### Random 하게 생성된 toy dataset 으로 K-Means clustering test

make_blobs :
```
    Input :
         n_samples :  cluster 에 균등 분할될 total data point 숫자
         centers : generate 할 centroid 숫자
         cluster_std : cluster 의 standard deviation

    Output :
        X: 생성된 sample 들
        y: 각 sample 의 label
```
KMeans :
> init : initialization method -> k-means++ (smart choosing of centroids)
> n_clusters : k 값
> n_init : 반복횟수

DBSCAN :

>eps : epsilon (radius)
>min_sample : minimum samples within the radius
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans, k_means
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

np.random.seed(101)
# 임의로 센터를 지정하고 데이터 생성
center_location = [[3, 2], [1, -1], [-1, 2]]
x, _ = make_blobs(n_samples=1500, centers=center_location)
print(x.shape)  # (1500, 2)

# 인공적으로 만든 데이터
plt.scatter(x[:, 0], x[:, 1], marker='.')
# plt.show()

# K-Means Clustering
k = KMeans(n_clusters=3)
k.fit(x)

"""
[[ 1.05328721 -0.96170352]
 [ 3.14738308  2.02818883]
 [-0.97958037  2.04290344]]
"""
print(k.labels_)
centers = k.cluster_centers_
print(centers)

colors_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
colors_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(x[k.labels_ == i, 0], x[k.labels_ == i, 1], marker='.', color=colors_bold(i), label=f'Cluster {i}')
    plt.plot(centers[i, 0], centers[i, 1], 'o', markersize=20, color=colors_light(i), markeredgecolor='k')
# plt.show()
print()

# DBSCAN
from sklearn.cluster import DBSCAN

x1, _ = make_blobs(n_samples=500, centers=[[-3, 3]])
x2, _ = make_blobs(n_samples=500, centers=[[3, 3]])
x3 = np.random.rand(500, 2) * 3 + 4
x4 = np.random.rand(10, 2) * 3 # outlier
print(x1.shape, x2.shape, x3.shape, x4.shape)

plt.figure(figsize= (8, 6))
plt.scatter(x1[:, 0], x1[:, 1], marker='.', color='red', label='x1')
plt.scatter(x2[:, 0], x2[:, 1], marker='.', color='blue', label='x2')
plt.scatter(x3[:, 0], x3[:, 1], marker='.', color='green', label='x3')
plt.scatter(x4[:, 0], x4[:, 1], marker='.', color='orange', label='x4 (outliers)')
plt.legend()
plt.title('Original Data Before DBSCAN')

"""
vstack : 배열들을 세로 방향(행 방향)으로 쌓는 함수
  - x1: (500, 2) - 500개 샘플, 2차원
  - x2: (500, 2) - 500개 샘플, 2차원
  - x3: (500, 2) - 500개 샘플, 2차원
  - x4: (10, 2) - 10개 샘플, 2차원

  np.vstack()으로 세로로 쌓으면:
  500 + 500 + 500 + 10 = 1510개 샘플
  """
x = np.vstack([x1, x2, x3, x4])
print(x.shape)  # (1510, 2)

plt.scatter(x[:, 0], x[:, 1], marker='.')
# DBSCAN
# eps=0.5 (반경)
db = DBSCAN(eps=0.3, min_samples=10)
db.fit(x)

cluster = set(db.labels_)
print(cluster)

# linspace -> 간격을 주는 것
"""
RGB 투명도 
[[0.61960784 0.00392157 0.25882353 1.        ]
 [0.97485582 0.557401   0.32272203 1.        ]
 [0.99807766 0.99923106 0.74602076 1.        ]
 [0.52733564 0.8106113  0.64521338 1.        ]
 [0.36862745 0.30980392 0.63529412 1.        ]]
"""
np.linspace(1, 10, 100)
colors = plt.cm.Spectral((np.linspace(0,1, len(cluster))))
print(colors)

# list(zip(cluster, colors)) ->cluster와 colors를 묶음
list(zip(cluster, colors))
plt.figure(figsize=(8, 6))
for k, col in zip(cluster, colors) :
    members = db.labels_ == k
    plt.scatter(x[members, 0], x[members, 1], marker='.', color=col, label=f'Cluster {k}')
plt.title('DBSCAN Clustering')
plt.show()


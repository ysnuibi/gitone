import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 读取原始数据
X = []
f = open('D:/Pydemo/数据与代码2021/第三部分/task3.txt', encoding='utf-8')
lineIndex = 1
for v in f:
    if lineIndex > 1:
        X.append([float(v.split()[1]), float(v.split()[2])])
    else:
        pass
    lineIndex += 1
# 转化为numpy array
X = np.array(X)

# 类簇的数量
n_clusters = 10

# 需要选手补全部分
#################################################################
model = KMeans(n_clusters)
model.fit(X)
X = pd.DataFrame(X)
print(X.head())

labels = model.labels_
print(model.cluster_centers_)

MARKERS = ['*', 'v', '+', '^', 's', 'x', 'o', 'q', '$']
COLORS = ['r', 'g', 'm', 'c', 'y', 'b', 'orange', 'pink', 'red', 'gray']

plt.figure(figsize=(10, 10))
plt.title("result", fontsize=20)
plt.xlabel("jing", fontsize=20)
plt.ylabel("wei", fontsize=20)
for i in range(n_clusters):
    members = labels == i
    plt.scatter(
        X[members, 1],
        X[members, 0],
        marker='o',
        c=COLORS[i],
        s=60,
        alpha=0.4

    )
plt.grid()

#################################################

# plt.title('China')
# plt.show()

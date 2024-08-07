import os
from experiment.visualization import make_ellipses
from robustgmm.gmm import GMM_EM
os.environ["OMP_NUM_THREADS"] = '1'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


# 载入数据集
file_path = r"../dataset/bank.csv"
data = pd.read_csv(file_path)
# 将数据分为特征和敏感属性
X = data[['balance', 'duration']].values
sensitive_attr = data['marital'].values
# 取一部分数据点
num_samples = 200
random_indices = np.random.choice(X.shape[0], num_samples, replace=False)
X = X[random_indices, :]
# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 设置聚类数量和轮廓系数分数的范围
min_clusters = 2
max_clusters = 10
best_score = -1
best_clusters = None
# 尝试不同的聚类数量
for n_clusters in range(min_clusters, max_clusters + 1):
    # 使用K-means进行初始聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=2)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_
    # 使用K-means聚类结果初始化GMM模型
    gmm = GaussianMixture(n_components=n_clusters, init_params='kmeans')
    gmm.fit(X, kmeans_labels)
    # 获取聚类结果和后验概率
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    # 计算轮廓系数
    score = silhouette_score(X, labels)
    # 保存最优的聚类数量和轮廓系数
    if score > best_score:
        best_score = score
        best_clusters = n_clusters
print("best_scote:", best_score)
print("best_clusters:", best_clusters)

# 使用最优的聚类数量进行最终的聚类
kmeans = KMeans(n_clusters=best_clusters, n_init=2)
kmeans.fit(X)
kmeans_labels = kmeans.labels_

# GMM using Standard EM Algorithm with random initial values
gmm = GaussianMixture(n_components=best_clusters, init_params='kmeans')
gmm.fit(X, kmeans_labels)
labels = gmm.predict(X)
gamma = gmm.predict_proba(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_
# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Standard EM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
make_ellipses(ax=ax2, means=means_sklearn, covs=covs_sklearn,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example1')
plt.savefig(r"../result/standard_EM", dpi=300)
plt.show()
plt.close()

# GMM using self EM Algorithm with random initial values
mu, cov, alpha = GMM_EM(X, best_clusters, 100)
# Visualization
plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=.2)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Real Data and Real Gaussian Distribution', fontsize=10)
ax1.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('self GMM with random initial values', fontsize=10)
ax2.scatter(X[:, 0], X[:, 1], marker='.', c='g', s=10)
make_ellipses(ax=ax2, means=mu, covs=cov,
              edgecolor='tab:red', m_color='tab:red', ls='-', n_std=3)
plt.suptitle('Example2')
plt.savefig(r"../result/self_GMM", dpi=300)
plt.show()

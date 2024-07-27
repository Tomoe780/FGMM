import matplotlib.pyplot as plt
from visualization import *
import matplotlib as mpl
import matplotlib.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
# Although Axes3D is not used directly,
# it is imported because it is needed for 3d projection.
from robustgmm.robustgmm import RobustGMM
from robustgmm.generator import Generator_Multivariate_Normal
from robustgmm.generator import Generator_Univariate_Normal


# 载入数据集
file_path = r"../dataset/bank.csv"
data = pd.read_csv(file_path)

# 将数据分为特征和敏感属性
X = data[['balance', 'duration']].values
sensitive_attr = data['marital'].values

# 模型个数，即聚类个数
K = 2

# GMM using Standard EM Algorithm with random initial values
# 从X中选择K个样本作为初始均值向量
init_idx = np.random.choice(np.arange(X.shape[0]), K)
means_init = X[init_idx, :]
gmm = GaussianMixture(n_components=K, means_init=means_init)
gmm.fit(X)
means_sklearn = gmm.means_
covs_sklearn = gmm.covariances_

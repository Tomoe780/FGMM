import pandas as pd
# 读取数据集
file_path = r"../dataset/original-bank.csv"
data = pd.read_csv(file_path, delimiter=';')
# 选择指定的列："balance"、"duration"为特征，"marital"为敏感属性
data = data[['balance', 'duration', 'marital']]
# 删除有空数据的行
data.dropna()
# 对"marital"的敏感组热编码
data['marital'] = data['marital'].map({'single': 0, 'married': 1, 'divorced': 2})
# 输出预处理后的数据集
new_file_path = r"../dataset/bank.csv"
data.to_csv(new_file_path, header=1, index=0)

import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = r"../dataset/bank.csv"
data = pd.read_csv(file_path)
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
print(pd.DataFrame(data))
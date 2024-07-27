import pandas as pd

file_path = r"../dataset/bank.csv"
data = pd.read_csv(file_path)
print(pd.DataFrame(data))
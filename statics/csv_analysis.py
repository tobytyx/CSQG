import os
import pandas as pd
out_dir = "../out/loop/"

def load_csv(filename):
    filename = os.path.join(out_dir, filename)
    data = pd.read_csv(filename)
    return data

data = load_csv("va_expand.csv")
print(data)
print(data["status"])
succ_data = data.loc[data["status"]=="success"]
print(succ_data)
print(succ_data["t5_acc"].mean())
fail_data = data.loc[data["status"]=="failure"]
print(fail_data)
print(fail_data["t5_acc"].mean())
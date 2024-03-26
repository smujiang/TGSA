import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data_dir = "/infodev1/non-phi-data/junjiang/TGSA/benchmark_dataset_pilot1_generator/csa_data"
fn = os.path.join(data_dir, "drug_response_with_IC50.csv")

dr_df = pd.read_csv(fn, sep=",")
new_dr_df = dr_df[dr_df["ic50"] >= 0]
dr_df[dr_df["ic50"] < 0] = 0
new_dr_df["ic50"].plot.hist(bins=100)
plt.show()
import numpy as np
x = np.array(dr_df["ic50"])
ic50_scaled = (x-np.min(x))/(np.max(x)-np.min(x))
dr_df["ic50"] = ic50_scaled


dr_df["ic50"].plot.hist(bins=100)
plt.show()




print("debug")















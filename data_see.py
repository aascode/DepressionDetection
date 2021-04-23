import pandas as pd
import numpy as np
import os
from glob import glob

#data_path="E:/csci535/project/DAIC/processed_data/AUs/"
data_path="E:/csci535/project/DAIC/processed_data/eGeMAPS/"

input_path = data_path + "*.npy"
path_set = glob(input_path)
minLen = float("inf")

for path in path_set:

    print(os.path.basename(path))

    input_data = np.load(path)

    print(input_data.shape)

    minLen = min(minLen,input_data.shape[0])

print("minLen:",minLen)
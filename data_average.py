import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import os
from glob import glob

input_path = "E:/csci535/project/DAIC/processed_data/"
output_path = "E:/csci535/project/DAIC/average_data/"

features = ['AUs','eGeMAPS']

data_path= input_path + features[1] +"/"

input_path = data_path + "*.npy"
path_set = glob(input_path)
minLen = float("inf")

for path in path_set:

    print(path)

    subj = os.path.basename(path)
    #print(subj)

    '''
    au_raw = np.load(path)

    # AU 每秒取平均值
    
    au_result = []

    auSum = [0 for _ in range(0, 49)]

    count = 0

    for k in range(0, len(au_raw)):
        if count < 30:
            for j in range(0, 49):
                auSum[j] += au_raw[k][j]
                count += 1
        else:
            res = []
            for j in range(0, 49):
                res.append(auSum[j] / 30)
                auSum[j] = au_raw[k][j]

            au_result.append(res.copy())
            count = 1
    '''

    # eGeMAPS 每秒取平均值

    ege_raw = np.load(path)

    ege_result = []

    egeSum = [0 for _ in range(0, 23)]

    count = 0

    for k in range(0, len(ege_raw)):
        if count < 100:
            for j in range(0, 23):
                egeSum[j] += ege_raw[k][j]
                count += 1
        else:
            res = []
            for j in range(0, 23):
                res.append(egeSum[j] / 100)
                egeSum[j] = ege_raw[k][j]

            ege_result.append(res.copy())
            count = 1

    #file_np = np.array(au_result[-400:])
    file_np = np.array(ege_result[-400:])
    print(file_np.shape)
    out_path = output_path + features[1] +"/"
    np.save(os.path.join(out_path, subj[:-4])+'.npy', file_np)
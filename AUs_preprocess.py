import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import os
from glob import glob


def convert_file_to_pickle(path, feature_set):
    if "AUs" in feature_set:
        df = pd.read_csv(path, header=0)
    elif path.endswith('.csv'):
        df = pd.read_csv(path, header=0, sep=';')
    elif path.endswith('.mat'):
        df = pd.DataFrame(loadmat(path)['feature'])

    col_ind = 4 if "AUs" in feature_set else 0 if '.mat' in feature_set else 2

    if not path.endswith('.mat'):
        out_df = df.iloc[:, col_ind:]
        return out_df
    else:
        out_df = df.iloc[:, col_ind:]
        return out_df.values

def preprocess1(path, feature_set, out_path):

    scaler = StandardScaler()

    subj = os.path.basename(path)

    print(subj)

    file_np = convert_file_to_pickle(path, feature_set)
    file_np = scaler.fit_transform(file_np)
    print(file_np.shape)
    np.save(os.path.join(out_path, subj[:-4])+'.npy', file_np)

source_dir = "E:/csci535/project/DAIC/data/"
features = ["AUs","eGeMAPS"]

path_set=glob(source_dir+features[0]+"/"+"*.csv")
out_dir="E:/csci535/project/DAIC/processed_data/"
out_path=out_dir+features[0]+"/"
for path in path_set:
    preprocess1(path,features[0],out_path)


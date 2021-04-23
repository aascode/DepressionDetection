import pandas as pd
import numpy as np
import os
from glob import glob
import keras
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error

from model import Net_2LSTM_1regress

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

test_path="E:/csci535/project/DAIC/data/test_split.csv"
df = pd.read_csv(test_path)
df = df.dropna(axis=0,how="any")
df = df.reset_index(drop=True)
#print(df)

data_path = "E:/csci535/project/DAIC/average_data/"

saved_weights = "../weights/Net_2LSTM_1regress_mse_LR4_ep40.hdf5"
model = Net_2LSTM_1regress()
model.load_weights(saved_weights)

# Initialization
batch_size = len(df['Participant_ID'])
#batch_size = 5

X1 = np.empty((batch_size, *(400, 49)))
X2 = np.empty((batch_size, *(400, 23)))

y1 = []
y2 = []
y3 = []
y4 = []

labels = {}
for i in range(0, len(df['Participant_ID'])):
    yLis = []
    yLis.append(df['PHQ_Binary'][i])
    yLis.append(df['PHQ_Score'][i])
    yLis.append(df['PCL-C (PTSD)'][i])
    yLis.append(df['PTSD Severity'][i])

    labels[df['Participant_ID'][i]] = yLis.copy()

for i,ID in enumerate(df['Participant_ID']):

    print("ID:",ID)

    # AU
    AUpath = data_path + "AUs/"
    eGePath = data_path + "eGeMAPS/"
    au_raw = np.load(AUpath + str(ID) + '.npy')
    ege_raw = np.load(eGePath + str(ID) + '.npy')

    X1[i,] = np.array(au_raw[-400:])
    X2[i,] = np.array(ege_raw[-400:])

    # Store class
    label = labels[ID]
    y1.append(label[0])
    y2.append(label[1])
    y3.append(label[2])
    y4.append(label[3])

y_predicted = model.predict([X1,X2])
#print(y_predicted)
y2_p = y_predicted


y2_ccc = ccc(np.array(y2),np.array(y2_p))
y2_RMSE = mean_squared_error(y2, y2_p)**0.5
print('y2_ccc:',y2_ccc)
print('y2_RMSE:',y2_RMSE)












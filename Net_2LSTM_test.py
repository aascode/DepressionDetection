import pandas as pd
import numpy as np
import os
from glob import glob
import keras
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error

from model import Net_2LSTM

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

data_path = "E:/csci535/project/DAIC/processed_data/"

saved_weights = "../weights/Net_2LSTM_CCC_LR3_ep80.hdf5"
model = Net_2LSTM()
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

    # eGeMAPS 每秒取平均值
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

    X1[i,] = np.array(au_result[-400:])
    X2[i,] = np.array(ege_result[-400:])

    # Store class
    label = labels[ID]
    y1.append(label[0])
    y2.append(label[1])
    y3.append(label[2])
    y4.append(label[3])

y_predicted = model.predict([X1,X2])
#print(y_predicted)

y1_p = y_predicted[0]
y2_p = y_predicted[1]
y3_p = y_predicted[2]
y4_p = y_predicted[3]

y1_p = np.argmax(y1_p, axis=1)
#y1_recall = recall_score(y1,y1_p,average="micro")
#y1_precision = precision_score(y1,y1_p,average="micro")
#print('y1_recall:',y1_recall)
#print('y1_precision:',y1_precision)
y1_f1 = f1_score(y1,y1_p,average="micro")
print('y1_f1:',y1_f1)

y2_ccc = ccc(np.array(y2),np.array(y2_p))
y2_RMSE = mean_squared_error(y2, y2_p)**0.5
print('y2_ccc:',y2_ccc)
print('y2_RMSE:',y2_RMSE)

y3_p = np.argmax(y3_p, axis=1)
y3_f1 = f1_score(y3,y3_p,average="micro")
print('y3_f1:',y3_f1)

y4_ccc = ccc(np.array(y4),np.array(y4_p))
y4_RMSE = mean_squared_error(y4, y4_p)**0.5
print('y4_ccc:',y4_ccc)
print('y4_RMSE:',y4_RMSE)










import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.metrics import f1_score, recall_score, precision_score
import keras
from keras.layers import Masking, Embedding, LSTM, Dense, Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam

from model import Net_2LSTM_1regress
from generator import Generator_Net_2LSTM_1regress
from loss import ccc_loss

LR = 0.0001
BATCH_SIZE = 32
EPOCH = 400

#AU: minLen: 12447; 3628*49
#eGeMAPS: 3628*23
label_path="E:/csci535/project/DAIC/data/train_split.csv"
data_path="E:/csci535/project/DAIC/padding_average_data/"

#label_path="../data/train_split.csv"
#data_path="../average_data/"


prefix = "Net_2LSTM_1regress_mse_LR4"

#saved_weights = "../weights/2LSTM_B32_ep140.hdf5"
model = Net_2LSTM_1regress()
#model.load_weights(saved_weights)

adam = Adam(LR)

#model.compile(optimizer=adam, loss=["binary_crossentropy","mse","binary_crossentropy","mse"])
model.compile(optimizer=adam, loss=["mse"])

ith = prefix +'_ep' + str(EPOCH)
log_name = '../logs/'+ ith + '.log'
csv_logger = keras.callbacks.CSVLogger(log_name)

ckpt_filepath = '../weights/'+ prefix +'_ep{epoch:02d}.hdf5'
model_ckpt = keras.callbacks.ModelCheckpoint(ckpt_filepath,period = 40)

callbacks = [csv_logger,model_ckpt]

train_gen = Generator_Net_2LSTM_1regress(label_path=label_path,data_path=data_path, output_type="PHQ", batch_size=BATCH_SIZE,shuffle=True)

model.fit_generator(generator=train_gen, epochs=EPOCH, callbacks=callbacks)


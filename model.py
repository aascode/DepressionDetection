import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Activation, Masking, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda, Input, Dense, Concatenate ,Conv2DTranspose, TimeDistributed
from keras.layers import LeakyReLU,BatchNormalization,AveragePooling2D,Reshape
from keras.layers import UpSampling2D,ZeroPadding2D
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.layers import Lambda,TimeDistributed, Bidirectional
from keras import layers

def Net_2LSTM():

    input1 = Input((400,49))
    input2 = Input((400,23))

    #print("input1:",input1.shape)

    l = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(400,49)))(input1)
    d1 = Dense(32,activation='relu')(l)

    #print('d1:',d1.shape)

    l2 = Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2, input_shape=(400,23)))(input2)
    d2 = Dense(32,activation="relu")(l2)


    d3 = Concatenate()([d1,d2])
    d4 = Dense(64,activation="relu")(d3)

    #branch1: PHQ_Binary
    b1 = Dense(16,activation="relu")(d4)
    o1 = Dense(2,activation="sigmoid",name="classify_1")(b1)

    #print('o1:',o1.shape)

    #branch2: PHQ_Score
    b2 = Dense(16,activation="relu")(d4)
    o2 = Dense(1,activation="relu", name="regress_1")(b2)

    #branch3: PCL-C (PTSD)
    b3 = Dense(16, activation="relu")(d4)
    o3 = Dense(2,activation="sigmoid",name="classify_2")(b3)

    #branch4: PTSD Severity
    b4 = Dense(16, activation="relu")(d4)
    o4 = Dense(1,activation="relu", name="regress_2")(b4)

    model = Model([input1, input2],[o1,o2,o3,o4])

    return model

def Net_2LSTM_1regress():

    input1 = Input((3628,49))
    input2 = Input((3628,23))

    #print("input1:",input1.shape)

    m1 = Masking(mask_value=0)(input1)
    l = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2))(m1)
    d1 = Dense(32,activation='relu')(l)

    #print('d1:',d1.shape)

    m2 = Masking(mask_value=0)(input2)
    l2 = Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2))(m2)
    d2 = Dense(32,activation="relu")(l2)


    d3 = Concatenate()([d1,d2])
    d4 = Dense(64,activation="relu")(d3)

    #branch2: PHQ_Score
    b2 = Dense(16,activation="relu")(d4)
    o2 = Dense(1,activation="relu", name="regress_1")(b2)

    model = Model([input1, input2],o2)

    return model



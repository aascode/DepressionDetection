import numpy as np
import keras
import pandas as pd


class Generator_Net_2LSTM(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, label_path, data_path, batch_size=32, shuffle=True):

        'Initialization'

        self.batch_size = batch_size
        self.data_path = data_path

        dF = pd.read_csv(label_path)

        labels = {}
        list_IDs = []

        for i in range(0,len(dF['Participant_ID'])):
            yLis = []
            yLis.append(dF['PHQ_Binary'][i])
            yLis.append(dF['PHQ_Score'][i])
            yLis.append(dF['PCL-C (PTSD)'][i])
            yLis.append(dF['PTSD Severity'][i])

            labels[dF['Participant_ID'][i]] = yLis.copy()

            list_IDs.append(dF['Participant_ID'][i])

        self.labels = labels
        self.list_IDs = list_IDs

        #self.n_classes = n_classes

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *(400,49)))
        X2 = np.empty((self.batch_size, *(400,23)))

        #y = np.empty((self.batch_size), dtype=int)
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            AUpath = self.data_path + "AUs/"
            eGePath = self.data_path + "eGeMAPS/"
            au_raw = np.load(AUpath + str(ID) + '.npy')
            ege_raw = np.load(eGePath + str(ID) + '.npy')

            X1[i,] = au_raw
            X2[i,] = ege_raw

            # Store class
            label = self.labels[ID]
            y1.append(label[0])
            y2.append(label[1])
            y3.append(label[2])
            y4.append(label[3])

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return [X1,X2], [keras.utils.to_categorical(np.array(y1),num_classes=2), np.array(y2,dtype=np.float64),
                         keras.utils.to_categorical(np.array(y3),num_classes=2), np.array(y4,dtype=np.float64)]

class Generator_Net_2LSTM_1regress(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, label_path, data_path, output_type, batch_size=32, shuffle=True):

        'Initialization'

        self.batch_size = batch_size
        self.data_path = data_path
        self.output_type = output_type

        dF = pd.read_csv(label_path)

        labels = {}
        list_IDs = []

        for i in range(0,len(dF['Participant_ID'])):
            yLis = []
            yLis.append(dF['PHQ_Binary'][i])
            yLis.append(dF['PHQ_Score'][i])
            yLis.append(dF['PCL-C (PTSD)'][i])
            yLis.append(dF['PTSD Severity'][i])

            labels[dF['Participant_ID'][i]] = yLis.copy()

            list_IDs.append(dF['Participant_ID'][i])

        self.labels = labels
        self.list_IDs = list_IDs

        #self.n_classes = n_classes

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *(400,49)))
        X2 = np.empty((self.batch_size, *(400,23)))

        #y = np.empty((self.batch_size), dtype=int)
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            AUpath = self.data_path + "AUs/"
            eGePath = self.data_path + "eGeMAPS/"
            au_raw = np.load(AUpath + str(ID) + '.npy')
            ege_raw = np.load(eGePath + str(ID) + '.npy')

            X1[i,] = au_raw
            X2[i,] = ege_raw

            # Store class
            label = self.labels[ID]
            y1.append(label[0])
            y2.append(label[1])
            y3.append(label[2])
            y4.append(label[3])

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.output_type=="PHQ":
            return [X1,X2], np.array(y2,dtype=np.float64)
        else:
            return [X1,X2], np.array(y4,dtype=np.float64)



class Generator_2inputs_original(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, label_path, data_path, batch_size=32, shuffle=True):

        'Initialization'

        self.batch_size = batch_size
        self.data_path = data_path

        dF = pd.read_csv(label_path)

        labels = {}
        list_IDs = []

        for i in range(0, len(dF['Participant_ID'])):
            yLis = []
            yLis.append(dF['PHQ_Binary'][i])
            yLis.append(dF['PHQ_Score'][i])
            yLis.append(dF['PCL-C (PTSD)'][i])
            yLis.append(dF['PTSD Severity'][i])

            labels[dF['Participant_ID'][i]] = yLis.copy()

            list_IDs.append(dF['Participant_ID'][i])

        self.labels = labels
        self.list_IDs = list_IDs

        # self.n_classes = n_classes

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *(400, 49)))
        X2 = np.empty((self.batch_size, *(400, 23)))

        # y = np.empty((self.batch_size), dtype=int)
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            AUpath = self.data_path + "AUs/"
            eGePath = self.data_path + "eGeMAPS/"
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

            '''
            x_result = []
            for l in range(0,len(au_result)):
                x_result.append(au_result[l]+ege_result[l])

            X[i,] = np.array(x_result)
            '''

            # print('au_result:',len(au_result[0]))
            # print('ege_result:',len(ege_result[0]))

            X1[i,] = np.array(au_result[-400:])
            X2[i,] = np.array(ege_result[-400:])

            # Store class
            label = self.labels[ID]
            y1.append(label[0])
            y2.append(label[1])
            y3.append(label[2])
            y4.append(label[3])

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return [X1, X2], [keras.utils.to_categorical(np.array(y1), num_classes=2), np.array(y2,type=np.float64),
                          keras.utils.to_categorical(np.array(y3), num_classes=2), np.array(y4, type=np.float64)]





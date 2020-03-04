import pickle
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input,Reshape
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.layers import LSTM
import tensorflow as tf

def generate_time_step_samples(np_array
                               ,ts_length = 16
                               ,offset = 16):
    """
    Generate windowing logic for Spectrograms to be used
    for model
    
    Inputs
    ------
    np_array : numpy array
        original spectrogram (e.g. 128X396) - Greyscale images. 
    ts_length : int
        timestep length of window (horizontal only)
    offset : int
        sliding offset     
        
    Returns
    -------
    (samples, ts_length, 128, 1) array ready for Keras
    """
    list_of_arrays = []
    col_dim = np_array.shape[0]
    time_len = np_array.shape[1]
    
    start = 0
    end = ts_length
    
    while(end < time_len):
        trn_data = np_array[:,start:end]
        trn_data = np.reshape(trn_data.T,(ts_length,col_dim,1),order='C')
        list_of_arrays.append(trn_data)
        
        start+=offset
        end+=offset
        
    new_array = np.stack(list_of_arrays)     
    return new_array           
    
   #Import files of Spectrograms & Targets in dictionary format
   
def generate_XY(dic,X,Y,ts_length,offset)
    """Generate X & Y"""
    for idx,key in enumerate(dev.keys()):
        if idx == 0:
            X = generate_time_step_samples(dic[key]['data'],ts_length=ts_length,offset=offset) 
            Y = [dic[key]['target'][1]]*X.shape[0]
        else:
            temp_array = generate_time_step_samples(dic[key]['data'],ts_length=ts_length,offset=offset)
            temp_y = [dic[key]['target'][1]]*temp_array.shape[0]
            X = np.vstack([X,temp_array])
            Y = Y_test + temp_y
    return X,Y

with open('/home/ubuntu/data/train_5000.pkl', "rb") as f:
    training_5000 = pickle.load(f)
with open('/home/ubuntu/data/train_10000.pkl', "rb") as f:
    training_10000 = pickle.load(f)    
with open('/home/ubuntu/data/train_15000.pkl', "rb") as f:
    training_15000 = pickle.load(f)
with open('/home/ubuntu/data/train_20000.pkl', "rb") as f:
    training_20000 = pickle.load(f)
with open('/home/ubuntu/data/train_25000.pkl', "rb") as f:
    training_25000 = pickle.load(f)
    
#Train Model 

time_steps = 64

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=2,kernel_size=8, activation="relu"), input_shape=(time_steps, 128, 1)))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(time_steps))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np

# Train the model, iterating on the data in batches of 32 samples
model.fit(X, Y, epochs=10, batch_size=32)

import numpy as np
from sklearn import metrics

ypred = model.predict_proba(X_test)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, ypred)
metrics.auc(fpr, tpr)
    

# Used to test some of the initial functionality of the dp_ml python package

import dp_ml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pywt
import seaborn
from statsmodels.robust import mad
import tensorflow as tf

f = h5py.File('test_24Oct2017_time.hdf5','r')
t_data = f['Data/Simulation/CircWG/time']
t_names = list(t_data)
ex_t = t_data[t_names[0]]

f_data = f['Data/Simulation/CircWG/frequency']
f_names = list(f_data)
ex_f = f_data[f_names[0]]

dbFile = h5py.File('test_24Oct2017_time.hdf5')
dbHier = 'Data/Simulation/CircWG'

X_batch, y_batch = dp_ml.getBatchFromDB(dbFile,'Data/Simulation/CircWG',0, 1, freqOrTime='time',procFx=dp_ml.getMax)

X_batch_f, y_batch_f = dp_ml.getBatchFromDB(dbFile,'Data/Simulation/CircWG',0, 1, freqOrTime='frequency',procFx=dp_ml.getMax)

#init, saver, loss, training_op, X_ph, y, logits = dp_ml.setupModel(n_inputs = 3,n_hidden1 = 10,n_hidden2=5, batch_size=100, activation_function = tf.nn.relu)

#dp_ml.runModel('test_24Oct2017_time.hdf5', 'Data/Simulation/CircWG', init, saver, loss, training_op, X_ph, y, procFx = dp_ml.getMax, freqOrTime = 'time',batch_size = 40, n_epochs = 100)



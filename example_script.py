import dp_ml
import h5py
import numpy as np

f = h5py.File('test_24Oct2017_time.hdf5','r')
t_data = f['Data/Simulation/CircWG/time']
t_names = list(t_data)
ex_t = t_data[t_names[0]]


init, saver, loss, training_op, X_ph, y, logits = dp_ml.setupModel(n_inputs = 3,n_hidden1 = 5,n_hidden2=5, batch_size=1)

dp_ml.runModel('test_24Oct2017_time.hdf5', 'Data/Simulation/CircWG', init, saver, loss, training_op, X_ph, y, procFx = dp_ml.getMax, freqOrTime = 'time',batch_size = 700, n_epochs = 100)



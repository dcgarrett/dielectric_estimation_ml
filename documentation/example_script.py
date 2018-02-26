import dp_ml
import h5py
import numpy as np



#f = h5py.File('test_24Oct2017_time.hdf5','r')
#t_data = f['Data/Simulation/CircWG/time']
#t_names = list(t_data)
#ex_t = t_data[t_names[0]]


init, saver, loss, training_op, X_ph, y, logits = dp_ml.setupModel(n_inputs = 10,n_hidden1 = 15,n_hidden2=5, batch_size=1)

dp_ml.runModel(dp_ml.HDF5Path + '/' + dp_ml.HDF5Filename, 'Data/Simulation/CircWG', init, saver, loss, training_op, X_ph, y, procFx = dp_ml.procIdentity, freqOrTime = 'time',batch_size = 1, n_epochs = 100)



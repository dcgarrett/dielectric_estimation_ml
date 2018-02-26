# Tensorflow functions for making and running models

import tensorflow as tf
import h5py
import time
import numpy as np
import pdb
from scipy.signal import argrelextrema
from tensorflow.python import debug as tf_debug
from .hdf5_fx import *

logs_path = "/tmp/dp_ml/2"


def procIdentity(sig, dbFile=None, scale = True):
	"""Identity function for use in batch retrieval."""
	return sig


def reset_graph(seed=42): # make notebook stable across runs
	tf.reset_default_graph()
	tf.set_random_seed(seed)
	np.random.seed(seed)


def setupModel(n_inputs = 10,  n_outputs = 2, n_hidden1=20,n_hidden2=20,n_epochs=10000,batch_size=50,dropout_rate=0.5,activation_function=tf.sigmoid, learning_rate = 0.01, opt = "gradient"):
	"""sets up tensorflow model for dielectric property estimation

		- n_inputs should be the number of dimensions  - e.g. sepDist, f, 8 s-param values
		- n_outputs is the number of outputs - e.g. 2 for permittivity and conductivity
		- n_hidden1 is the number of hidden layers in the first layer
		- n_hidden2 is the number of hidden layers in the second layer
		- n_epochs is the number of times the algorithm will iterate
	"""
	
	reset_graph()
	my_graph = tf.Graph()

	X_ph = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X_ph")
	y_ph = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y_ph")

	training = tf.placeholder_with_default(False, shape=(), name='training')

	with tf.name_scope("dnn"):
		hidden1 = tf.layers.dense(X_ph, n_hidden1, name="hidden1", activation=activation_function)

		# Add dropout:
		hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)

		hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", activation=activation_function)
		
		# Add dropout:
		hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)

		logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs") # SHOULD THIS INCLUDE HIDDEN 1 TOO?
		#logits = tf.transpose(logits)

	with tf.name_scope("loss"):
		loss = tf.reduce_mean(tf.squared_difference(logits, y_ph))

	with tf.name_scope("train"):
		if opt == "gradient":
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		elif opt == "adam":
			optimizer = tf.train.AdamOptimizer(learning_rate)
		
		training_op = optimizer.minimize(loss)

	with tf.name_scope("eval"):
		accuracy = tf.reduce_mean(tf.squared_difference(logits, y_ph))

	init = tf.global_variables_initializer()
	#init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
	saver = tf.train.Saver()

	return init, saver, loss, training_op, X_ph, y_ph, logits


def runModel(dbName, dbHier, init, saver, loss, training_op, X_ph, y_ph, n_epochs=10000, batch_size = 50, freqOrTime = 'frequency', procFx = procIdentity):
	timestr = time.strftime("%Y%m%d-%H%M%S")
	modelFileName = "./Models/ANN_model_"+ timestr +"_" + str(n_epochs)+"-epochs_" + ".ckpt"
	print("Model filename: ", modelFileName)

	loss_train = np.zeros((n_epochs))
	loss_test = np.zeros((n_epochs))

	tf.summary.scalar("loss", loss)
	summary_op = tf.summary.merge_all()

	with tf.Session() as sess: # with graph.as_default()
		init.run()
		sess.run(tf.initialize_all_variables())

		writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
		
		with h5py.File(dbName,'r') as dbFile:
			n_files = len(list(dbFile[dbHier+"/"+freqOrTime]))
			for epoch in range(n_epochs):
				for iteration in range(n_files // batch_size):
					X_batch, y_batch = getBatchFromDB(dbFile, dbHier, iteration, batch_size, freqOrTime, procFx = procFx)
					# pdb.set_trace()
					X_dim = X_batch.shape

					# Reshape the batch data:
					X_batch = X_batch.reshape(X_dim[0],X_dim[1]*X_dim[2])
					

					# now for some tensorflow debugging code:
					#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
					#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

					summary = sess.run(training_op, feed_dict={X_ph: X_batch, y_ph: y_batch})
					#writer.add_summary(summary, epoch * iteration)
					writer.add_graph(sess.graph)

				loss_train[epoch] = loss.eval(feed_dict={X_ph: X_batch, y_ph: y_batch}) / batch_size
				#print(epoch, "Train loss:", loss_train[epoch], "Test loss:", loss_test[epoch])
				print(epoch, "Train loss:", loss_train[epoch])
			
		save_path = saver.save(sess, modelFileName)


def getBatchFromDB(dbFile, dbHier, iteration, batch_size, freqOrTime='frequency', procFx = procIdentity, scale=True,order=False):
	'''dbFile is the h5py pointer to the databse file

		- dbHier is the relative path to the data you're looking for (without 'frequency' or 'time')
		- index is the range of values to extract from the file
		- procFx is the function used to treat the X data
		- returns X_batch and y_batch for use with neural net
	'''

	group = dbFile[dbHier+'/'+freqOrTime]
	groupNames = list(group)

	raw_dat = group[groupNames[iteration*batch_size]][:].T

	# Get some example data to see what the size is:	
	exData = procFx(raw_dat,dbFile=dbFile)


	X_batch = np.zeros((batch_size, exData.shape[0]+1, exData.shape[1]))

	y_batch = np.zeros((batch_size, 2))
		
	lenDat = exData.shape[1]
	
	for i in range(0,batch_size):
		if order:
			ind = iteration*batch_size + i # Changing this to be a random value!
		else:
			ind = np.random.randint(len(groupNames)) 
		
		f_S = group[groupNames[ind]][:].T
        
		eps_i = group[groupNames[ind]].attrs['eps']
		sig_i = group[groupNames[ind]].attrs['sig']
		sep_i = group[groupNames[ind]].attrs['sepDist']

		eps_avg = eps_i*sep_i/np.sum(sep_i)
		sig_avg = sig_i*sep_i/np.sum(sep_i)
		sep_tot = np.sum(sep_i)

		y_batch[i,0] = eps_avg 
		y_batch[i,1] = sig_avg

		if scale:
			X_batch[i,0,:] = sep_tot/10 # puts it in cm instead
		else:
			X_batch[i,0,:] = sep_tot
		X_batch[i,1:,:] = procFx(f_S, dbFile=dbFile, scale=scale)
	
	X_batch_smaller = X_batch[:,0:3,:]
	X_batch = X_batch_smaller
	#X_batch[:,1:,0] = X_batch[:,1:,0] * 1e9

	return X_batch, y_batch

def filterData(X, y, sepDist=None, eps=None, sig=None):
    '''Returns the data meeting specified input criteria.
    For each parameter not given, the dataset meeting the other criteria is given
    
    Inputs: 

        - X: extracted data with dimensions [nBatches, nParameters, nSamples]
        - y: corresponding eps and sig with dimensions [nBatches, 2]
        - sepDist: desired separation distance
        - eps: desired permittivity
        - sig: desired conductivity

    Outputs:

        - X_des: filtered X data according to sepDist, eps, and sig criteria
        - y_des: filtered y data according to sepDist, eps, and sig criteria.
    '''

    # filter for separation distance:
    if sepDist is not None:
        X_d = X[X[:,0,0] == sepDist]
        y_d = y[X[:,0,0] == sepDist]
    else:
        X_d = X
        y_d = y
    
    # filter for permittivity:
    if eps is not None:
        X_de = X_d[y_d[:,0] == eps]
        y_de = y_d[y_d[:,0] == eps]
    else:
        X_de = X_d
        y_de = y_d
    
    # filter for conductivity:
    if sig is not None:
        X_des = X_de[y_de[:,1] == sig]
        y_des = y_de[y_de[:,1] == sig]
    else:
        X_des = X_de
        y_des = y_de
        
    return X_des, y_des


def getMax(sig, dbFile=None, nmax = 1, scale = True):
        # returns the nmax local maxima in a given vector. Intended for time domain signals
        # If nmax > num local maxima, extra entries are padded with zeros
        '''getMax of a given signal. 
        Inputs: 
       	- sig: some matrix of dimensions[timeVec+numSParams, numPoints]
       	- nmax: number of maxima to choose
       	- scale: whether or not to scale the magnitude values by dividing by 1e9, time values by mult by 1e9
        Output is [numSParams, 2*nmax], where both the magnitude and time of the maxima are given
        '''

        SMax = np.zeros((sig.shape[0]-1, nmax*2))

        t = sig[0,:]
        for i in range(1,sig.shape[0]): 
                sij = sig[i,:]
                

                maxInd = argrelextrema(np.absolute(sij), np.greater)
                ind = np.argpartition(np.absolute(sij[maxInd]), -nmax)[-nmax:]

                maxNInd = [maxInd[0][i] for i in ind]
                sijMax = [np.absolute(sij[i]) for i in maxNInd] 
                tMax_i = [t[i] for i in maxNInd]

                if len(sijMax) == 0:
                    SMax[(i-1),:] = 0
                else:
                    for j in range(0,nmax):
                        if scale:
                            SMax[(i-1),2*j] = sijMax[-(j+1)] / 1e9
                            SMax[(i-1),2*j+1] = tMax_i[-(j+1)] * 1e9
                        else:
                            SMax[(i-1),2*j] = sijMax[-(j+1)]
                            SMax[(i-1),2*j+1] = tMax_i[-(j+1)]

        return SMax


def predictFromModel(modelFileName, X_data, saver, logits, X_ph):
	with tf.Session() as sess:
		saver.restore(sess, modelFileName)
		Z = logits.eval(feed_dict={X_ph: X_data})
		y_pred = Z
	return y_pred


def plotMagPhase(X_epssigdist, scale=1.0, absPlot=True, mode = 'Transmission', domain='time'):
    
    #plt.plot(dp_ml.mag2db(X_epssigdist[:,4]))
    if mode == 'transmission':
        if domain == 'time':
            ind = 3
        elif domain =='frequency':
            ind = 4
    elif mode == 'reflection':
        ind = 2
        
    if absPlot:
        if domain == 'time':
            plt.plot(X_epssigdist[:,1],np.absolute(X_epssigdist[:,ind]))
            plt.ylim([0,scale*scaleFactor])
        elif domain == 'frequency':
            plt.plot(X_epssigdist[:,1],dp_ml.mag2db(X_epssigdist[:,ind]))
            plt.ylim([-80,0])
            dbString = '[dB]'
        plt.xlabel(xlabel)
    else:
        if domain == 'time':
            plt.plot(X_epssigdist[:,1],(X_epssigdist[:,ind]))
            plt.ylim([-scale*scaleFactor,scale*scaleFactor])
        elif domain == 'frequency':
            plt.plot(X_epssigdist[:,1],np.absolute(X_epssigdist[:,ind]))
            plt.ylim([0,scale*scaleFactor])
        plt.xlabel(xlabel)
    plt.xlim([X_epssigdist[0,1], X_epssigdist[-1,1]])
    plt.ylabel('Magnitude ' + dbString)



# Wavelet analysis

def genericInner(sig, ex_sig):
    num_el = np.shape(ex_sig)
    inner = np.zeros(num_el)
    zero_vec = np.zeros(num_el)
    ex_shifted = np.zeros(num_el)
    
    for i in range(0,num_el[0]):
        ex_shifted[0:i] = zero_vec[0:i]
        ex_shifted[i+1:] = ex_sig[0:num_el[0]-(i+1)]
        inner[i] = np.inner(ex_shifted, sig) / np.inner(ex_shifted,ex_shifted)
        
    return inner

def waveletInnerMax(sig,dbFile=None, scale=True,sim=True):
	# Get calibration data
	if sim: 
		dbHier_cal = 'Calibration/Simulation/CircWG'
	else:
		dbHier_cal = 'Calibration/Measurement/Nahanni'
	X_cal_t, y_cal_t = dp_ml.getBatchFromDB(dbFile, dbHier_cal, 0, 2, freqOrTime='time', procFx = dp_ml.procIdentity)
	thru = X_cal_t[1,3,:]
	refl = X_cal_t[0,2,:]


	inner_mtx = np.zeros(sig.shape)
	inner_mtx[0,:] = sig[0,:] # time
	for i in range(1,sig.shape[0]):
		if i == 1 or i == 4:
			ref = refl
		else:
			ref = thru
		inner_mtx[i,:] = genericInner(sig[i,:],ref) 
	SMax_wv = getMax(inner_mtx,scale=scale)

	return SMax_wv






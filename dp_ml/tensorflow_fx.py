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



def procIdentity(sig):
	"""Identity function for use in batch retrieval."""
	return sig

def reset_graph(seed=42): # make notebook stable across runs
	tf.reset_default_graph()
	tf.set_random_seed(seed)
	np.random.seed(seed)

def setupModel(n_inputs = 10,  n_outputs = 2, n_hidden1=20,n_hidden2=20,n_epochs=10000,batch_size=50,dropout_rate=0.5,activation_function=tf.sigmoid, learning_rate = 0.01):
	"""sets up tensorflow model for dielectric property estimation"""
	#n_inputs should be the number of dimensions  - e.g. sepDist, f, 8 s-param values
	#n_outputs is the number of outputs - e.g. 2 for permittivity and conductivity
	#n_hidden1 is the number of hidden layers in the first layer
	#n_hidden2 is the number of hidden layers in the second layer
	#n_epochs is the number of times the algorithm will iterate
	
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

	with tf.name_scope("loss"):
		loss = tf.reduce_mean(tf.squared_difference(logits, y_ph))

	with tf.name_scope("train"):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
					pdb.set_trace()
					

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




def getBatchFromDB(dbFile, dbHier, iteration, batch_size, freqOrTime='frequency', procFx = procIdentity):
	# dbFile is the h5py pointer to the databse file
	# dbHier is the relative path to the data you're looking for (without 'frequency' or 'time')
	# index is the range of values to extract from the file
	# procFx is the function used to treat the X data
	# returns X_batch and y_batch for use with neural net

	group = dbFile[dbHier+'/'+freqOrTime]
	groupNames = list(group)


	# Get some example data to see what the size is:	
	exData = procFx(group[groupNames[iteration*batch_size]][:])
	X_batch = np.zeros((exData.shape[0]*batch_size, exData.shape[1]+1))
	y_batch = np.zeros((exData.shape[0]*batch_size, 2))
		
	lenDat = exData.shape[0]
	
	for i in range(0,batch_size):
		ind = iteration*batch_size + i
		f_S = group[groupNames[ind]][:]
		#f_S = group[groupNames[ind]][1:-1]
		eps_i = group[groupNames[ind]].attrs['eps']
		sig_i = group[groupNames[ind]].attrs['sig']
		sep_i = group[groupNames[ind]].attrs['sepDist']

		eps_avg = eps_i*sep_i/np.sum(sep_i)
		sig_avg = sig_i*sep_i/np.sum(sep_i)
		sep_tot = np.sum(sep_i)

		y_batch[lenDat*i:lenDat*(i+1),0] = eps_avg
		y_batch[lenDat*i:lenDat*(i+1),1] = sig_avg

		X_batch[lenDat*i:lenDat*(i+1),0] = sep_tot
		X_batch[lenDat*i:lenDat*(i+1),1:] = procFx(f_S)

	return X_batch, y_batch



def getMax(sig, nmax = 1):
        # returns the nmax local maxima in a given vector. Intended for time domain signals
        # If nmax > num local maxima, extra entries are padded with zeros

        #SMax = np.zeros((nmax, (sig.shape[0]-1)*2))        
        SMax = np.zeros((nmax,2))

        t = sig[0,:]
        #for i in range(1, sig.shape[0]):
        for i in range(2,3): # This should be a greater range if we want to get reflection data too. 
                sij = sig[i,:]
                

                maxInd = argrelextrema(np.absolute(sij), np.greater)
                ind = np.argpartition(np.absolute(sij[maxInd]), -nmax)[-nmax:]

                maxNInd = [maxInd[0][i] for i in ind]
                sijMax = [np.absolute(sij[i])/1e9 for i in maxNInd] # normalized by dividing by 1e9
                tMax_i = [t[i]*1e9 for i in maxNInd] # normalized by multiplying by 1e9
                
                #SMax[(i-1)*2,:] = sijMax
                #SMax[(i-1)*2-1,:] = tMax_i
		
                SMax[:,(i-1)*2-1] = sijMax
                SMax[:,(i-1)*2-2] = tMax_i
        return SMax

def predictFromModel(modelFileName, X_data, saver, logits, X_ph):
	with tf.Session() as sess:
		saver.restore(sess, modelFileName)
		Z = logits.eval(feed_dict={X_ph: X_data})
		y_pred = Z
	return y_pred

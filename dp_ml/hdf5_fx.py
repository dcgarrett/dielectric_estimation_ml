# Some helper functions for storing and retrieving large amounts of data in HDF5 format
	# David Garrett, October 2017

import h5py
import numpy as np
import dp_ml

from .dp_ml_fx import *


def createFile(dbName):
	""" Creates a new hdf5 file according to the groups below

		dbName is the name of the hdf5 you wish to create """

	f = h5py.File(dbName, 'w-')
	
	data_group = f.create_group("Data")
	model_group = f.create_group("Models")
	cal_group = f.create_group("Calibration")

	sim_group = data_group.create_group("Simulation")
	meas_group = data_group.create_group("Measurement")

	sim_cal_group = cal_group.create_group("Simulation")
	meas_cal_group = cal_group.create_group("Measurement")

	cwg_group = sim_group.create_group("CircWG")
	nahanni_group = sim_group.create_group("Nahanni")

	cwg_group_f = cwg_group.create_group('frequency')
	cwg_group_t = cwg_group.create_group('time')

	cwg_cal_group = sim_cal_group.create_group("CircWG")
	nahanni_cal_group = sim_cal_group.create_group("Nahanni")

	cwg_cal_group_f = cwg_cal_group.create_group('frequency')
	cwg_cal_group_t = cwg_cal_group.create_group('time')
	
	f.close()


def searchForEntry(dbName, dbHier, indFileName, freqOrTime = '/frequency/'):
	""" Searches through an HDF5 file for existence of an entry
	
		- dbName is the hdf5 file
		- dbHier is the hierarchical path within the database - e.g. '/Data/Simulations/CircWG'
		- indFileName  is the name of the recorded file according to your convention. Used to check existence 
	"""

	with h5py.File(dbName,'r') as f:
		entryThere =  dbHier + freqOrTime +  indFileName in f
		f.close()
		return entryThere


def newEntryFrequency(dbName, dbHier, indFileName, sepDist, eps, sig, freq, S_f):
	"""Creates a new entry with data in the frequency domain"""
	# dbName is the hdf5 file
	# dbHier is the hierarchical path within the database - e.g. '/Data/Simulations/CircWG'
	# indFileName is the name of the recorded file according to your convention. Used to check existence
	# sepDist is the separation distance. If multilayered, this is an array of each layer e.g. [2,20,2] 
	# eps is the permittivity. If multilayered, this is an array of each layer e.g. [30, 10, 30]
	# sig is the conductivity. If multilayered, this is an array of each layer e.g. [10, 2, 10]
	# f is the array of recorded frequency - should be 5001 length for simulation or 1000 length for meas
	# S_f is the 8xlen(f) array of S11mag(dB), S11dang, S21mag(dB), S21dang, S12mag(dB), S12dang, S22mag(dB), S22dang 

	with h5py.File(dbName,'r+') as f:
		dset_f = f.create_dataset(dbHier + '/frequency/' + indFileName.replace('.xls',''),( max(freq.shape),9,),dtype='float64',compression='gzip')
		dset_f[:,0] = freq.T
		dset_f[:,1:9] = S_f.T
		dset_f.attrs['sepDist'] = sepDist
		dset_f.attrs['eps'] = eps
		dset_f.attrs['sig'] = sig
		f.close()


def newEntryTime(dbName, dbHier, indFileName, sepDist, eps, sig, t, S_t):
		"""Creates a new entry with data in the frequency domain"""
		# dbName is the hdf5 file
		# dbHier is the hierarchical path within the database - e.g. '/Data/Simulations/CircWG'
		# indFileName is the name of the recorded file according to your convention. Used to check existence
		# sepDist is the separation distance. If multilayered, this is an array of each layer e.g. [2,20,2]
		# eps is the permittivity. If multilayered, this is an array of each layer e.g. [30, 10, 30]
		# sig is the conductivity. If multilayered, this is an array of each layer e.g. [10, 2, 10]
		# t is the array of the signal in the time domain
		# S_t is the 4xlen(t) array of S11, S21, S12, S22 in the time domain

		with h5py.File(dbName,'r+') as f:
				dset_t = f.create_dataset(dbHier + '/time/' + indFileName.replace('.xls',''),(max(t.shape),5,),dtype='float64',compression='gzip')
				dset_t[:,0] = t.T
				dset_t[:,1:5] = S_t.T
				dset_t.attrs['sepDist'] = sepDist
				dset_t.attrs['eps'] = eps
				dset_t.attrs['sig'] = sig
				f.close()

def readEntry(dbName, dbHier, indFileName, freqOrTime = '/frequency/'):

	with h5py.File(dbName,'r') as f:
		entry = f[dbHier+freqOrTime+indFileName]
		entry = entry[:]
		f.close()
		return entry


def convertXLtoHDF5(dataPath, dbName, dbHier, timeDomain = True):
	"""Imports XLS files, checks if the data exists yet in the HDF5 file, and adds it if not"""
	from pathlib import Path

	dirContents = os.listdir(dataPath)
	if '.DS_Store' in dirContents:
			dirContents.remove('.DS_Store')

	if not Path(dbName).is_file():
			createFile(dbName)

	for i in range(0,len(dirContents)):
			eps, sig, dist = dp_ml.get_params_from_filename(dirContents[i])

			# check if entry exists
			with h5py.File(dbName,'r') as f:
				if dbHier + '/frequency/' + dirContents[i] in f:
					print("Entry exists in db. Continuing")
					continue
					f.close()

			f, S_f, S_comp = dp_ml.importSingleXL(dataPath,dirContents[i])
			# dimensions should be (numParams, numFrequencyPoints)
			S_f_dim = np.shape(S_f)
			if S_f_dim[0] > S_f_dim[1]:
				S_f = np.transpose(S_f)
			# add to the HDF5 database
			dp_ml.newEntryFrequency(dbName, dbHier, dirContents[i], dist, eps, sig, f[:-1], S_f)

			if timeDomain:
				# dimensions should be (numParams, numFrequencyPoints)
				t, S_t = dp_ml.S_compToTimeDomain(f, S_comp)
				S_t_dim = np.shape(S_t)
				if S_f_dim[0] > S_f_dim[1]:
					S_t = np.transpose(S_t)
				dp_ml.newEntryTime(dbName, dbHier, dirContents[i], dist, eps, sig, t, S_t)

def importCTI(fileName, numF=1000):
    file_obj = open(dp_ml.measPath+"/"+fileName,'r')


    line = file_obj.readline()
    while line != "VAR_LIST_BEGIN":
        line = file_obj.readline()
        line = line.strip()

    f = np.zeros((numF,))
    i = 0
    line = file_obj.readline()
    line = line.strip()
    while line != "VAR_LIST_END":
        f[i] = float(line)
        i = i+1
        line = file_obj.readline()
        line = line.strip()

    line = file_obj.readline()


    S = np.zeros((8,numF))
    S_comp = np.zeros((4,numF),dtype=np.complex)
    S_t = np.zeros((4,numF))

    for j in range(0,4):
        i = 0
        line = file_obj.readline()
        line = line.strip()
        while line != "END":
            S[2*j,i], S[2*j+1,i] = dp_ml.realimag_to_magphase(float(line.split(',')[0]),float(line.split(',')[1]))
            i = i+1
            line = file_obj.readline()
            line = line.strip()
    
        sii_comp = S[2*j,:] * (np.cos(S[2*j+1,:]) + 1j*np.sin(S[2*j+1,:]))
        S_comp[j,:] = sii_comp
    
        line = file_obj.readline()
    
    t, S_t = dp_ml.S_compToTimeDomain(f, S_comp)    
    return f, S_comp, S, t, S_t

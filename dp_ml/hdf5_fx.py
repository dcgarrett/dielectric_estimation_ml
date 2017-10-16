import h5py
import numpy as np

def createFile(dbName):
# Creates a new hdf5 file according to the groups below
	# dbName is the name of the hdf5 you wish to create

	f = h5py.File(dbName, 'w-')
	
	data_group = f.create_group("Data")
	model_group = f.create_group("Models")

	sim_group = data_group.create_group("Simulations")
	meas_group = data_group.create_group("Measurements")

	cwg_group = sim_group.create_group("CircWG")
	nahanni_group = sim_group.create_group("Nahanni")

	cwg_group_f = cwg_group.create_group('frequency')
	cwg_group_t = cwg_group.create_group('time')



def searchForEntry(dbName, dbHier, indFileName):
# Searches through an HDF5 file for existence of an entry
	# dbName is the hdf5 file
	# dbHier is the hierarchical path within the database - e.g. '/Data/Simulations/CircWG'
	# indFileName  is the name of the recorded file according to your convention. Used to check existence 

	with h5py.File(dbName,'r') as f:
		return dbHier + '/frequency/' +  indFileName in f



def newEntryFrequency(dbName, dbHier, indFileName, sepDist, eps, sig, f, S_f):
# Creates a new entry with data in the frequency domain
	# dbName is the hdf5 file
	# dbHier is the hierarchical path within the database - e.g. '/Data/Simulations/CircWG'
	# indFileName is the name of the recorded file according to your convention. Used to check existence
	# sepDist is the separation distance. If multilayered, this is an array of each layer e.g. [2,20,2] 
	# eps is the permittivity. If multilayered, this is an array of each layer e.g. [30, 10, 30]
	# sig is the conductivity. If multilayered, this is an array of each layer e.g. [10, 2, 10]
	# f is the array of recorded frequency - should be 5001 length for simulation or 1000 length for meas
	# S_f is the 8xlen(f) array of S11mag(dB), S11dang, S21mag(dB), S21dang, S12mag(dB), S12dang, S22mag(dB), S22dang 

	with h5py.File(dbName,'a') as f:
		dset_t = f.create_dataset(dbHier + '/time/' + indFileName,(5, len(t),))
		dset_t[0,:] = t
		dset_t[1:5,:] = S_t
		dset_t.attrs['sepDist'] = sepDist
		dset_t.attrs['eps'] = eps
		dset_t.attrs['sig'] = sig


def newEntryTime(dbName, dbHier, indFileName, sepDist, eps, sig, t, S_t):
# Creates a new entry with data in the frequency domain
        # dbName is the hdf5 file
        # dbHier is the hierarchical path within the database - e.g. '/Data/Simulations/CircWG'
        # indFileName is the name of the recorded file according to your convention. Used to check existence
        # sepDist is the separation distance. If multilayered, this is an array of each layer e.g. [2,20,2]
        # eps is the permittivity. If multilayered, this is an array of each layer e.g. [30, 10, 30]
        # sig is the conductivity. If multilayered, this is an array of each layer e.g. [10, 2, 10]
        # t is the array of the signal in the time domain
        # S_t is the 4xlen(t) array of S11, S21, S12, S22 in the time domain

        with h5py.File(dbName,'a') as f:
                dset_t = f.create_dataset(dbHier + '/time/' + indFileName,(5, len(t),))
                dset_t[0,:] = t
                dset_t[1:5,:] = S_t
                dset_t.attrs['sepDist'] = sepDist
                dset_t.attrs['eps'] = eps
                dset_t.attrs['sig'] = sig

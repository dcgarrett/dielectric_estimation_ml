""" Some helper functions for the dielectric property estimation toolbox
	David Garrett, October 2017 """

from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import numpy as np
import pandas
import os
import re
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import string
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.signal import lfilter
import scipy.io as sio
import tensorflow as tf
import platform
import h5py
from scipy.signal import tukey

from .config import * # import all the config parameters natively

from .hdf5_fx import *
from .tsar_math import *
from .test_dpml import *



# Start the functions
def say_something():
	return (u'What a day to be alive! \n Arnold Schwarzenegger, 2017')

def get_params_from_filename(fileName): # Handles both int or float values, get eps and sig from filename
    """Get permittivity, conductivity, and separation distance from a filename.
       Example formatted filename: eps20_sig30_50mm.xls
       For calibration procedures, gives '0' for every param. e.g. Thru.xls"""
    #print(fileName)
    if ("Thru" in fileName) or ("Reflect" in fileName): # calibration procedures
        return 0, 0, 0 
    
    strList = fileName.split("_")
    nums = re.findall('\d+', fileName)
    
    if re.findall("\d+\.\d*",strList[0]):
        nums0 = re.findall("\d+\.\d*",strList[0])
    else:
        nums0 = re.findall("\d+",strList[0])
    eps = nums0
    
    if re.findall("\d+\.\d*",strList[1]):
        nums1 = re.findall("\d+\.\d*",strList[1])
    else:
        nums1 = re.findall("\d+",strList[1])
    sig = nums1
    
    if re.findall("\d+\.\d*",strList[2]):
        nums2 = re.findall("\d+\.\d*",strList[2])
    else:
        nums2 = re.findall("\d+",strList[2])
    dist = nums2
    return float(eps[0]), float(sig[0]), float(dist[0])

def realimag_to_magphase(real, imag):
    """Converts real and imaginary values to a magnitude and phase [rad] value."""
    mag = np.absolute([real + 1j*imag])
    phase = np.angle([real+ 1j*imag ])
    return mag, phase

def mag2db(mag):
    """Converts a magnitude value to a db representation (using 20*log10(mag))"""
    db = 20.*np.log10(mag)
    return db

# Load data
def importFromDir(dataPath):
    """Imports data from a given datapath"""
    dirContents = os.listdir(dataPath)
    if '.DS_Store' in dirContents:
        dirContents.remove('.DS_Store')
    
    #initialize data
    eps = np.zeros(len(dirContents))
    sig = np.zeros(len(dirContents))
    dist = np.zeros(len(dirContents))
    s11_real = np.zeros((len(dirContents),numFreqPoints))
    s11_imag = np.zeros((len(dirContents),numFreqPoints))
    s11_mag = np.zeros((len(dirContents),numFreqPoints))
    s11_ang = np.zeros((len(dirContents),numFreqPoints))
    s12_real = np.zeros((len(dirContents),numFreqPoints))
    s12_imag = np.zeros((len(dirContents),numFreqPoints))
    
    s21_real = np.zeros((len(dirContents),numFreqPoints))
    s21_imag = np.zeros((len(dirContents),numFreqPoints))
    
    s22_real = np.zeros((len(dirContents),numFreqPoints))
    s22_imag = np.zeros((len(dirContents),numFreqPoints))
    
    s11_comp = np.zeros((len(dirContents),numFreqPoints), dtype=np.complex_)
    s12_comp = np.zeros((len(dirContents),numFreqPoints), dtype=np.complex_)
    s21_comp = np.zeros((len(dirContents),numFreqPoints), dtype=np.complex_)
    s22_comp = np.zeros((len(dirContents),numFreqPoints), dtype=np.complex_)
    
    s12_mag = np.zeros((len(dirContents),numFreqPoints))
    s12_ang = np.zeros((len(dirContents),numFreqPoints))
    s11_dang = np.zeros((len(dirContents),numFreqPoints-1))
    s12_dang =  np.zeros((len(dirContents),numFreqPoints-1))
    
    S_train = np.zeros((2,2,numFreqPoints, len(dirContents)), dtype=np.complex_) # full complex S-matrix
    
    
    # import S11 and S12 for each simulation
    for i in range(0,len(dirContents)):
        eps[i], sig[i], dist[i] = get_params_from_filename(dirContents[i])
        #print(eps[i])
    
        fullFile_i = dataPath + '/' + dirContents[i]
        data_i = pandas.read_excel(fullFile_i, sheetname="Scattering Sij(f)")
        freqStep = data_i.values[1,0] - data_i.values[2,0]
        
        f = data_i.values[:,0]
    
        s11_real[i,:] = data_i.values[:,1]
        s11_imag[i,:] = data_i.values[:,2]
    
        s11_mag[i,:], s11_ang[i,:] = realimag_to_magphase(s11_real[i,:], s11_imag[i,:])
        # unwrap phase:
        s11_ang[i,:] = np.unwrap(s11_ang[i,:])
        #s11_dang[i,:] = np.diff(s11_ang[i,:])*freqStep
        s11_dang[i,:] = np.diff(s11_ang[i,:])
    
        s12_real[i,:] = data_i.values[:,3]
        s12_imag[i,:] = data_i.values[:,4]
    
        s12_mag[i,:], s12_ang[i,:] = realimag_to_magphase(s12_real[i,:], s12_imag[i,:])
        s12_ang[i,:] = np.unwrap(s12_ang[i,:])
        #s12_dang[i,:] = np.diff(s12_ang[i,:])*freqStep
        s12_dang[i,:] = np.diff(s12_ang[i,:])
        
        s21_real[i,:] = data_i.values[:,5]
        s21_imag[i,:] = data_i.values[:,6]
        
        s22_real[i,:] = data_i.values[:,7]
        s22_imag[i,:] = data_i.values[:,8]
        
        s11_comp[i,:] = s11_real[i,:] + s11_imag[i,:]*1j
        s12_comp[i,:] = s12_real[i,:] + s12_imag[i,:]*1j
        s21_comp[i,:] = s21_real[i,:] + s21_imag[i,:]*1j
        s22_comp[i,:] = s22_real[i,:] + s22_imag[i,:]*1j
        
        for k in range(0,numFreqPoints):
            S_train[:,:,k,i] = np.matrix([[s11_comp[i,k],s12_comp[i,k]], [s21_comp[i,k],s22_comp[i,k]]])
        
        
    return f, s11_mag, s11_ang, s11_dang, s12_mag, s12_ang, s12_dang, eps, sig, dist, S_train

def reshapeData(eps,sig,s11_mag,s11_ang,s11_dang,s12_mag,s12_ang,s12_dang):
    print("Is reshape even used? Yes it is!!")
    epsList = sorted(set(eps))
    sigList = sorted(set(sig))

    j = 0
    lenEr = len(epsList)
    lenSig = len(sigList)
    lenF = len(f)-1
    
    mag11 = np.zeros((lenF,lenEr,lenSig))
    dang11 = np.zeros((lenF,lenEr,lenSig))
    
    mag12 = np.zeros((lenF,lenEr,lenSig))
    dang12 = np.zeros((lenF,lenEr,lenSig))
    
    ang12 = np.zeros((lenF,lenEr,lenSig))

    for freqInd in range(0,len(f)-1):
        for i in range(0,len(eps)):
                # Select some frequency value, see how mag and ang are affected
                mag11[freqInd,epsList.index(eps[i]), sigList.index(sig[i])] = s11_mag[i,freqInd]
                dang11[freqInd,epsList.index(eps[i]), sigList.index(sig[i])] = s11_dang[i,freqInd]
                
                mag12[freqInd,epsList.index(eps[i]), sigList.index(sig[i])] = s12_mag[i,freqInd]
                dang12[freqInd,epsList.index(eps[i]), sigList.index(sig[i])] = s12_dang[i,freqInd]
                ang12[freqInd,epsList.index(eps[i]), sigList.index(sig[i])] = s12_ang[i,freqInd]
                
    return epsList, sigList, mag11, dang11, mag12, dang12, ang12


def plotEstimate(eps_true, eps_est, sig_true, sig_est): # eps_true is 1-D
    plt.plot(f[np.asarray(fRange)],eps_est)
    #if len(eps_true) == 1:
    #    plt.plot([2e9, 12e9],[eps_true,eps_true])
    #else:
    #    plt.plot(f[np.asarray(fRange)],eps_true[np.asarray(fRange)])
    plt.plot([2e9, 12e9],[eps_true,eps_true])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Permittivity')
    plt.legend(['Estimation','True'])
    axes = plt.gca()
    axes.set_ylim([eps_true-10,eps_true+10])
    plt.savefig(figPath + '/epsEst_ANN_TrueApprox' + str(int(np.mean(eps_true))) +  '.png')
    plt.show()

    n = 100 # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b,a,sig_est)
    plt.plot(f[np.asarray(fRange)],sig_est)
    #if len(eps_true) == 1:
    #    plt.plot([2e9, 12e9],[sig_true,sig_true])
    #else:
    #    plt.plot(f[np.asarray(fRange)],sig_true[np.asarray(fRange)])
    plt.plot([2e9, 12e9],[sig_true,sig_true])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Conductivity [S/m]')
    plt.legend(['Estimation','True'])
    axes = plt.gca()
    axes.set_ylim([sig_true-2,sig_true+2])
    plt.savefig(figPath + '/sigEst_ANN_TrueApprox' + str(int(np.mean(sig_true))) + '.png')
    plt.show()
    

## Generate model

def generateMultiMLP(X_train_minmax, y_train_minmax, hidden_layer_size = [20,10], max_iter = 10000, solver = 'lbfgs', activation = 'logistic'):
    # Make sure that X and y are scaled before using this function using the sklearn.preprocessing.MinMaxScaler tool
    multReg = MultiOutputRegressor(MLPRegressor(activation=activation,  random_state = 323, hidden_layer_sizes=hidden_layer_size, max_iter = max_iter,solver=solver))
    multReg.fit(X_train_minmax, y_train_minmax)
    
    y_pred = multReg.predict(X_train_minmax)
    error = np.sum(np.absolute(y_train_minmax - y_pred))
    
    return multReg, error

## Predict from model

def predictFromMLP(X_test_minmax, multReg):
    y_pred = multReg.predict(X_test_minmax)
    
    return y_pred

def S_to_T(S):
    """ Converts S-parameters to T-parameters"""
    x11 = S[0,1]*S[1,0]-S[0,0]*S[1,1]
    x12 = S[0,0]
    x21 = -S[1,1]
    x22 = 1
    T = (1.0/S[1,0])*np.matrix([[x11, x12], [x21, x22]])
    
    return T
    
def T_to_S(T):
    """ Converts T-parameters back to S-parameters"""
    x11 = T[0,1]
    x12 = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    x21 = 1
    x22 = -T[1,0]
    S = (1/T[1,1])*np.matrix([[x11, x12], [x21, x22]])
    
    return S

def loadCalibration_WG():
    """Load calibration data for the waveguide"""
    calibration_WG = sio.loadmat(calPath + 'CylindricalWG_OP.mat')

    O_wg = calibration_WG['O_sim']
    P_wg = calibration_WG['P_sim']
    f_wg = calibration_WG['f_wg']
  
    O_wg_T = np.zeros((2,2,len(f_wg)),dtype=np.complex_)
    P_wg_T = np.zeros((2,2,len(f_wg)),dtype=np.complex_)
    
    O_wg_T_inv = np.zeros((2,2,len(f_wg)),dtype=np.complex_)
    P_wg_T_inv = np.zeros((2,2,len(f_wg)),dtype=np.complex_)

    for i in range(0,len(f_wg)):
        O_wg_T[:,:,i] = S_to_T(O_wg[:,:,i])
        P_wg_T[:,:,i] = S_to_T(P_wg[:,:,i])
        
        O_wg_T_inv[:,:,i] = np.linalg.inv(O_wg_T[:,:,i])
        P_wg_T_inv[:,:,i] = np.linalg.inv(P_wg_T[:,:,i])
    
        
    return O_wg, P_wg, O_wg_T, P_wg_T, O_wg_T_inv, P_wg_T_inv

def performCalibration_WG(S_wg, O_wg_T_inv, P_wg_T_inv):
    """Perform antenna calibration with the waveguide O and P matrices"""
    # T_cal = O_t^-1 * T_meas * P_t^-1
    # Note - perform inverse of O and P first!
    T_wg = np.zeros((2,2,S_wg.shape[2]),dtype = np.complex_)
    T_cal = np.zeros((2,2,S_wg.shape[2]),dtype = np.complex_)
    S_cal = np.zeros((2,2,S_wg.shape[2]),dtype = np.complex_)
    
    for i in range(0, S_wg.shape[2]):
        T_wg[:,:,i] = S_to_T(S_wg[:,:,i])
        T_cal[:,:,i] = np.multiply(O_wg_T_inv[:,:,i], T_wg[:,:,i], P_wg_T_inv[:,:,i])
        S_cal[:,:,i] = T_to_S(T_cal[:,:,i])
    
    return S_cal

def SMatrixToArrays(S_mtx):
    # changed second dim from numFreqPoints to S_mtx.shape([2])
    s11_real = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s11_imag = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s11_mag = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s11_ang = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s12_real = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s12_imag = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    
    
    s12_mag = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s12_ang = np.zeros((S_mtx.shape[3],S_mtx.shape[2]))
    s11_dang = np.zeros((S_mtx.shape[3],S_mtx.shape[2]-1))
    s12_dang =  np.zeros((S_mtx.shape[3],S_mtx.shape[2]-1))
    
    
    # import S11 and S12 for each simulation
    for i in range(0,S_mtx.shape[3]): 
        s11_real[i,:] = np.real(S_mtx[0,0,:,i])
        s11_imag[i,:] = np.imag(S_mtx[0,0,:,i])
    
        s11_mag[i,:], s11_ang[i,:] = realimag_to_magphase(s11_real[i,:], s11_imag[i,:])
        # unwrap phase:
        s11_ang[i,:] = np.unwrap(s11_ang[i,:])
        #s11_dang[i,:] = np.diff(s11_ang[i,:])*freqStep
        s11_dang[i,:] = np.diff(s11_ang[i,:])
    
        s12_real[i,:] = np.real(S_mtx[0,1,:,i])
        s12_imag[i,:] = np.real(S_mtx[0,1,:,i])
    
        s12_mag[i,:], s12_ang[i,:] = realimag_to_magphase(s12_real[i,:], s12_imag[i,:])
        s12_ang[i,:] = np.unwrap(s12_ang[i,:])
        #s12_dang[i,:] = np.diff(s12_ang[i,:])*freqStep
        s12_dang[i,:] = np.diff(s12_ang[i,:])
        
    return s11_mag, s11_ang, s11_dang, s12_mag, s12_ang, s12_dang

def plotFromS(f,S):
    plt.plot(f,mag2db(np.absolute(S[1,0,:,:])))

def importSingleXL(dataPath, fileName):
    fullFile_i = dataPath + '/' + fileName
    data_i = pandas.read_excel(fullFile_i, sheetname="Scattering Sij(f)")
    freqStep = data_i.values[1,0] - data_i.values[2,0]

    f = data_i.values[:,0]

    s11_real = data_i.values[:,1]
    s11_imag = data_i.values[:,2]

    s11_mag, s11_ang = realimag_to_magphase(s11_real, s11_imag)
    # unwrap phase:
    s11_ang = np.unwrap(s11_ang)
    #s11_dang[i,:] = np.diff(s11_ang[i,:])*freqStep
    s11_dang = np.diff(s11_ang)

    s21_real = data_i.values[:,3]
    s21_imag = data_i.values[:,4]

    s21_mag, s21_ang = realimag_to_magphase(s21_real, s21_imag)
    s21_ang = np.unwrap(s21_ang)
    #s21_dang = np.diff(s21_ang)*freqStep
    s21_dang = np.diff(s21_ang)


    s12_real = data_i.values[:,5]
    s12_imag = data_i.values[:,6]

    s12_mag, s12_ang = realimag_to_magphase(s12_real, s12_imag)
    # unwrap phase:
    s12_ang = np.unwrap(s12_ang)
    #s12_dang[i,:] = np.diff(s11_ang[i,:])*freqStep
    s12_dang = np.diff(s12_ang)

    s22_real = data_i.values[:,7]
    s22_imag = data_i.values[:,8]

    s22_mag, s22_ang = realimag_to_magphase(s22_real, s22_imag)
    s22_ang = np.unwrap(s22_ang)
    #s12_dang = np.diff(s12_ang)*freqStep
    s22_dang = np.diff(s22_ang)

    S_f = np.zeros((8, max(f.shape)))
    S_f = np.vstack((s11_mag[0,:-1], s11_dang[0,:], s21_mag[0,:-1], s21_dang[0,:], s12_mag[0,:-1], s12_dang[0,:], s22_mag[0,:-1], s22_dang[0,:]))

    s11_comp = s11_real[:-1] + s11_imag[:-1]*1j
    s21_comp = s21_real[:-1] + s21_imag[:-1]*1j
    s12_comp = s12_real[:-1] + s12_imag[:-1]*1j
    s22_comp = s22_real[:-1] + s22_imag[:-1]*1j

    S_comp = np.vstack((s11_comp, s21_comp, s12_comp, s22_comp))

    return f, S_f, S_comp

def S_compToTimeDomain(freq, S_comp,tukeywin=0.5):
	#freq is the array of frequency
	#S_comp is the 4*len(freq) array of complex s11,s21,s12,s22
	for i in range(0,4):
		#S_comp[i,:] = tukey(len(freq)-1,tukeywin)*S_comp[i,:];
		S_comp[i,:] = tukey(len(freq),tukeywin)*S_comp[i,:];
	
	t, s11_t = inverseczt_charlotte(freq[:-1], S_comp[0,:])
	t, s21_t = inverseczt_charlotte(freq[:-1], S_comp[1,:])
	t, s12_t = inverseczt_charlotte(freq[:-1], S_comp[2,:])
	t, s22_t = inverseczt_charlotte(freq[:-1], S_comp[3,:])

	S_t = np.vstack((s11_t,s21_t,s12_t,s22_t))
	
	return t, S_t

""" Some helper functions for the dielectric property estimation toolbox.
Mostly relating to data importing and formatting """

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


def say_something():
	return (u'What a day to be alive! \n Arnold Schwarzenegger, 2017')


def get_params_from_filename(fileName):
    """Get permittivity, conductivity, and separation distance from an Excel filename.

    Inputs:

        - fileName -- file name of the data. Formatted as: 'epsXX_sigXX_XXmm.xls' Example formatted filename: eps20_sig30_50mm.xls

    Outputs:

        - eps -- permittivity 
        - sig -- conductivity
        - dist -- separation distance

    Note: for calibration procedures, gives '0' for every param. e.g. Thru.xls, Reflect.xls"""

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
    """Converts real and imaginary values to a magnitude and phase [rad] value.
    Works for arrays or single points

    Inputs:

        - real -- real part of the signal
        - imag -- imaginary part of the signal

    Outputs:

        - mag -- magnitude
        - phase -- phase [rad]"""

    mag = np.absolute([real + 1j*imag])
    phase = np.angle([real+ 1j*imag ])
    return mag, phase

def mag2db(mag):
    """Converts a magnitude value to a db representation (using 20*log10(mag))
    
    Inputs:

        - mag -- signal magnitude (generally from 0-1)

    Outputs:

        - db -- decibel for of the magnitude"""
    db = 20.*np.log10(mag)
    return db


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


def S_to_T(S):
    """ Converts S-parameters to T-parameters

    Inputs:

        - S -- 2x2xnumF S-parameters

    Outputs:

        - T -- 2x2xnumF T-parameters"""
    x11 = S[0,1]*S[1,0]-S[0,0]*S[1,1]
    x12 = S[0,0]
    x21 = -S[1,1]
    x22 = 1
    T = (1.0/S[1,0])*np.matrix([[x11, x12], [x21, x22]])
    
    return T
    
def T_to_S(T):
    """ Converts T-parameters to S-parameters

    Inputs:

        - T -- 2x2xnumF T-parameters

    Outputs:

        - S -- 2x2xnumF S-parameters"""

    x11 = T[0,1]
    x12 = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    x21 = 1
    x22 = -T[1,0]
    S = (1/T[1,1])*np.matrix([[x11, x12], [x21, x22]])
    
    return S

def loadCalibration_WG():
    """Load calibration data for the waveguide.
    Uses the global calPath variable defined in the dp_ml installation to find the O and P
    matrices for calibration.
    
    Outputs:

        - O -- O matrix for port 1 of calibration for simulation
        - P -- P matrix for port 2 of calibration for simulation
    """

    calibration_WG = sio.loadmat(calPath + 'CylindricalWG_OP.mat')

    O_wg = calibration_WG['O_sim']
    P_wg = calibration_WG['P_sim']
    f_wg = calibration_WG['f_wg']    
        
    return O_wg, P_wg

def performCalibration(S, O, P):
    """Perform antenna calibration with the waveguide O and P matrices

    Inputs:

        - S -- 2x2xnumF complex signal of interest from either simulation or measurement
        - O -- 2x2xnumF complex calibration matrix for port 1
        - P -- 2x2xnumF complex calibration matrix for port 2

    Outputs:

        - S_cal -- 2x2xnumF complex calibrated signal of interest (up to antenna aperture)"""

    T_wg = np.zeros((2,2,S_wg.shape[2]),dtype = np.complex_)
    T_cal = np.zeros((2,2,S_wg.shape[2]),dtype = np.complex_)
    S_cal = np.zeros((2,2,S_wg.shape[2]),dtype = np.complex_)

    O_wg_T = np.zeros((2,2,len(f_wg)),dtype=np.complex_)
    P_wg_T = np.zeros((2,2,len(f_wg)),dtype=np.complex_)
    
    O_wg_T_inv = np.zeros((2,2,len(f_wg)),dtype=np.complex_)
    P_wg_T_inv = np.zeros((2,2,len(f_wg)),dtype=np.complex_)

    for i in range(0,S_wg.shape[2]):
        O_wg_T[:,:,i] = S_to_T(O_wg[:,:,i])
        P_wg_T[:,:,i] = S_to_T(P_wg[:,:,i])
        
        O_wg_T_inv[:,:,i] = np.linalg.inv(O_wg_T[:,:,i])
        P_wg_T_inv[:,:,i] = np.linalg.inv(P_wg_T[:,:,i])
    
    for i in range(0, S_wg.shape[2]):
        T_wg[:,:,i] = S_to_T(S_wg[:,:,i])
        T_cal[:,:,i] = np.multiply(O_wg_T_inv[:,:,i], T_wg[:,:,i], P_wg_T_inv[:,:,i])
        S_cal[:,:,i] = T_to_S(T_cal[:,:,i])
    
    return S_cal


def plotFromS(f,S):
    plt.plot(f,mag2db(np.absolute(S[1,0,:,:])))


def S_compToTimeDomain(freq, S_comp,tukeywin=0.5):
	#freq is the array of frequency
	#S_comp is the 4*len(freq) array of complex s11,s21,s12,s22
	for i in range(0,4):
		if len(freq) == len(S_comp[i,:]):
			S_comp[i,:] = tukey(len(freq),tukeywin)*S_comp[i,:]
		else:
			S_comp[i,:] = tukey(len(freq)-1,tukeywin)*S_comp[i,:]
	
	t, s11_t = inverseczt_charlotte(freq[:-1], S_comp[0,:])
	t, s21_t = inverseczt_charlotte(freq[:-1], S_comp[1,:])
	t, s12_t = inverseczt_charlotte(freq[:-1], S_comp[2,:])
	t, s22_t = inverseczt_charlotte(freq[:-1], S_comp[3,:])

	S_t = np.vstack((s11_t,s21_t,s12_t,s22_t))
	
	return t, S_t

def polarToRect(radii, angles):
    return radii * np.exp(1j*angles)






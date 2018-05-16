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

def importSingleXL(dataPath, fileName):
    """Reads in one excel file from a given dataPath

    Inputs:

        - dataPath ---  """
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

def importCTI_4Files(fileName1, path = measPath, numF=1000):
    S = np.zeros((8,numF))
    S_comp = np.zeros((4,numF),dtype=np.complex)
    sii_comp = np.zeros((numF,),dtype=np.complex)
    f = np.zeros((numF,))
    
    for j in range(0,4):
        measNum = (int(fileName1[-7:-4]) + j)
        if measNum < 10:
            zeroStr = '00'
        elif 10 <= measNum < 100:
            zeroStr = '0'
        else:
            zeroStr = ''
        fileName_i = fileName1[:-7] + zeroStr + str(int(fileName1[-7:-4]) + j) + '.cti'

        file_obj = open(path+"/"+fileName_i,'r')


        line = file_obj.readline()
        
        if j == 0:
            while not '!ANTPOS_TX:' in line:
                line = file_obj.readline()
                line = line.strip()
            distances = line.replace('!ANTPOS_TX: 0E+0 ','')
            distances = distances.replace(' 0E+0 0 90 270','')
            distance = np.absolute(float(distances))
            
        while line != "VAR_LIST_BEGIN":
            line = file_obj.readline()
            line = line.strip()

        
        i = 0
        line = file_obj.readline()
        line = line.strip()
        while line != "VAR_LIST_END":
            if j == 0:
                f[i] = float(line)
            i = i+1
            line = file_obj.readline()
            line = line.strip()

        line = file_obj.readline()

        # Read S data
        i = 0
        line = file_obj.readline()
        line = line.strip()
        while line != "END":
            #sii_comp[i] = float(line.split(',')[0]) + 1j*float(line.split(',')[1]) 
            S[2*j,i], S[2*j+1,i] = dp_ml.realimag_to_magphase(float(line.split(',')[0]),float(line.split(',')[1]))
            i = i+1
            line = file_obj.readline()
            line = line.strip()

        #sii_comp = S[2*j,:] * (np.cos(S[2*j+1,:]) + 1j*np.sin(S[2*j+1,:]))
        sii_comp = np.zeros((numF,),dtype=np.complex)
        sii_comp = polarToRect(S[2*j,:], S[2*j+1,:]) 
        S_comp[j,:] = sii_comp

        line = file_obj.readline()

    t, S_t = dp_ml.S_compToTimeDomain(f, S_comp)

    S_comp_2x2 = np.zeros((2,2,numF),dtype=np.complex)
    S_comp_2x2[0,0,:] = S_comp[0,:]
    S_comp_2x2[0,1,:] = S_comp[1,:]
    S_comp_2x2[1,0,:] = S_comp[2,:]
    S_comp_2x2[1,1,:] = S_comp[3,:]

    return f, S_comp_2x2, S, t, S_t, distance




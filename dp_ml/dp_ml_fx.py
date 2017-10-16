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
import IPython
import string
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.signal import lfilter
import scipy.io as sio
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import tensorflow as tf
import platform
import h5py

# Assign appropriate directories
dataPath_win = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/"
dataPath_mac = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/"

if (platform.system() == "Windows") & os.path.isdir(dataPath_win): # Dave computer in ICT
        dataPath = dataPath_win
        calPath = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        figPath = "C:/Users/dgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
	HDF5Path = "C:/Users/dgarrett/Google Drive/Work/Software/MachineLearning/Data"
        print("Windows detected, data path found")
elif (platform.system() == "Darwin") & os.path.isdir(dataPath_mac): # Dave Mac
        dataPath = dataPath_mac
        calPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        figPath = "/Users/davidgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
	HDF5Path = "/Users/davidgarrett/Google Drive/Work/Software/MachineLearning/Data"
        print("Mac detected, data path found")
else:
        print("Data cannot be located - package will be used for predicting from existing models.")

numFreqPoints = 5001
freqInt = 50



# Start the functions


def say_something():
	return (u'What a day to be alive! \n Arnold Schwarzenegger, 2017')

def reset_graph(seed=42): # make notebook stable across runs
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def get_params_from_filename(fileName): # Handles both int or float values, get eps and sig from filename
    print(fileName)
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
    mag = np.absolute([real + 1j*imag])
    phase = np.angle([real+ 1j*imag ])
    return mag, phase

def mag2db(mag):
    db = 20.*np.log10(mag)
    return db

# Load data
def importFromDir(dataPath):
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
    # Converts S-parameters to T-parameters
    x11 = S[0,1]*S[1,0]-S[0,0]*S[1,1]
    x12 = S[0,0]
    x21 = -S[1,1]
    x22 = 1
    T = (1.0/S[1,0])*np.matrix([[x11, x12], [x21, x22]])
    
    return T
    
def T_to_S(T):
    # Converts T-parameters back to S-parameters
    x11 = T[0,1]
    x12 = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    x21 = 1
    x22 = -T[1,0]
    S = (1/T[1,1])*np.matrix([[x11, x12], [x21, x22]])
    
    return S

def loadCalibration_WG():
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



import h5py

def convert_XLtoHDF5(dataPath):
	# Imports XLS files, checks if the data exists yet in the HDF5 file, and adds it if not
	f = h5py.File('sim4life_cwg_data.hdf5','w')

	dirContents = os.listdir(dataPath)
	if '.DS_Store' in dirContents:
		dirContents.remove('.DS_Store')

	

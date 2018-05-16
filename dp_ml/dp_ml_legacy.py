""" Legacy code for dp_ml """

def importFromDir(dataPath):
    """Imports data from a given datapath.
    Data should be stored in separate .xls files.
    In each file"""
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
"""Assign global variables for use in the dp_ml package"""
import platform
import os

numFreqPoints = 5001
freqInt = 50

# Assign appropriate directories
dataPath_win = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/"
dataPath_mac = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/"

if (platform.system() == "Windows") & os.path.isdir(dataPath_win): # Dave computer in ICT
        dataPath = dataPath_win
        calPath = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        figPath = "C:/Users/dgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
        HDF5Path = "C:/Users/dgarrett/Google Drive/Work/Software/machine_learning/data"
        HDF5Filename = 'dp_ml_data.hdf5'
        print("Windows detected, data path found")
elif (platform.system() == "Darwin") & os.path.isdir(dataPath_mac): # Dave Mac
        dataPath = dataPath_mac
        calPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        figPath = "/Users/davidgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
        HDF5Path = "/Users/davidgarrett/Google Drive/Work/Software/machine_learning/data"
        HDF5Filename = 'dp_ml_data.hdf5'
        print("Macintosh detected, data path found")
else:
        print("Data cannot be located - package will be used for predicting from existing models.")

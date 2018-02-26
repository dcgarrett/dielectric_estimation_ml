"""Assign global variables for use in the dp_ml package"""
import platform
import os

numFreqPoints = 5001
freqInt = 50

if (platform.system() == "Windows"): # Windows 
        dataPath = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/" 
        calPath = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        measPath = "C:/Users/dgarrett/OneDrive - University of Calgary/dp_ml_meas_data/"
        figPath = "C:/Users/dgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
        HDF5Path = "C:/Users/dgarrett/Google Drive/Work/Software/machine_learning/data"
        HDF5Filename = 'dp_ml_data.hdf5'
        print("Windows detected, data path found")
elif (platform.system() == "Darwin"): # Mac
        dataPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/"
        calPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        measPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_meas_data"
        figPath = "/Users/davidgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
        HDF5Path = "/Users/davidgarrett/Google Drive/Work/Software/machine_learning/data"
        HDF5Filename = 'dp_ml_data.hdf5'
        print("Macintosh detected, data path found")
elif (platform.system() == "Linux"): # Linux 
        dataPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cwg_data/"
        calPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_cal_data/"
        measPath = "/Users/davidgarrett/OneDrive - University of Calgary/dp_ml_meas_data"
        figPath = "/Users/davidgarrett/Google Drive/Work/Simulations/Sim4Life/Documentation/Figures"
        HDF5Path = "/Users/davidgarrett/Google Drive/Work/Software/machine_learning/data"
        HDF5Filename = 'dp_ml_data.hdf5'
        print("Linux detected, data path found")
else:
        print("Data cannot be located - package will be used for predicting from existing models.")

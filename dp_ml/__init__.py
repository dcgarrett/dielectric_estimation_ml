#%matplotlib inline

# import necessary libraries
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


# Assign appropriate directories
numFreqPoints = 5001
freqInt = 50


# Import functions from the other files
from .dp_ml_fx import reset_graph
from .dp_ml_fx import get_params_from_filename
from .dp_ml_fx import realimag_to_magphase
from .dp_ml_fx import mag2db
from .dp_ml_fx import importFromDir
from .dp_ml_fx import reshapeData
from .dp_ml_fx import plotEstimate
from .dp_ml_fx import generateMultiMLP
from .dp_ml_fx import predictFromMLP
from .dp_ml_fx import S_to_T
from .dp_ml_fx import T_to_S
from .dp_ml_fx import loadCalibration_WG
from .dp_ml_fx import performCalibration_WG
from .dp_ml_fx import SMatrixToArrays
from .dp_ml_fx import plotFromS





# Some testing
def say_something_init():
	return(u'how dya do, govna?')


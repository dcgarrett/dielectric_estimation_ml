#%matplotlib inline

# import necessary libraries
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
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

# Import functions from the other files
from .dp_ml_fx import *
from .hdf5_fx import *
from .tsar_math import *
from .tensorflow_fx import *

def add(x,y):
	return x+y

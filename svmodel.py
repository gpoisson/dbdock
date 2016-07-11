import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import MDAnalysis as md
import functions
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import operator
import os, sys

data_file_name = "sorted_ligs_by_deltaG.npy"
lig_data = np.load(data_file_name)
lig_count = len(lig_data)


def 
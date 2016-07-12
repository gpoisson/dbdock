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
import matplotlib.pyplot as plt

data_file_name = "sorted_ligs_by_deltaG.npy"
samples_per_step = sys.argv[1]
lig_data = np.load(data_file_name)
lig_count = len(lig_data)

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
	slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
	peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
	feats = [smr_vsa,slogp_vsa,peoe_vsa]
	feats = np.asarray(feats)						# convert to numpy array
	return feats

# Measures the distribution of delta G values in the dataset
def countDistribDeltaG():
	dgs = {}
	for mol in lig_data:
		dgs[mol[2]] = 0		# allot space for all present delta Gs
	for mol in lig_data:
		dgs[mol[2]] += 1	# populate the delta Gs
	x = []
	y = []
	for dg, count in dgs.iteritems():
		x.append(dg)
		y.append(count)
	return x, y
import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import functions
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import operator
import os, sys
import time
import matplotlib.pyplot as plt

start = time.time()

data_file_name = "sorted_no_dg.npy"
sample_layers = int(sys.argv[1])				# train on <sample_layer> number of mols after naive set
lig_data = np.load(data_file_name)
lig_count = len(lig_data)
test_count = 5000
test_ligs = []

print "\n Ligand binary read: {} molecules".format(lig_count)

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
	slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
	peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
	hbd = rdMolDescriptors.CalcNumHBD(mol)
	hba = rdMolDescriptors.CalcNumHBA(mol)

	feats = [smr_vsa,slogp_vsa,peoe_vsa,hbd,hba]

	feature_data = []
	for f in feats:
		if (isinstance(f,int)):
			feature_data.append(f)
		else:
			for data in f:
				feature_data.append(data)
	#feature_data = np.asarray(feature_data)						# convert to numpy array
	return feature_data

def drawNaiveSet(lig_data):
	naive_set = []
	# get distribution of ligands with respect to a one-dimensional feature (h bond donor)
	# choose a small set of ligands which are roughly equidistant in this small feature space
	return naive_set

def main():
	# lig_data = [ ['ZINC1234567', <rdkit_mol>], ['ZINC2345678', <rdkit_mol>], ...]
	lig_feature_data = getAllFeatureData(lig_data)
	# lig_feature_data = [ [13.53001	0	2.139	0...], [23.310	0	13.391...], ...]
	naive_set = drawNaiveSet(lig_feature_data)
	# naive_set = 
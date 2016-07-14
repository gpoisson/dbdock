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
sample_layers = int(sys.argv[1]) + 1			# naive dataset takes <sample_layer> samples per bin
lig_data = np.load(data_file_name)
lig_count = len(lig_data)

print " Ligand binary read: {} molecules".format(lig_count)

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

# Returns a naive sample of the molecules
# samples = [[dg, <rdkit mol>],[dg, <rdkit mol>],...]
def getNaiveDataset():
	data_dict = {}
	for d in lig_data:						# Build a dictionary, with mols being stored under the entry for
		if (d[2] in data_dict):				#    their respective delta G values
			data_dict[d[2]].append(d[1])
		else:
			data_dict[d[2]] = [d[1]]
	samples = []
	for layer in range(sample_layers-1):
		for dg in data_dict:
			sample_count = len(data_dict[dg])							# number of molecules at each delta g value
			rand_index = int(np.random.random() * sample_count - 1)		# choose a random molecule from each bin
			samples.append([dg,data_dict[dg][rand_index]])
	return samples

samples = getNaiveDataset()
print len(samples)
plt.figure()
plt.plot(x,y,'x',mew=3, ms=5)

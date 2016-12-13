import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os, sys
import time
import matplotlib
from scipy.spatial import distance
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

c_val = 1.0
rebuild_db = False
db = "lig_db_full_3D.npy"

if (len(sys.argv) > 2):
	c_val = int(sys.argv[1])
	rebuild_db = sys.argv[2]


def getDG(name):
	energies = open("energies_sorted.dat",'r')
	for line in energies:
		line = line.split()
		if (name in line):
			return float(line[1])
	return None

def shuffleXY(X,y):
    p = np.random.permutation(len(y))
    X_new = []
    y_new = []
    for index in p:
        X_new.append(X[index])
        y_new.append(y[index])
    X = np.asarray(X_new)
    y = np.asarray(y_new)
    return (X, y)

# Gets Euclidian distances for all HBA/HBD pairs, returns a histogram of their values
def getDistBins(mol):
	hbd = Chem.MolFromSmarts('[#7H,#7H2,#7H3,#8H]')
	hba = Chem.MolFromSmarts('[#7X1,#7X2,#7X3,#8,#9,#17]')
	try:
		mol =  Chem.AddHs(mol)
		max_dist = 15							# keep track of hba/hbd distances of up to max_dist angstroms
		AllChem.EmbedMolecule(mol,useRandomCoords=True)
		r = 1
		while (r == 1):		# This optimization returns 1 if it did not converge, so it iterates until it converges
			r = AllChem.UFFOptimizeMolecule(mol)
		atom_coords = []
		mol_dists = np.arange(max_dist)
		bins = np.zeros(max_dist)
		block = Chem.MolToMolBlock(mol)
		block = block.split("\n")
		for line in block:
			line = line.split()
			if (len(line) == 16):
				atom_coords.append([float(line[0]),float(line[1]),float(line[2])])
		atoms = mol.GetAtoms()
		hbds = mol.GetSubstructMatches(hbd)
		hbas = mol.GetSubstructMatches(hba)
		for d, in hbds:				# go through all hbas and all hbds, compute relative distances
			for a, in hbas:
				if a != d:
					sq_dist = np.sum(np.asarray(atom_coords[a])**2 + np.asarray(atom_coords[d])**2)
					dist = np.sqrt(sq_dist)
					bins[int(dist)] += 1
	except:
		return [None]
	
	return bins

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	global index_of_1d_features
	
	dist_bins = getDistBins(mol)

	smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
	slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
	peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
	hbd = rdMolDescriptors.CalcNumHBD(mol)
	hba = rdMolDescriptors.CalcNumHBA(mol)
	
	molwt = rdMolDescriptors.CalcExactMolWt(mol)
	nalrings = rdMolDescriptors.CalcNumAliphaticRings(mol)
	nalhet = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
	narohet = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
	narorings = rdMolDescriptors.CalcNumAromaticRings(mol)
	nHBA = rdMolDescriptors.CalcNumHBA(mol)
	nHBD = rdMolDescriptors.CalcNumHBD(mol)
	rotbounds = rdMolDescriptors.CalcNumRotatableBonds(mol)

	#feats = [smr_vsa,slogp_vsa,peoe_vsa,hbd,hba]
	#feats = [molwt,nalrings,nalhet,narohet,narorings,nHBA,nHBD,rotbounds]
	feats = [smr_vsa,slogp_vsa,peoe_vsa,hbd,hba,molwt,nalrings,nalhet,narohet,narorings,nHBA,nHBD,rotbounds,dist_bins]
	
	feature_data = []
	for f in feats:
		if (isinstance(f,int)):
			feature_data.append(f)
		elif (isinstance(f,float)):
			feature_data.append(f)
		else:
			for data in f:
				feature_data.append(data)
	feature_data = np.asarray(feature_data)						# convert to numpy array
	'''
	print ("smr_vsa:\t{}".format(smr_vsa))
	print ("slogp_vsa:\t{}".format(slogp_vsa))
	print ("peoe_vsa:\t{}".format(peoe_vsa))
	print ("hbd:\t{}".format(hbd))
	print ("hba:\t{}".format(hba))
	print ("molwt:\t{}".format(molwt))
	print ("nalrings:\t{}".format(nalrings))
	print ("nalhet:\t{}".format(nalhet))
	print ("narohet:\t{}".format(narohet))
	print ("narorings:\t{}".format(narorings))
	print ("nHBA:\t{}".format(nHBA))
	print ("nHBD:\t{}".format(nHBD))
	print ("rotbounds:\t{}".format(rotbounds))
	print ("dist bins:\t{}".format(dist_bins))
	print ("")
	print feature_data
	'''
	return feature_data

def readOriginalData(saveNewBin=True):
	print ("Reading original ligand data")
	ligs1 = np.load("sorted_ligs_by_deltaG_part_0.npy")
	ligs2 = np.load("sorted_ligs_by_deltaG_part_1.npy")
	X = []
	y = []

	ligs = []
	print("Extracting feature data")
	for lig in ligs1:
		ligs.append(lig)
		X.append(extractFeatureData(lig[1]))
		y.append(lig[2])

	for lig in ligs2:
		ligs.append(lig)
		X.append(extractFeatureData(lig[1]))
		y.append(lig[2])

	#X, y = shuffleXY(X,y)

	if (saveNewBin):
		np.save(db,(X,y))

	print("Original ligand data loaded and feature data extracted.")
	print("Feature data saved into {}".format(db))

	return X, y

def readBinary():
	ligs = np.load(db)
	data = ligs[0]
	X = []
	for d in data:
		X.append(np.asarray(d))
	y = ligs[1]

	#X, y = shuffleXY(X,y)

	return X, y

# normalize
def normalize(X):
	mean_features = []
	std_features = []
	# compute mean and std of each feature
	for f in range(len(X[0])):
		mean_features.append(np.mean(X[:][f]))
		std_features.append(np.std(X[:][f]))
	# normalize each ligand feature set
		X[:][f] -= mean_features[f]
		X[:][f] /= std_features[f]
	'''
	# normalize each ligand feature set
	for feature in range(len(X)):
		X[:][f] -= mean_features[f]
		X[:][f] /= std_features[f]
	'''
	return X

def main():
	if (rebuild_db):
		X, y = readOriginalData(saveNewBin=True)
		X, y = readBinary()

		X = normalize(X)

		np.save("X.npy",X)
		np.save("y.npy",y)
	
	else:
		X = np.load("X.npy")
		y = np.load("y.npy")

	X, y = shuffleXY(X, y)

	naive_x, test_x, naive_y, test_y = train_test_split(X, y, test_size=0.4, random_state=42)

	print ("Checking train size: {}".format(len(naive_x)))

	model = SVR(kernel='rbf',C=c_val)
	model.fit(naive_x,naive_y)

	train_predictions = model.predict(naive_x)
	predictions = model.predict(test_x)
	error = abs(test_y - predictions)

	print ("Mean Error: {}".format(np.mean(error)))
	print ("Max Error:  {}".format(np.max(error)))

	plt.figure()
	plt.suptitle("Prediction vs Actual Delta G\nTest Set")
	plt.xlabel("Predicted Delta G")
	plt.ylabel("Actual Delta G")
	plt.plot(predictions,test_y,'x',ms=2,mew=3)
	plt.grid(True)
	plt.show()

	plt.savefig("check_results/oos_{}k_c{}.png".format(len(train_predictions),c_val,))

	plt.figure()
	plt.suptitle("Prediction vs Actual Delta G\nTraining Set")
	plt.xlabel("Predicted Delta G")
	plt.ylabel("Actual Delta G")
	plt.plot(train_predictions,naive_y,'x',ms=2,mew=3)
	plt.grid(True)
	plt.show()

	plt.savefig("check_results/is_{}k_c{}.png".format(len(train_predictions),c_val,))


main()
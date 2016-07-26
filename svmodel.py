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

data_file_name = "dev_data.npy"
# data_file_name = "sorted_no_dg.npy"
sample_layers = int(sys.argv[1])				# train on <sample_layer> number of mols after naive set
threshold = int(sys.argv[2])					# minimum value accepted for max_error of prediction
lig_data = np.load(data_file_name)
lig_count = len(lig_data)
lig_feature_data = []
test_count = 5000
test_ligs = []
index_of_1d_feature = None
print "\n Ligand binary read: {} molecules".format(lig_count)

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	global index_of_1d_feature
	smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
	slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
	peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
	hbd = rdMolDescriptors.CalcNumHBD(mol)
	hba = rdMolDescriptors.CalcNumHBA(mol)

	index_of_1d_feature = -1		# Need to make sure this references the index of a 1D feature
									#  (a negative index refers to counting backwards from the end of a list)
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

def getAllFeatureData(lig_data):
	lfd = []
	print " Reading feature data"
	prg = ""
	for c in range(len(lig_data)):
		if (c % 5000 == 0):
			prg += '='
	prg += " Progress:"
	print prg
	for mol in lig_data:
		if (len(lfd) % 5000 == 0):
			sys.stdout.write('.')
			sys.stdout.flush()
		lfd.append([mol[0],extractFeatureData(mol[1])])
	return lfd

def drawNaiveSet(lig_feature_data):
	naive_set = []			# the list of ligands which are ultimately chosen for the naive set
	deltaGs = []			# the delta G values for the initially chosen ligands
	small_data_dict = {}	# key: feature value   	value: list of mols

	for mol in lig_feature_data:
		d = int(mol[1][index_of_1d_feature])
		if d in small_data_dict:
			small_data_dict[d].append(mol)
		else:
			small_data_dict[d] = [mol]

	print " {} unique values for this 1D feature".format(len(small_data_dict))

	for dg in small_data_dict:
		rand_index = int(np.random.random() * len(small_data_dict[dg]))
		naive_set.append(small_data_dict[dg][rand_index])
		deltaGs.append(getDeltaG(small_data_dict[dg][rand_index]))

	print " Obtained naive dataset containing {} ligands.".format(len(naive_set))

	removeSampledLigs(naive_set)
	print " {} unsampled ligands remaining".format(len(lig_feature_data))

	# get distribution of ligands with respect to a one-dimensional feature (h bond donor)
	# choose a small set of ligands which are roughly equidistant in this small feature space
	return naive_set, deltaGs

# Combines the newest ligand with the last set
def drawNextSet(last_set, new_lig, deltaGs):
	next_set = last_set
	next_set.append(new_lig)
	deltaGs.append(getDeltaG(new_lig))
	return next_set, deltaGs

# Removes a list of ligands from the database of ligand feature data
def removeSampledLigs(lig_set):
	global lig_feature_data
	sampled_names = lig_set
	remaining_names = lig_feature_data

	for s in sampled_names:
		if s in remaining_names:
			rem_index = remaining_names.index(s)
			del lig_feature_data[rem_index]

# Simulates running autodock by looking up delta G in a table and waiting for some time
def getDeltaG(mol):
	ens = open("energies_sorted.dat",'r')
	for line in ens:
		line = line.split()
		if line[0] == mol[0]:
			print ("Computing delta G for {}...".format(line[0]))
			#time.sleep(1)
			print ("  Result = {}".format(line[1]))
			return float(line[1])
	print "Failed to find a delta G for {}".format(mol)
	return 0

# Return a model fitted to all the current known data
def fitSet(ligand_set, deltaGs):
	model = SVR(kernel='rbf')
	x = []
	y = deltaGs

	for lig in ligand_set:
		x.append(lig[1])	# compile feature data

	x = np.asarray(x)
	sample_count = len(x)
	print "Fitting model, {} data points...".format(sample_count)
	model.fit(x,y)
	print "Model successfully fitted."
	return model

# Identify the predicted delta G which is most unlike the other known values
def getMostUniquePrediction(predictions, deltaGs, lig_feature_data):
	max_diff = 0
	index = 0
	predictions = predictions.tolist()
	for val in deltaGs:
		for pred in predictions:
			if (abs(val - pred) > max_diff):
				index = predictions.index(pred)
	return lig_feature_data[index]

# Use model to choose the next ligand to be tested
def getNextLigand(predictions, lig_feature_data, deltaGs):
	print " Determining next sample to include in model..."
	next_lig = getMostUniquePrediction(predictions, deltaGs, lig_feature_data)
	removeSampledLigs([next_lig[0]])
	print "  Returned {}".format(next_lig[0])
	return next_lig



# Picks next sample, updates current model, measures error
def updateModel(model, lig_feature_data, this_set, deltaGs):
	test_data = []
	for lig in lig_feature_data:
		test_data.append(lig[1])
	test_data = np.asarray(test_data)
	predictions = model.predict(test_data)
	new_lig = getNextLigand(predictions, lig_feature_data, deltaGs)
	next_set, deltaGs = drawNextSet(this_set, new_lig, deltaGs)
	model = fitSet(next_set, deltaGs)
	removeSampledLigs(new_lig)
	return next_set, deltaGs

def main():
	global lig_feature_data
	lig_feature_data = getAllFeatureData(lig_data)
	naive_set, deltaGs = drawNaiveSet(lig_feature_data)
	model = fitSet(naive_set, deltaGs)
	max_error = 100
	next_set = naive_set
	while (max_error > threshold):
		this_set = next_set
		model = fitSet(this_set, deltaGs)
		next_set, deltaGs = updateModel(model, lig_feature_data, this_set, deltaGs)
		max_error = testModel(model, deltaGs)

main()
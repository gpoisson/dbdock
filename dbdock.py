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

#################
## GLOBAL VARS ##
#################
coords_dir = "lig_mols/"
energies_dir = "energies/"
energies_file = "energies_sorted.dat"
dataSize = 100000
additions_per_fitting = 50			# Number of new points to add to the model at each fitting
rbf_c = 1.0
v = False
n = False
if (len(sys.argv) > 1):
	v = sys.argv[1]
if (len(sys.argv) > 2):
	dataSize = sys.argv[2]
if (len(sys.argv) > 3):
	n = sys.argv[3]

# Find, read, and compile input data
# Due to file size limits on GitHub, I have used split binaries which are
#   recombined before use.
def readInputData():
	if n:							# Allows user to force building a new binary from raw data. Will likely overwrite existing binary.
		functions.makeUnsortedBinary()
	try:							# Load a saved binary containing our data
		binary_outfile = "unsorted_ligs.npy"
		unsortedData = loadUnsortedBin(binary_outfile)
		return unsortedData
	except Exception:				# If a binary isn't found, look for partial binaries and combine them.
		try:
			if v:
				print ("Failed to find binary. Looking for partial binaries...")
			functions.combineSplitBinaries()
			readInputData()
		except Exception:			# If no binaries of any type are found, assemble a new one.
			if v:
				print ("Failed to find partial binaries. Looking for raw data...")
			functions.makeUnsortedBinary()
			readInputData()


# Reads in energy valus for each ligand
def readInputEnergies():
	deltaGs = {}
	engs = open(energies_file,'r')
	for line in engs:
		line = line.split()
		if (len(line) >= 2):
			name = line[0]
			if (name[-1] == '.'):		# trim off trailing periods
				name = name[:-1]
			dg = float(line[1])
			try:
				deltaGs[name] = dg
			except KeyError:
				continue
	return deltaGs

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
	slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
	peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
	feats = [smr_vsa,slogp_vsa,peoe_vsa]
	feats = np.asarray(feats)						# convert to numpy array
	return feats

# Take features from the training set
def makeXValues(data,count=dataSize,f="train"):
	x = [[]]
	for mol in data:
		for feature in mol[1]:
			for val in feature:
				x[-1].append(val)
		x.append([])
	x = x[:-1]			# remove final empty entry
	x = preprocessing.scale(np.asarray(x[:int(count)]))
	return x

# Take energies from training set
def makeYValues(data,count=dataSize,f="train"):
	y = []
	for mol in data:
		y.append(mol[2])
	y = preprocessing.scale(y[:int(count)])
	return y

# Sort input data into training and testing datasets
def sortInputNew(mols,deltaGs):
	train = []
	test = []
	trainIndeces = []		# Keep track of molecules used for training; use the rest for testing
	c = 0
	p = 0
	count = 0

	if v:
		print "Assemble training and testing data..."

	sorted_mols = sorted(mols.items(), key=operator.itemgetter(1))

	# Build dict, sort by deltaG
	# Starting from smallest deltaG, working towards largest, extract features and make a data point
	# Store the data points in a long np array and save the binary
	# Sort the data points into training and testing
	

# Sort input data into training and testing datasets
# Picks n evenly-indexed mols for training, rest go to testing
def sortInputData(mols,deltaGs):
	train = []
	test = []
	trainIndeces = []		# Keep track of molecules used for training; use the rest for testing
	c = 0
	p = 0
	count = 0

	if v:
		print "Assemble training and testing data..."
	while (count < int(dataSize)):
		if (count == 0):									# If no molecules have been sorted, take the first one
			try:	
				if (mols[0][1] == None):
					continue
				else:
					ligand_name = mols[0][0]
					features = extractFeatureData(mols[0][1])
					dg = deltaGs[ligand_name]
					train.append([ligand_name,features,dg])
					trainIndeces.append(0)
			except KeyError:
				continue
		else:												# If more than two molecules have been sorted, evenly distribute the rest
			c += 1
			index = int(len(mols)/2**p) + int(len(mols)*c/2**(p-1)) - 1				# Apply a logarithmic distribution to evenly choose training/testing samples
			if (index >= len(mols)):
				p += 1
				c = 0
				index = int(len(mols)/2**p) + int(len(mols)*c/2**(p-1)) - 1
			try:	
				if (mols[index][1] == None):
					continue
				else:
					ligand_name = mols[index][0]
					features = extractFeatureData(mols[index][1])
					dg = deltaGs[ligand_name]
					train.append([ligand_name,features,dg])
					trainIndeces.append(index)
			except KeyError:
				continue
		count = len(train)

	count = 0
	for mol in mols:										# Use record of training molecule indeces to add non-training molecules to test
		if count in trainIndeces:
			continue
		else:
			try:
				if (mol[1] == None):
					continue
				else:
					ligand_name = mols[count][0]
					features = extractFeatureData(mols[count][1])
					dg = deltaGs[ligand_name]
					test.append([ligand_name,features,dg])
			except KeyError:
				continue
		count += 1
	if v:
		print "Sorted {} molecules into {} training points and {} test points.".format(len(mols),len(train),len(test))
	return train, test


# Sort input data into training and testing datasets
# Places every nth mol into train, rest into test
def sortInputStep(mols,deltaGs):
	train = []
	test = []
	count = 0
	test_frac = int(len(mols)/int(dataSize))

	if v:
		print "Assemble training and testing data..."
	for mol in mols:
		if (mol[1] == None):
			continue
		else:
			ligand = mol[0]									# ligand name (string)
			features = extractFeatureData(mol[1])			# mol[1] = mol object (rdkit)
			try:
				dg = deltaGs[ligand]							# deltaG for this ligand
				if (count % test_frac != 0):
					test.append([ligand,features,dg])
				else:
					train.append([ligand,features,dg])
				count += 1
			except KeyError:
				continue
	return train, test

# Fit the training data to a model
def fitModel(trainingData):
	if v:
		print "Fit model..."
	model = SVR(C=rbf_c,kernel='rbf',gamma='auto')
	x = makeXValues(trainingData)
	y = makeYValues(trainingData)
	if v:
		print " Considering {} training samples with {} total features each.".format(len(x),len(x[0]))
	model.fit(x,y)
	if v:
		print "Model fitted."
	return model

# Test our model on some known values
def testModel(model,testData):
	if v:
		print "Test model..."
	inData = makeXValues(testData,f="test")
	expected = makeYValues(testData,f="test")
	predicted = model.predict(inData)

	predicted_norm = predicted / np.linalg.norm(predicted)
	expected_norm = expected / np.linalg.norm(expected)
	predicted_scale = preprocessing.scale(predicted)
	expected_scale = preprocessing.scale(expected)

	#print predicted_norm
	functions.makeSinglePlot(expected,predicted,title="Expected vs Predicted Delta G (scaled) - {} training points, {} test points".format(dataSize,len(expected)),x_label="Expected",y_label="Predicted",axis_equal=True)
	return predicted

def saveModel(model):
	model_filename = "svr_model/svr_model_trained_{}.pkl".format(dataSize)
	joblib.dump(model,model_filename)

# Save the compiled data as a binary file for faster loading
def saveAsBin(train,test):
	binary_outfile = "sorted_ligs_{}.npz".format(dataSize)
	np.savez(binary_outfile,train,test)

# Load binary data file
def loadBin(binary_outfile):
	if v:
		print ("Loading binary file: {}".format(binary_outfile))
	arrs = np.load(binary_outfile)
	trainingData = arrs['arr_0.npy']
	if v:
		print (" Training data loaded.\n\tSize: {} molecules".format(len(trainingData)))
	testData = arrs['arr_1.npy']	
	if v:
		print (" Test data loaded.\n\tSize: {} molecules".format(len(testData)))
	return trainingData, testData

def loadUnsortedBin(binary_outfile):
	if v: 
		print ("Loading binary file: {}".format(binary_outfile))
	arr = np.load(binary_outfile)
	if v:
		print ("Training data loaded.\n\tSize: {} unsorted molecules".format(len(arr)))
	return arr

# Determines the indeces of the next molecules to be included in the model
def getNextIndeces(model,trainingIndeces,totalMolCount):
	print

# Fit a small model using evenly distributed data points.
# Then use the model to determine another small data set which will 
def fitModelContinuous(allMolData):
	initialModel = SVR()
	totalMolCount = len(allMolData)
	trainingIndeces = []

	# Train a model on a small, evenly distributed dataset
	initialIndeces = getNextIndeces(initialModel,trainingIndeces,totalMolCount)
	# Recursively use the model to determine the next dataset to train on

	# Recursively train a new model on the entire current training dataset

	return model, unusedData


#########################################
################  MAIN  #################
#########################################
def main():
	allMolData = readInputData()
	model, unusedData = fitModelContinuous(allMolData)
	saveModel(model)
	testModel(model,testData)

main()

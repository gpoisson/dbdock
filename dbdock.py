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
		binary_outfile = "sorted_ligs_by_deltaG.npy"
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

# Keeps track of which molecules have been trained on already
# Maintains a running list of the indeces of used data
def getNextIndeces(model,trainingIndeces,totalMolCount):
	if (len(trainingIndeces) == 0):
		ratio = totalMolCount / additions_per_fitting
		index_list = np.arange(additions_per_fitting) * ratio
		for index in index_list:
			trainingIndeces.append(index)

# Returns a set of data on which to train the newest model
def getTrainData(trainingIndeces,allMolData):
	tData = [[]]
	for i in trainingIndeces:
		tDataTemp = extractFeatureData(allMolData[i][1])
		for feature in tDataTemp:
			for data in feature:
				tData[-1].append(data)
		tData.append([])
	return np.asarray(tData[:-1])

# Returns a set of known delta G values to train the newest model
def getTargetData(trainingIndeces,allMolData):
	tData = []
	for i in trainingIndeces:
		tData.append(allMolData[i][2])
	return tData

# Fit a small model using evenly distributed data points.
# Then use the model to determine another small data set which will 
#   be used for further training
def fitModelContinuousTest(allMolData):
	initialModel = SVR(kernel='rbf')
	model = SVR(kernel='rbf')
	totalMolCount = len(allMolData)
	trainingIndeces = []

	# Train a model on a small, evenly distributed dataset
	getNextIndeces(initialModel,trainingIndeces,totalMolCount)
	trainData = getTrainData(trainingIndeces,allMolData)
	targetData = getTargetData(trainingIndeces,allMolData)
	initialModel.fit(trainData,targetData)
	model = initialModel
	
	# Use the model to determine the next dataset to train on

	# Train a new model on the entire current training dataset

	return model, trainingIndeces


#########################################
################  MAIN  #################
#########################################
def main():
	allMolData = readInputData()
	model, trainingIndeces = fitModelContinuousTest(allMolData)
	#saveModel(model)
	testModel(model,testData)

main()

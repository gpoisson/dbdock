import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import MDAnalysis as md
import functions
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import os, sys

#################
## GLOBAL VARS ##
#################
coords_dir = "lig_mols/"
energies_dir = "energies/"
energies_file = "energies_sorted.dat"
dataSize = 100000
rbf_c = 1.0
v = False
if (len(sys.argv) > 1):
	v = sys.argv[1]
if (len(sys.argv) > 2):
	dataSize = sys.argv[2]

# Find, read, and compile input data
def readInputData():
	if v:
		print "Read input data..."
	try:							# Try to load a saved binary containing our data
		trainingData, testData = loadBin(binary_outfile)
	except Exception:				# If the binary does not exist, compile data from scratch and save binary
		allValidMols = []
		ligand_list = os.listdir(coords_dir)
		fails = 0
		for ligand_file in ligand_list:
			ligand_name = ligand_file[:-4]		# ligand name is ligand file name without ".mol" extention
			try:
				mol = Chem.MolFromMolFile("{}{}".format(coords_dir,ligand_file))
				allValidMols.append([ligand_name,mol])
			except IOError:
				fails += 1
				continue
		if v:
			print " Read in all {} molecules, encountered {} failures.".format(len(ligand_list),fails)
		deltaGs = readInputEnergies()
		trainingData, testData = sortInputData(allValidMols,deltaGs)
		if v:
			print " Read in {} training data, {} test data.".format(len(trainingData),len(testData))
		saveAsBin(trainingData, testData)
	return [trainingData, testData]

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
	#if (len(x) < count):
		#return (np.asarray(x[:-1]))
	x = x[:-1]			# remove final empty entry
	x = preprocessing.scale(np.asarray(x[:int(count)]))
	return x

# Take energies from training set
def makeYValues(data,count=dataSize,f="train"):
	y = []
	for mol in data:
		y.append(mol[2])
	#if (len(y) < count):
		#return y
	y = preprocessing.scale(y[:int(count)])
	return y

# Sort input data into training and testing datasets
def sortInputData(mols,deltaGs):
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
	binary_outfile = "inputData_{}.npz".format(dataSize)
	np.savez(binary_outfile,train,test)

# Load binary data file
def loadBin(binary_outfile):
	if v:
		print ("Loading binary file: {}".format(binary_outfile))
	binary_outfile = "inputData_{}.npz".format(dataSize)
	arrs = np.load(binary_outfile)
	trainingData = arrs['arr_0.npy']
	if v:
		print (" Training data loaded.\n\tSize: {} molecules".format(len(trainingData)))
	testData = arrs['arr_1.npy']	
	if v:
		print (" Test data loaded.\n\tSize: {} molecules".format(len(testData)))
	return trainingData, testData

#########################################
################  MAIN  #################
#########################################
def main():
	trainingData, testData = readInputData()
	model = fitModel(trainingData)
	saveModel(model)
	testModel(model,testData)

main()

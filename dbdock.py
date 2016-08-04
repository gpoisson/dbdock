import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import os, sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start = time.time()

data_file_name = "dev_data.npy"
sample_layers = int(sys.argv[1])				# train on <sample_layer> number of mols after naive set
#c_val = None
c_val = float(sys.argv[2])						# C Parameter for SVR
lig_data = np.load(data_file_name)
lig_count = len(lig_data)
index_of_1d_features = [-1,-2]
training_set = []
total_dg_hist = {}

print "\n Ligand binary read: {} molecules".format(lig_count)

class ligand():
	name = None
	mol = None
	feature_data = None
	known_dg = None
	predicted_dg = None
	total_training_sample_count_at_last_prediction = 0
	feature_distance = None

def randomizeLigData():
	global lig_data
	print "Suffling ligand data"
	lig_data = np.asarray(lig_data)
	np.random.shuffle(lig_data)
	lig_data = np.array(lig_data.tolist())
	print "...done."

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	global index_of_1d_features
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
	feature_data = np.asarray(feature_data)						# convert to numpy array
	return feature_data

def extractAllFeatureData():
	global lig_data
	print " Reading feature data"
	lig_db = []
	prg = ""
	for c in range(len(lig_data)):
		if (c % 5000 == 0):
			prg += '='
	prg += " Progress:"
	print prg
	for lig in lig_data:
		drug = ligand()
		drug.name = lig[0]
		drug.mol = lig[1]
		drug.feature_data = extractFeatureData(drug.mol)
		f_dist = 0
		for i in index_of_1d_features:
			if index_of_1d_features.index(i) == 0:
				f_dist += drug.feature_data[i] ** 2
			else:
				f_dist -= drug.feature_data[i] ** 2
		drug.feature_distance = (f_dist)
		lig_db.append(drug)
		if (len(lig_db) % 5000 == 0):
			sys.stdout.write('.')
			sys.stdout.flush()
	print ".\n Feature Data Extracted."
	lig_data = lig_db

def moveLigandToTrainingSet(ligand):
	global lig_data, training_set
	ligand.known_dg = getDeltaG(ligand.name)
	training_set.append(ligand)
	del lig_data[lig_data.index(ligand)]
	print "Ligand {} moved to training set.".format(training_set[-1].name)
	print "\tFeature distance: {}".format(ligand.feature_distance)

# Simulates running autodock by looking up delta G in a table and waiting for some time
def getDeltaG(mol):
	ens = open("energies_sorted.dat",'r')
	for line in ens:
		line = line.split()
		if line[0] == mol:
			print ("Computing delta G for {}...".format(line[0]))
			#time.sleep(2)
			print ("  Result = {}".format(line[1]))
			return float(line[1])
	print "Failed to find a delta G for {}".format(mol)
	return 0

# Initial Set Selection Method A:
#	Choose a fraction of ligands which are evenly distributed through the dataset.
def chooseInitialSet_A(perc=0.01):
	lig_count = len(lig_data)
	spacing = int(lig_count/(lig_count * perc))
	c = 0
	for ligand in lig_data:
		if c >= spacing:
			moveLigandToTrainingSet(ligand)
			c = 0
		c += 1

# Returns feature and target data for model training
def getTrainingData():
	tfd = []							# training feature data
	dgs = []							# delta Gs
	for drug in training_set:
		tfd.append(drug.feature_data)
		dgs.append(drug.known_dg)
	tfd = np.asarray(tfd)
	return tfd, dgs

def makeNewModel():
	print " Compiling current training data..."
	x, y = getTrainingData()
	model = SVR(kernel='rbf',C=c_val)
	model.fit(x,y)
	print " ...Model successfully fitted. {} samples".format(len(y))
	return model

def testModelOnTrainingSet(model):
	print " Testing model on training data."
	predictions = []
	errors = []
	for drug in training_set:
		drug.predicted_dg = round(model.predict([drug.feature_data]),1)
		drug.total_training_sample_count_at_last_prediction = len(training_set)
		predictions.append(drug.predicted_dg)
		errors.append(abs(drug.predicted_dg - drug.known_dg))
	max_err = np.amax(errors)
	mean_err = np.mean(errors)
	print "Model tested against training set.\n\tTraining sample count: {}\n\tMax error: {}\tMean error: {}".format(len(training_set),max_err,mean_err)
	return errors, max_err, mean_err

def testModelOnUnknownSet(model):
	print " Testing model on unknown data"
	predictions = []
	errors = []
	for drug in lig_data:
		drug.predicted_dg = round(model.predict([drug.feature_data]),1)
		drug.total_training_sample_count_at_last_prediction = len(training_set)
		predictions.append(drug.predicted_dg)
		if drug.known_dg == None:
			drug.known_dg = getDeltaG(drug.name)
		errors.append(abs(drug.predicted_dg - drug.known_dg))
	max_err = np.amax(errors)
	mean_err = np.mean(errors)
	print "Model tested against unknown set."
	return errors, max_err, mean_err

def getKnownDeltaGHistogram():
	dg_histogram = {}
	for drug in training_set:
		if drug.known_dg in dg_histogram:
			dg_histogram[drug.known_dg] += 1
		else:
			dg_histogram[drug.known_dg] = 1
	return dg_histogram

def getFeatureDistanceHistogram():
	f_hist = {}
	for drug in lig_data:
		if drug.feature_distance in f_hist:
			f_hist[drug.feature_distance] += 1
		else:
			f_hist[drug.feature_distance] = 1
	return f_hist

# base uniqueness on error of dg prediction (original algorithm)
def getMostUniqueDrug_0(dg_histogram, f_histogram, model):
	max_error = 0
	drug_w_max_error = None
	for drug in lig_data:
		if drug.known_dg == None:
			drug.known_dg = getDeltaG(drug.name)
		drug.predicted_dg = model.predict([drug.feature_data])
		drug.total_training_sample_count_at_last_prediction += 1
		error = abs(drug.known_dg - drug.predicted_dg)
		if (error > max_error):
			max_error = error
			drug_w_max_error = drug
	return drug

# base uniqueness on error of dg prediction (original algorithm)
def getMostUniqueDrug_1(dg_histogram, f_histogram, model):
	min_error = 12.0
	drug_w_min_error = None
	for drug in lig_data:
		if drug.known_dg == None:
			drug.known_dg = getDeltaG(drug.name)
		drug.predicted_dg = model.predict([drug.feature_data])
		drug.total_training_sample_count_at_last_prediction += 1
		error = abs(drug.known_dg - drug.predicted_dg)
		if (error < min_error):
			min_error = error
			drug_w_min_error = drug
	return drug
	

# base uniqueness on max lowest pairwise distance in delta G spectrum between prediction and known
def getMostUniqueDrug_A(dg_histogram, f_histogram, model):
	most_unique_drug = None
	max_difference = 0
	for drug in lig_data:
		drug.predicted_dg = round(model.predict([drug.feature_data]),1)
		drug.total_training_sample_count_at_last_prediction = len(training_set)
		for observed in dg_histogram:
			if (observed - drug.predicted_dg) > max_difference:
				max_difference = abs(observed - drug.predicted_dg)
				most_unique_drug = drug
	return most_unique_drug

# base uniquness on square of difference between specified 1d features
def getMostUniqueDrug_B(dg_histogram, f_histogram, model):
	most_unique_drug = None
	f_lowest_occurances = 99999
	print f_histogram
	for val in f_histogram.iterkeys():
		occurances = int(f_histogram[val])
		if occurances < f_lowest_occurances:
			f_lowest_occurances = occurances
	for drug in lig_data:
		if drug.feature_distance == f_lowest_occurances:
			return drug
	return None

def fitNextLigand(model):
	dg_histogram = getKnownDeltaGHistogram()
	f_histogram = getFeatureDistanceHistogram()
	most_unique_drug = getMostUniqueDrug_1(dg_histogram, f_histogram, model)
	moveLigandToTrainingSet(most_unique_drug)
	new_model = makeNewModel()
	return new_model

def plotResults(errors, max_errs, mean_errs):
	f, (ax_ar0,ax_ar1) = plt.subplots(2, 1)
	#f, ((ax_ar0,ax_ar1,ax_ar2),(ax_ar3,ax_ar4,ax_ar5)) = plt.subplots(2, 3)
	f.subplots_adjust(hspace=0.52)
	ax_ar0.plot(range(len(max_errs)),max_errs,'x',ms=1,mew=3)
	ax_ar0.grid(True)
	ax_ar0.plot(range(len(max_errs)),np.poly1d(np.polyfit(range(len(max_errs)), max_errs, 1))(range(len(max_errs))))
	ax_ar0.set_title("Max Model Error Over Time (kcal/mol)")
	ax_ar0.set_xlabel("Number of Training Iterations")
	ax_ar0.set_ylabel("Max Model Error")

	ax_ar1.plot(range(len(mean_errs)),mean_errs,'x',color='r',ms=1,mew=3)
	ax_ar1.grid(True)
	ax_ar1.set_title("Mean Model Error Over Time (kcal/mol)")
	ax_ar1.set_xlabel("Number of Training Iterations")
	ax_ar1.set_ylabel("Mean Model Error")
	'''
	ax_ar2.plot(range(len(predictions)),predictions,'x',color='k',ms=1,mew=3)
	ax_ar2.grid(True)
	ax_ar2.set_title("Predictions over Time (kcal/mol)")
	ax_ar2.set_xlabel("Number of Training Iterations")
	ax_ar2.set_ylabel("Predicted Delta G (kcal/mol)")

	ax_ar3.plot(range(len(durations)),durations,'x',color='g',ms=1,mew=3)
	ax_ar3.grid(True)
	ax_ar3.set_title("Duration of New Ligand Selection (seconds)")
	ax_ar3.set_xlabel("Number of Training Iterations")
	ax_ar3.set_ylabel("Time (seconds")

	ax_ar4.plot(range(len(mean_error)),mean_error,'x',color='r',ms=1,mew=3)
	ax_ar4.grid(True)
	ax_ar4.set_title("Mean Error During Sampling")
	ax_ar4.set_xlabel("Number of Training Iterations")
	ax_ar4.set_ylabel("Mean Error (kcal/mol)")

	ax_ar5.plot(training_data_known,training_data_predictions,'x',color='k',ms=1,mew=3)
	ax_ar5.plot(training_data_known,np.poly1d(np.polyfit(training_data_known, training_data_predictions, 1))(training_data_known))
	ax_ar5.grid(True)
	ax_ar5.set_title("Training Data Predicted vs Actual Delta G (kcal/mol)")
	ax_ar5.set_xlabel("Known Delta G (kcal/mol)")
	ax_ar5.set_ylabel("Predicted Delta G (kcal/mol)")
	'''
	#plt.show()
	#pp = PdfPages("plots/{}_samples.pdf".format(len(predictions))
	plt.savefig("expData/c_{}_{}_samps.png".format(c_val,sample_layers))
	#pp.close()

def main():
	lig_count = len(lig_data)
	randomizeLigData()
	extractAllFeatureData()
	chooseInitialSet_A(0.005)
	model = makeNewModel()
	max_errs = []
	mean_errs = []
	prediction_errs = []
	e, max_err, mean_err = testModelOnUnknownSet(model)
	prediction_errs.append(e)
	max_errs.append(max_err)
	mean_errs.append(mean_err)
	for layer in range(sample_layers):
		model = fitNextLigand(model)
		e, max_err, mean_err = testModelOnUnknownSet(model)
		prediction_errs.append(e)
		max_errs.append(max_err)
		mean_errs.append(mean_err)
	plotResults(prediction_errs, max_errs, mean_errs)
	prediction_errs = np.asarray(prediction_errs)
	np.save("expData/prediction_errs_c_{}_samps_{}.npy".format(c_val,sample_layers),prediction_errs)
	max_errs = np.asarray(max_errs)
	np.save("expData/max_err_c_{}_samps_{}.npy".format(c_val,sample_layers),max_errs)
	mean_errs = np.asarray(mean_errs)
	np.save("expData/mean_err_c_{}_samps_{}.npy".format(c_val,sample_layers),mean_errs)
	return prediction_errs, max_errs, mean_errs

main()

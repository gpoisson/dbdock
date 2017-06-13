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
import matplotlib.pylab
from matplotlib.backends.backend_pdf import PdfPages

start = time.time()

data_file_name = "dev_data.npy"
# data_file_name = "sorted_no_dg.npy"
sample_layers = int(sys.argv[1])				# train on <sample_layer> number of mols after naive set
threshold = float(sys.argv[2])					# minimum value accepted for max_error of prediction
lig_data = np.load(data_file_name)
lig_count = len(lig_data)
lig_feature_data = []
test_count = 5000
test_ligs = []
index_of_1d_feature = None
training_data_known = []
training_data_predictions = []
dgs = []
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
	global training_data_known
	global lig_feature_data
	#print " lig feature data: {} things".format(len(lig_feature_data))
	#print " lig feature [0]: {} things".format(len(lig_feature_data[0]))
	#print " lig feature [0][0]: {} things".format(len(lig_feature_data[0][0]))
	sampled_names = []
	for lig in lig_set:
		sampled_names.append(lig[0])
	remaining_names = []
	for lig in lig_feature_data:
		remaining_names.append(lig[0])

	for s in sampled_names:
		if s in remaining_names:
			rem_index = remaining_names.index(s)
			print " Removing sampled ligand from database: {}".format(lig_feature_data[rem_index][0])
			training_data_known.append(lig_feature_data[rem_index])
			del lig_feature_data[rem_index]
		else:
			print "Didn't find {} in the remaining {} names. (Example: {})".format(s, len(remaining_names), remaining_names[0])

# Simulates running autodock by looking up delta G in a table and waiting for some time
def getDeltaG(mol):
	ens = open("energies_sorted.dat",'r')
	for line in ens:
		line = line.split()
		if line[0] == mol[0]:
			print ("Computing delta G for {}...".format(line[0]))
			#time.sleep(2)
			print ("  Result = {}".format(line[1]))
			return float(line[1])
	print "Failed to find a delta G for {}".format(mol)
	return 0

# Return a model fitted to all the current known data
def fitSet(ligand_set, deltaGs):
	model = SVR(kernel='rbf',C=1e6)
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
	c_pred = 0									# predicted value for chosen ligand
	for val in deltaGs:
		for pred in predictions:
			if (abs(val - pred) > max_diff):
				index = predictions.index(pred)
				c_pred = pred
				max_diff = abs(val-pred)

	print " There are {} known delta Gs and {} predictions to compare.".format(len(deltaGs),len(predictions))
	return lig_feature_data[index], c_pred

# Return a random ligand to verify usefulness of the algorithm
def getRandomPrediction(predictions, deltaGs, lig_feature_data):
	rand_index = int(np.random.random() * len(lig_feature_data))
	return lig_feature_data[rand_index], predictions[rand_index]

# Use model to choose the next ligand to be tested
def getNextLigand(predictions, lig_feature_data, deltaGs):
	print " Determining next sample to include in model..."
	start = time.time()
	next_lig, pred = getMostUniquePrediction(predictions, deltaGs, lig_feature_data)
	end = time.time()
	dur = end-start
	print " Finding next ligand took {} seconds".format(dur)
	removeSampledLigs([next_lig])
	print "  Returned {}\tPredicted {}".format(next_lig[0], pred)
	return next_lig, pred, dur

def getTrainingDataStats(model):
	global training_data_known,training_data_predictions,dgs
	known_dgs = dgs
	test_data = []
	print "Measuring model quality"
	for lig in training_data_known:
		test_data.append(lig[1])
	training_data_predictions = model.predict(test_data)
	return known_dgs, training_data_predictions
	

# Picks next sample, updates current model, measures error
def updateModel(model, this_set, deltaGs, meas, durations):
	global lig_feature_data, dgs
	test_data = []
	for lig in lig_feature_data:
		test_data.append(lig[1])
	test_data = np.asarray(test_data)
	predictions = model.predict(test_data)
	new_lig, pred, dur = getNextLigand(predictions, lig_feature_data, deltaGs)
	durations.append(dur)
	next_set, deltaGs = drawNextSet(this_set, new_lig, deltaGs)
	dgs = deltaGs
	meas.append(deltaGs[-1])
	model = fitSet(next_set, deltaGs)
	#removeSampledLigs([new_lig])
	error = abs(pred - deltaGs[-1])
	print " Error: {}".format(error)
	return model, next_set, deltaGs, error, meas, pred, durations

def makePlots(errors, predictions, deltaGs, durations, mean_error, training_data_known, training_data_predictions):
	f, ((ax_ar0,ax_ar1),(ax_ar2,ax_ar3),(ax_ar4,ax_ar5)) = plt.subplots(3, 2)
	f.subplots_adjust(hspace=0.52)
	ax_ar0.plot(range(len(errors)),errors,'x',ms=3,mew=5)
	ax_ar0.grid(True)
	ax_ar0.plot(range(len(errors)),np.poly1d(np.polyfit(range(len(errors)), errors, 1))(range(len(errors))))
	ax_ar0.set_title("Model Error Over Time (kcal/mol)")
	ax_ar0.set_xlabel("Number of Training Iterations")
	ax_ar0.set_ylabel("Model Error")

	ax_ar1.plot(predictions,deltaGs,'x',color='r',ms=3,mew=5)
	ax_ar1.grid(True)
	ax_ar1.set_title("Predicted vs Actual Delta G (kcal/mol)")
	ax_ar1.set_xlabel("Predicted Delta G (kcal/mol)")
	ax_ar1.set_ylabel("Actual Delta G (kcal/mol)")

	ax_ar2.plot(range(len(predictions)),predictions,'x',color='k',ms=3,mew=5)
	ax_ar2.grid(True)
	ax_ar2.set_title("Predictions over Time (kcal/mol)")
	ax_ar2.set_xlabel("Number of Training Iterations")
	ax_ar2.set_ylabel("Predicted Delta G (kcal/mol)")

	ax_ar3.plot(range(len(durations)),durations,'x',color='g',ms=3,mew=5)
	ax_ar3.grid(True)
	ax_ar3.set_title("Duration of New Ligand Selection (seconds)")
	ax_ar3.set_xlabel("Number of Training Iterations")
	ax_ar3.set_ylabel("Time (seconds")

	ax_ar4.plot(range(len(mean_error)),mean_error,'x',color='r',ms=3,mew=5)
	ax_ar4.grid(True)
	ax_ar4.set_title("Mean Error During Sampling")
	ax_ar4.set_xlabel("Number of Training Iterations")
	ax_ar4.set_ylabel("Mean Error (kcal/mol)")

	ax_ar5.plot(training_data_known,training_data_predictions,'x',color='k',ms=3,mew=5)
	ax_ar5.plot(training_data_known,np.poly1d(np.polyfit(training_data_known, training_data_predictions, 1))(training_data_known))
	ax_ar5.grid(True)
	ax_ar5.set_title("Training Data Predicted vs Actual Delta G (kcal/mol)")
	ax_ar5.set_xlabel("Known Delta G (kcal/mol)")
	ax_ar5.set_ylabel("Predicted Delta G (kcal/mol)")

	plt.show()
	#pp = PdfPages("plots/{}_samples.pdf".format(len(predictions))
	#plt.savefig()
	#pp.close()

def main():
	global lig_feature_data
	lig_feature_data = getAllFeatureData(lig_data)
	naive_set, deltaGs = drawNaiveSet(lig_feature_data)
	model = fitSet(naive_set, deltaGs)
	error = None
	errors = []
	mean_error = []
	predictions = []
	meas = []
	durations = []
	next_set = naive_set
	current_layers = 0
	while (((error == None) or (error > threshold)) | (current_layers < sample_layers)):
		this_set = next_set
		model, next_set, deltaGs, error, meas, pred, durations = updateModel(model, this_set, deltaGs, meas, durations)
		errors.append(error)
		mean_error.append(np.mean(errors))
		predictions.append(pred)
		tdk, tdp = getTrainingDataStats(model)
		current_layers += 1
		print "Iterated {} layers out of {}\n".format(current_layers, sample_layers)
	makePlots(errors, predictions, meas, durations, mean_error, tdk, tdp)

main()
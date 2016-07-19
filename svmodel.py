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
import matplotlib.pyplot as plt

data_file_name = "sorted_ligs_by_deltaG.npy"
sample_layers = int(sys.argv[1])				# naive dataset takes <sample_layer> samples per bin
lig_data = np.load(data_file_name)
lig_count = len(lig_data)

print "\n Ligand binary read: {} molecules".format(lig_count)

# Extract the features of interest from a specified molecule
def extractFeatureData(mol):
	smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
	slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
	peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)

	feats = [smr_vsa,slogp_vsa,peoe_vsa]

	feature_data = []
	for f in feats:
		for data in f:
			feature_data.append(data)
	#feature_data = np.asarray(feature_data)						# convert to numpy array
	return feature_data

# Measures the distribution of delta G values in the dataset
# Used for plotting the dataset histogram
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
def getNaiveDataset():
	global lig_data
	print "\nFinding initial sample dataset"
	data_dict = []
	naive_samples = []
	dgs = []
	sampled = []
	for d in lig_data:						# Build a data set, with mols being stored under the entry for
		if d[2] in data_dict:				#    their delta G value
			data_dict[data_dict.index(d[2]) + 1].append([d[0],d[1]])
		else:
			data_dict.append(d[2])
			data_dict.append([[d[0],d[1]]])
	# data_dict = [-6.0, ['ZINC1234567',<rdkit mol>,'ZINC987654',<rdkit mol>,...], -5.9, ['ZINC2345678',<rdkit mol>,'ZINC8383838',<rdkit mol>,...], ...]
	deltaG_count = len(data_dict)/2
	print " Found {} unique delta G values in dataset.".format(deltaG_count)
	print " Getting initial feature data..."
	for dg in data_dict:
		if (isinstance(dg,float)):
			continue
		else:
			rand_index = int(np.random.random() * (len(dg) - 1))
			if (isinstance(dg[rand_index],str)):
				rand_index += 1

			sampled.append(dg[rand_index][0])
			naive_samples.append(extractFeatureData(dg[rand_index][1]))
	print " Getting initial target data..."
	for lig in lig_data:
		if lig[0] in sampled:
			dgs.append(float(lig[2]))
	print " Removing sampled data from unsampled data."
	for samp in sampled:
		i = 0
		for mol in lig_data:
			if mol[0] == samp:
				lig_data = np.delete(lig_data,i,0)
			i += 1
	print "Successfully sampled total of {} molecules.\n".format(lig_count - len(lig_data))
	return naive_samples, dgs

# Focus another sampling on the delta G region where the model's error is highest
def getNextDataset(sampling_count, max_sampling_count, dg_max_err, max_err):
	global lig_data
	print "Value of model's max error -- expected {}, prediction was off by {}".format(dg_max_err, max_err)
	print "\nAdding sample layer {} of {}".format(sampling_count + 1,max_sampling_count)
	data_dict = []
	new_samples = []
	dgs = []
	sampled = []
	for d in lig_data:						# Build a data set, with mols being stored under the entry for
		if d[2] in data_dict:				#    their delta G value
			data_dict[data_dict.index(d[2]) + 1].append([d[0],d[1]])
		else:
			data_dict.append(d[2])
			data_dict.append([[d[0],d[1]]])
	# data_dict = [-6.0, ['ZINC1234567',<rdkit mol>,'ZINC987654',<rdkit mol>,...], -5.9, ['ZINC2345678',<rdkit mol>,'ZINC8383838',<rdkit mol>,...], ...]
	deltaG_count = len(data_dict)
	i = 0									# index of the dg entry where the model's error is highest
	print "Making new samples"
	print " Getting layer {} feature data...".format(sampling_count + 1)
	for d in range(len(data_dict)):
		if (isinstance(data_dict[d],float)):
			if data_dict[d] == dg_max_err:
				i = d + 1
				rand_index = int(np.random.random() * len(data_dict[i]))
				sampled.append(data_dict[i][rand_index][0])
				new_samples.append(extractFeatureData(data_dict[i][rand_index][1]))
	print " Getting layer {} target data...".format(sampling_count + 1)
	for lig in lig_data:
		if lig[0] in sampled:
			dgs.append(float(lig[2]))
	print " Removing sampled data from unsampled data."
	for samp in sampled:
		i = 0
		for mol in lig_data:
			if mol[0] == samp:
				lig_data = np.delete(lig_data,i,0)
	#new_samples = getRandomSamples(data_dict,i,sampled)
	print "Successfully sampled total of {} molecules.\n".format(lig_count - len(lig_data))
	return new_samples, dgs


				
# Returns normally distributed set of samples, focused around a given delta G
def getRandomSamples(data_dict, i, sampled):
	min_i = 0
	max_i = 0
	dgs_to_sample = [i]
	new_samples = []
	inc = 1
	if len(sys.argv >= 3):
		resample_factor = int(sys.argv[2])
	for p in range(resample_factor):		#  Determine the spread from which to draw random samples
		for d in dgs_to_sample:
			dgs_to_sample.append(d)
		if (i + inc < len(data_dict)):
			dgs_to_sample.append(i + inc)
		if (i - inc >= 0):
			dgs_to_sample.append(i - inc)
		inc += 1
		print dgs_to_sample, inc
	print dgs_to_sample
	print " Adding {} more samples, focused around delta G: {}".format(len(dgs_to_sample),data_dict[i-1])
	for dg in dgs_to_sample:
		rand_index = int(np.random.random() * len(data_dict[dg]))
		if (isinstance(data_dict[rand_index],str)):
			rand_index += 1
		sampled.append(data_dict[rand_index - 1])
		print data_dict[dg][rand_index]
		print dg
		new_samples.append(extractFeatureData(data_dict[dg][rand_index]))
	print " Found {} more samples, for a total of {} sampled molecules.".format(len(new_samples),len(sampled))
	return new_samples, sampled


# Fit a dataset to an SVR model
def fitModel(samples, dgs):
	model = SVR(kernel='rbf')
	x = []
	y = []

	data_dict = []
	sample_count = len(samples)
	for d in lig_data:						# Build a data set, with mols being stored under the entry for
		if d[2] in data_dict:				#    their delta G value
			data_dict[data_dict.index(d[2]) + 1].append([d[0],d[1]])
		else:
			data_dict.append(d[2])
			data_dict.append([[d[0],d[1]]])
	for s in range(sample_count):
		x.append(samples[s])
		y.append(dgs[s])
	x = np.asarray(x)

	print "Fitting model, {} data points...".format(sample_count)
	model.fit(x,y)
	print "Model fitted."
	return model, sample_count

# Test SVR model against the test data
def testModel(model, sample_count):
	predicted = []
	expected = []
	test_data = []
	print " Compiling test data from {} of the remaining {} ligands...".format(len(lig_data)/3, len(lig_data))
	c = 0
	for lig in lig_data:
		if c % 3 == 0:
			expected.append(float(lig[2]))
			test_data.append(extractFeatureData(lig[1]))
		c += 1
	print "Test data compiled"
	test_data = np.asarray(test_data)
	predicted = model.predict(test_data)
	print "Predictions made"
	errs = abs(predicted - expected)
	max_err = 0
	dg_max_err = 0
	for e in errs:
		if e > max_err:
			max_err = e
			dg_max_err = expected[np.where(errs==e)[0][0]]
	return (dg_max_err, e, np.mean(errs), predicted, expected)

# Plot model results
def plotData(predicted,expected,sample_count):
	plt.figure()
	plt.plot(predicted,expected,'x')
	plt.suptitle("Mean Error of Model Over {} Samples".format(sample_count ))
	plt.xlabel("Sample Count")
	plt.ylabel("Mean Model Error")
	plt.show()



samples, dgs = getNaiveDataset()
model, init_sample_count = fitModel(samples, dgs)

dg_max_err, max_err, mean_err, predicted, expected = testModel(model, init_sample_count)

sample_layers_done = 0
mean_errs = []
max_errs = []

while (sample_layers_done < sample_layers):
	new_samples, new_dgs = getNextDataset(sample_layers_done,sample_layers,dg_max_err, max_err)
	for s in new_samples:
		samples.append(s)
	for d in new_dgs:
		dgs.append(d)
	model, last_sample_count = fitModel(samples, dgs)
	dg_max_err, max_err, mean_err, predicted, expected = testModel(model, last_sample_count)
	max_errs.append(max_err)
	mean_errs.append(mean_err)
	sample_layers_done += 1

plotData(len(mean_errs),mean_errs,last_sample_count)
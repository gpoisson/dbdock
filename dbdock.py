import sys, os
import numpy as np 
from functions import getRigidDockingEnergies
from functions import getFlexibleDockingEnergies
from functions import getNamesMols
from functions import getAllFeatures
from svm_nn_dbdock import train_and_test_svm_and_nn
from svm_nn_dbdock import train_SVM

#############################################################
### GLOBAL VARIABLES
#############################################################
input_ligands_path = None
rigid_output_ligands_path = None
flexible_output_ligands_path = None
protein_path = None
autodock_path = None
rigid_ligand_count = None
flexible_ligand_count = None
data_binaries_dir = "data_binaries/"
feature_binary_dir = "{}sorted_features_normalized.npy".format(data_binaries_dir)
labels_binary_dir = "{}sorted_labels.npy".format(data_binaries_dir)
rigid_energies_dir = "{}rigid_energy_data.npy".format(data_binaries_dir)
flexible_energies_dir = "{}flexible_energy_data.npy".format(data_binaries_dir)


# Read through config file and set global parameters
def checkInput(configure):
	global input_ligands_path, rigid_output_ligands_path, flexible_output_ligands_path, protein_path, svm_param_path, nn_param_path, autodock_path, rigid_ligand_count, flexible_ligand_count
	print("\n####################################################\n###  DBDOCK - MACHINE LEARNING-AIDED LIGAND SCREENING\n####################################################\n")
	config = open(configure,'r')
	for line in config:
		if line[0] == '#':
			continue
		else:
			split = line.split(" ")
			if (split[0] == "input_ligands_path"):
				input_ligands_path = split[2][:-1]
			elif (split[0] == "rigid_output_ligands_path"):
				rigid_output_ligands_path = split[2][:-1]
			elif (split[0] == "flexible_output_ligands_path"):
				flexible_output_ligands_path = split[2][:-1]
			elif (split[0] == "protein_path"):
				protein_path = split[2][:-1]
			elif (split[0] == "autodock_path"):
				autodock_path = split[2][:-1]
			elif (split[0] == "rigid_ligand_count"):
				rigid_ligand_count = (int)(split[2][:-1])
			elif (split[0] == "flexible_ligand_count"):
				flexible_ligand_count = (int)(split[2][:-1])
	print("User config loaded")


# Execute rigid docking on <rigid_ligand_count> ligands
def rigidDocking():
	input_lig_contents = os.listdir(input_ligands_path)
	protein_path_contents = os.listdir(protein_path)
	protein = protein_path_contents[0]
	print("Performing rigid docking on {} of {} ligands against receptor {}...".format(rigid_ligand_count,len(input_lig_contents),protein_path_contents))

	os.system("./rigid_docking_script.sh {} {} {}".format(input_ligands_path,protein,rigid_energies_dir))			# Executes bash script to run rigid docking on input ligands

	print("...Rigid docking complete")


# Execute flexible docking on <rigid_ligand_count> ligands
def flexibleDocking():
	protein_path_contents = os.listdir(protein_path)
	protein = protein_path_contents[0]
	print("Performing flexible docking on {} of {} ligands...".format(flexible_ligand_count,rigid_ligand_count))

	os.system("./flexible_docking_script.sh {} {} {}".format(rigid_energies_dir,flexible_energies_dir,protein))			# Executes bash script to run rigid docking on input ligands

	print("...Flexible docking complete")

def parseOutAllLigandsWhichAppearInBothDatasets(rigid_energies, flexible_energies):
	# To create a good set of training and testing data, we can only use ligands which appear in both sets (rigid and flexible).
	# We must first go through and see how many of these values we have. Since this was developed with some previously-
	# generated data, and not by using actual autodock, there wasn't a guarantee that these two sets would contain the same ligands.
	# When using real autodock, however, this should not be an issue.

	features_dataset = []
	labels_dataset = []

	for rigid_energy in rigid_energies:
		ligand_name_r = rigid_energy[0]
		for flexible_energy in flexible_energies:
			ligand_name_f = flexible_energy[0]
			if (ligand_name_r == ligand_name_f):
				features_dataset.append(rigid_energy)
				labels_dataset.append(flexible_energy)
				break

	return features_dataset, labels_dataset


def splitIntoTrainingAndTestingGroup(rigid_energies, flexible_energies):
	# Each entry in rigid_energies and flexible_energies is of the form:
	#   ['LIGAND_NAME', -7.2]
	features_dataset, labels_dataset = parseOutAllLigandsWhichAppearInBothDatasets(rigid_energies, flexible_energies)

	total_samples_count = len(features_dataset)
	training_to_testing_ratio = 0.7
	split_index = (int)(total_samples_count * training_to_testing_ratio)
	data_index = 1

	training_features = features_dataset[:split_index]
	temp = []
	for entry in training_features:
		temp.append([(float)(entry[data_index])])
	training_features = temp

	training_labels = labels_dataset[:split_index]
	temp = []
	for entry in training_labels:
		temp.append([(float)(entry[data_index])])
	training_labels = temp

	testing_features = features_dataset[split_index:]
	temp = []
	for entry in testing_features:
		temp.append([(float)(entry[data_index])])
	testing_features = temp

	testing_labels = labels_dataset[split_index:]
	temp = []
	for entry in testing_labels:
		temp.append([(float)(entry[data_index])])
	testing_labels = temp

	return training_features, training_labels, testing_features, testing_labels


#############################################################
### MAIN PROGRAM
#############################################################
if __name__ == "__main__":
	
	# READ USER INPUT
	checkInput("CONFIG")


	# FEATURE DATA COMPILATION
	#   Attempts to find the feature data in a binary file
	#   If that file does not yet exist, it generates it by reading through
	#     the library of ligand .pdbqt files specified in the config file.
	#   Note: The "features" variable is defined for code legibility purposes but 
	#         the train_and_test_svm_and_nn() method just loads the features binary from disk.

	try:
		print("Attempting to load feature data from a binary at {}...".format(feature_binary_dir))
		names, features = np.load(feature_binary_dir)
		features = features[0]
	except:
		print(" No feature binary available. Checking for existance of binaries containing RDKit mol objects from which to compute features...")
		names, mols = getNamesMols(input_ligands_path,data_binaries_dir)
		names, features = getAllFeatures(names,mols,feature_binary_dir)

	print("Loaded total of {} samples, {} features each.".format(len(features),len(features[0])))



	# RIGID DOCKING  --  TARGET DATA COMPILATION
	#   Uses Autodock Vina to simulate rigid docking for generation of sample target data
	rigidDocking()
	try:
		print(" Attempting to load rigid energy data from a binary...")
		rigid_energies = np.load(rigid_energies_dir)
	except:
		print(" Parsing new rigid docking results for energy values.")
		rigid_energies = getRigidDockingEnergies(rigid_output_ligands_path,rigid_energies_dir)	
	print("Rigid energies data loaded ({} samples)...".format(len(rigid_energies)))


	print("{} sample feature sets".format(len(features)))
	print("{} rigid energies".format(len(rigid_energies)))


	# MACHINE LEARNING MODEL TRAINING ON RIGID DOCKING RESULTS
	#   Results of rigid docking are used with generated feature data to train a Support Vector Machine and a Neural Network which
	#   can try to predict rigid docking energies for future ligand samples. The parameters used in the models are user-specified.
	#   
	#   The r2 value is computed using new test data to measure predictive performance
	svm_model, r2_svm, nn_model, r2_nn = train_and_test_svm_and_nn(feature_binary_dir,rigid_energies_dir)

	# FLEXIBLE DOCKING  --  TARGET DATA COMPILATION
	flexibleDocking()

	print("Rigid energies data ready to be used as feature data ({} samples)...".format(len(rigid_energies)))
	try:
		print(" Attempting to load known flexible docking energy data from a binary for training a new SVM...")
		flexible_energies = np.load(flexible_energies_dir)
		print(" Flexible binding energies binary was successfully loaded.")
	except:
		print(" Parsing new flexible docking results for energy values.")
		flexible_energies_dat_file = "energies_sorted_flexible.dat"
		flexible_energies = getFlexibleDockingEnergies(flexible_energies_dat_file, flexible_energies_dir)	


	svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y = splitIntoTrainingAndTestingGroup(rigid_energies, flexible_energies)

	svm_model, r2_svm = train_SVM(svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y, C=1.0, epsilon=0.01)

	print("Predicting the flexible docking energy of ligands with rigid docking energies of -3.0, -6.0, and -9.0  --> {}".format(svm_model.predict([[-3.0], [-6.0], [-9.0]])))

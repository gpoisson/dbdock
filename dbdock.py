import sys, os
import numpy as np 
from functions import getRigidDockingEnergies
from functions import getNamesMols
from functions import getAllFeatures
from svm_nn_dbdock import train_and_test_svm_and_nn

#############################################################
### GLOBAL VARIABLES
#############################################################
input_ligands_path = None
rigid_output_ligands_path = None
flexible_output_ligands_path = None
protein_path = None
svm_param_path = None
nn_param_path = None
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
			elif (split[0] == "svm_param_path"):
				svm_param_path = split[2][:-1]
			elif (split[0] == "nn_param_path"):
				nn_param_path = split[2][:-1]
			elif (split[0] == "autodock_path"):
				autodock_path = split[2][:-1]
			elif (split[0] == "rigid_ligand_count"):
				rigid_ligand_count = (int)(split[2][:-1])
			elif (split[0] == "flexible_ligand_count"):
				flexible_ligand_count = (int)(split[2][:-1])
	print("User config loaded")


# Execute rigid docking on <rigid_ligand_count> ligands
def rigidDocking():
	input_lig_list = os.listdir(input_ligands_path)
	print("Performing rigid docking on {} of {} ligands...".format(rigid_ligand_count,len(input_lig_list)))

	###################
	#### Insert rigid docking scripts here:
	# os.system("rigid_docking_script.sh")			# Executes bash script to run rigid docking on input ligands

	print("  < INSERT RIGID DOCKING SCRIPT >")
	print("...Rigid docking complete")


# Execute flexible docking on <rigid_ligand_count> ligands
def flexibleDocking():
	print("Performing flexible docking on {} of {} ligands...".format(flexible_ligand_count,rigid_ligand_count))

	###################
	#### Insert flexible docking scripts here:
	# os.system("flexible_docking_script.sh")			# Executes bash script to run rigid docking on input ligands

	print("  < INSERT FLEXIBLE DOCKING SCRIPT >")
	print("...Flexible docking complete")



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
		print("Attempting to load feature data from a binary...")
		names, features = np.load(feature_binary_dir)
		features = features[0]
	except:
		print(" No feature binary available. Checking for existance of binaries containing RDKit mol objects from which to compute features...")
		names, mols = getNamesMols(input_ligands_path,data_binaries_dir)
		names, features = getAllFeatures(names,mols,feature_binary_dir)
		features = features[0]

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
	print(features[0])
	print(rigid_energies[0])


	# MACHINE LEARNING MODEL TRAINING ON RIGID DOCKING RESULTS
	#   Results of rigid docking are used with generated feature data to train a Support Vector Machine and a Neural Network which
	#   can try to predict rigid docking energies for future ligand samples. The parameters used in the models are user-specified.
	#   
	#   The r2 value is computed using new test data to measure predictive performance
	svm_model, r2_svm, nn_model, r2_nn = train_and_test_svm_and_nn(feature_binary_dir,rigid_energies_dir)



	# FLEXIBLE DOCKING  --  TARGET DATA COMPILATION
	flexibleDocking()
	

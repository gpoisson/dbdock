import sys, os
import numpy as np 
from functions import getRigidDockingEnergies
from functions import getNamesMols
from functions import getAllFeatures
from svm_nn_dbdock import train_and_test_svm_and_nn

#############################################################
### GLOBAL VARIABLES
#############################################################
perform_rigid_docking = True
perform_flexible_docking = True
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
	global input_ligands_path, rigid_output_ligands_path, flexible_output_ligands_path, protein_path, svm_param_path, nn_param_path, autodock_path, rigid_ligand_count, flexible_ligand_count, perform_rigid_docking, perform_flexible_docking
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
			elif (split[0] == "perform_rigid_docking"):
				if (split[2][:-1] == "No"):
					perform_rigid_docking = False
			elif (split[0] == "perform_flexible_docking"):
				if (split[2][:-1] == "No"):
					perform_flexible_docking = False

# Execute rigid docking on <rigid_ligand_count> ligands
def rigidDocking():
	input_lig_list = os.listdir(input_ligands_path)
	print("Performing rigid docking on {} of {} ligands...".format(rigid_ligand_count,len(input_lig_list)))
	print("...Rigid docking complete")

# Execute flexible docking on <rigid_ligand_count> ligands
def flexibleDocking():
	print("Performing flexible docking on {} of {} ligands...".format(flexible_ligand_count,rigid_ligand_count))
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
	try:
		features = np.load(feature_binary_dir)
	except:
		names, mols = getNamesMols(input_ligands_path,data_binaries_dir)
		features = getAllFeatures(names,mols,feature_binary_dir)

	# RIGID DOCKING
	if (perform_rigid_docking):
		rigidDocking()
	else:
		print("Skipping rigid docking...")
	try:
		rigid_energies = np.load(rigid_energies_dir)
	except:
		rigid_energies = getRigidDockingEnergies(rigid_output_ligands_path)	
	print("Rigid energies data loaded...")


	# MACHINE LEARNING MODEL TRAINING ON RIGID DOCKING RESULTS
	svm_model, r2_svm, nn_model, r2_nn = train_and_test_svm_and_nn(feature_binary_dir,rigid_energies_dir)


	# FLEXIBLE DOCKING
	if (perform_flexible_docking):
		flexibleDocking()
	else:
		print("Skipping flexible docking...")
	

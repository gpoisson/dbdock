import sys, os
import numpy as np 
from functions import getRigidDockingEnergies
import svm_nn_dbdock

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
initial_ligand_count = None

# Read through config file and set global parameters
def checkInput(configure):
	global input_ligands_path, rigid_output_ligands_path, flexible_output_ligands_path, protein_path, svm_param_path, nn_param_path, autodock_path, initial_ligand_count
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
			elif (split[0] == "initial_ligand_count"):
				initial_ligand_count = (int)(split[2][:-1])

# Execute rigid docking on <initial_ligand_count> ligands
def rigidDocking():
	input_lig_list = os.listdir(input_ligands_path)
	print("Performing rigid docking on {} of {} ligands...".format(initial_ligand_count,len(input_lig_list)))
	print("...Rigid docking complete")

# Execute flexible docking on <initial_ligand_count> ligands
def flexibleDocking():
	print("Performing flexible docking...")
	print("...Flexible docking complete")

if __name__ == "__main__":
	checkInput("CONFIG")
	rigidDocking()

	rigid_energies = getRigidDockingEnergies(rigid_output_ligands_path)

	flexibleDocking()
	
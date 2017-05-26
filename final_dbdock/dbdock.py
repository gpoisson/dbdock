########################
#    dbdock 
#    author: Greg Poisson
#

verbose = True
import get_config, dock
import os

#	User provides ligands and proteins as input

[ligands_path,protein_path,svm_param_path,autodock_path,initial_ligand_count] = get_config.read()

#   Define list of unknown ligands

ligand_list = os.listdir(ligands_path)
ligand_count = len(ligand_list)

if (ligand_count <= 0):
	if (verbose):
		print("No files found in <ligands_path> directory listed:  {}".format(ligands_path))
elif (initial_ligand_count > ligand_count):
	if (verbose):
		print("Initial ligand count {} is greater than number of files in <ligands_path> directory ({} found).".format(initial_ligand_count, ligand_count))
elif ((ligand_count > 0) & (initial_ligand_count > ligand_count)):
	if (verbose):
		dock.rigidDock(initial_ligand_count,ligands_path,verbose)
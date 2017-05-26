########################
#   get_config 
#   author: Greg Poisson
#
#	Reads dbdock config file and returns arguments to dbdock.
#

# Return dbdock config args to dbdock
def read():
	config = open("CONFIG",'r')

	ligands_path = None
	protein_path = None
	svm_param_path = None
	autodock_path = None

	for line in config:
		line_split = line.split(" ")
		if (line_split[0] == "#"):										# Ignore comments
			continue
		elif (len(line_split) == 3):
			if(line_split[1] == "="):
				if(line_split[0] == "ligands_path"):
					ligands_path = line_split[2]
				elif(line_split[0] == "protein_path"):
					protein_path = line_split[2]
				elif(line_split[0] == "svm_param_path"):
					svm_param_path = line_split[2]
				elif(line_split[0] == "autodock_path"):
					autodock_path = line_split[2]
				elif(line_split[0] == "initial_ligand_count"):
					initial_ligand_count = (int)(line_split[2])

	output = [ligands_path,protein_path,svm_param_path,autodock_path,initial_ligand_count]
	for index in range(len(output)):
		if (isinstance(output[index],str)):
			output[index] = output[index][:-1]							# Remove new lines from strings
	return output
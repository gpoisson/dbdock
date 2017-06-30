# dbdock

NOTE: Be cautious about manually opening the provided ligands directory in a file explorer as there are several thousand files, and it will consume significant resources to view them. If one wishes to inspect a list of the ligands in the directory, there is a brief Python script below which will help.

Cloning the dbdock repo will include a full set of 198,628 .pdb files representing ligands, as well as 186,132 .pdbqt files representing the ligands after rigid docking (minus approximately 12,000 which experienced errors).





To create a list of the ligand coordinate files contained in a directory called "ligands/", this Python script will help:

	import os
	directory = "ligands/"
	ligand_list = os.listdir(directory)
	for line in ligand_list:
		os.system("{}\n >> list_of_ligands.txt".format(line))
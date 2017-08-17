# dbdock

dbdock is a package of Python and bash scripts, including sample data, to demonstrate the use of drug binding simulation data with machine learning methods to predict binding behavior. 

Required Python 2.7 dependencies are:

	numpy
	scipy
	rdkit
	sklearn
	torch
	matplotlib
	sys
	os

torch will make you go to the website for install instructions; pip will not work for it. However the instructions are straightforward:
	http://pytorch.org/

dbdock's non-machine learning parameters are interactive via the CONFIG file in the repo root directory. 
 1) "Raw" ligand data (pre-rigid docking; '.pdb' files expected) go into the 'input_ligands_path' directory specified in CONFIG.
 2) Receptors go into the 'protein_path' directory specified in CONFIG. (Note: Only one receptor is used if multiple are provided, so just provide one.)
 3) Rigid docking is performed and the output is placed in the 'rigid_output_ligands_path' directory specified in CONFIG.
 4) The version of autodock used by dbdock is the 'autodock_path' directory specified in CONFIG.
 5) The maximum number of ligands to run through rigid and flexible docking are specified respectively in CONFIG.
 6) The option to add noise to the data after it is initially processed is specified in CONFIG.

 dbdock's machine learning parameters are currently global variables listed at the top of this file: 
 	svm_nn_dbdock.py
 and can be edited there. The default values should be sufficient for the provided sample data but they are a good place fore experimentation.

dbdock currently simulates the use of autodock for rigid docking by including sample data for pre- and post-rigid docking. Flexible docking is not simulated with any data, but there are scripts: 
	rigid_docking_script.sh
	flexible_docking_script.sh
included with this repository which are addressed below, and which can be edited to enable the "live" use of autodock for both rigid and flexible docking. These scripts are already executed during dbdock and just need the provided stencil code to be replaced with actual autodock commands.

Make sure that the 'data_binaries' directory is empty any time dbdock is executed on a dataset for the first time. The results of various compute-intensive operations are saved as binaries in this directory to save time in the case of successive executions on the same dataset (since the data is shuffled randomly during the machine learning stage and results may vary from one execution to the next).
import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import operator
import os, sys



# Iterate through autodock outputs and obtain list of energies
def getRigidDockingEnergies(autodock_output_directory,rigid_energies_dir):
	ligand_files = os.listdir(autodock_output_directory)
	energies = []
	for filename in ligand_files:
		ligand_name = filename[:-6]
		file = open("{}/{}".format(autodock_output_directory,filename),'r')
		for line in file:
			split = line.split(" ")
			if (len(split) > 1):
				if ((split[1]) == "VINA"):															# autodock vina prints the computed binding affinity in a specifically-formatted line
					try:
						ki = (float)(split[8])
						energies.append([ligand_name,ki])
					except:
						try:
							ki = (float)(split[7])
							energies.append([ligand_name,ki])
							print("Energy for {}".format(ligand_name))
						except:
							print("FILE NOT WELL-FORMED: {}\n{}".format(filename,split))

	# Since mulitple modes can (and should be) present in the rigid docking file, we only keep the most negative value for each ligand
	# This is done by using a dictionary, which acts as a set, and only allows one value per key.
	# Then the ligand energies are fed in one at a time, comparing with any possible previous modes
	energies_reduced = {}
	for e in energies:
		name = e[0]
		energy = (float)(e[1])
		#  Attempt to find another energy value for the current ligand
		#  Replace it with the new sample if its energy value is lower
		try:
			if (energies_reduced[name] > energy):
				energies_reduced[name] = energy
		#  If there isn't an energy value for the current ligand in the dictionary yet, add the new sample
		except:
			energies_reduced[name] = energy

	temp = []
	for index in energies_reduced:
		temp.append([index,energies_reduced[index]])
	print("Saving rigid energy file. Sample entry [0]: {}".format(temp[0]))
	print("len(temp) = {}".format(len(temp)))
	np.save(rigid_energies_dir,temp)
	return temp

# Executes a Linux command to create a new file which is an existing PDBQT file converted to a PDB format
def convert_PDBQT_to_PDB(filename):
	pdbName = filename[:-2]
	os.system("cut -c-66 {} > {}".format(filename,pdbName))

# Get name and RDKit Mol representation for each ligand
def getNamesMols(input_ligands_path,data_binaries_dir):
	try:
		print(" Attempting to load RDKit mol objects.")
		allValidMols = np.load("ligand_name_rdkit_mol.npy")
		print(" {} RDKit mol objects loaded.".format(len(allValidMols)))
	except:
		print(" No RDKit mol object binary found. Using ligand PDB / PDBQT files to generate new features...")
		allValidMols = []
		ligand_list = os.listdir(input_ligands_path)
		fails = 0
		for ligand_file in ligand_list:
			if ligand_file[-6:] == ".pdbqt":
				try:
					convert_PDBQT_to_PDB("{}{}".format(input_ligands_path,ligand_file))
				except IOError:
					fails += 1
					continue
		ligand_list = os.listdir(input_ligands_path)
		for ligand_file in ligand_list:
			if ligand_file[-4:] == ".pdb":
				ligand_name = ligand_file[:-4]
				mol = Chem.MolFromPDBFile("{}{}".format(input_ligands_path,ligand_file))
				if (mol != None):
					allValidMols.append([ligand_name,mol])
		print " Read in {} ligand files, encountered {} failures.".format(len(ligand_list),fails)
	
	allValidMols = np.asarray(allValidMols)
	np.save("{}ligand_name_rdkit_mol.npy".format(data_binaries_dir),allValidMols)
	names,mols = allValidMols[:,0],allValidMols[:,1]
	return names,mols
	
# Returns a numpy array containing the ligand feature data for all ligands
def getAllFeatures(names,ligands,features_bin_dir):
	features = []
	labels = []
	count = 0
	fails = 0
	print("Using generated RDKit mol objects to produce feature sets...")
	for lig in range(len(ligands)):
		if (count % 100 == 0):
			print("Ligand No: {} / {}".format(count,len(ligands)))
		f_data = computeFeatures(ligands[lig])
		d_hist = f_data[13:33]															# Checking for ligands where distribution of path distance
		all_zero = True																	#   from hba/hbd pairs failed to compute and marking their
		keep = True																		#   samples for removal from the dataset
		for d in d_hist:
			if d != 0:
				all_zero = False
			elif d == 99999:		
				fails += 1
				keep = False
				break
		if all_zero:
			fails += 1
			keep = False
		if keep:
			features.append(f_data)
		count += 1
	features = np.asarray(features)
	print("Collected  {}  features per sample for {} samples ({} failures)".format(len(features[0]),len(features),fails))
	features = preprocessing.normalize(features,norm='l2',axis=0)
	allValidFeatures = [names, [features]]
	np.save("{}".format(features_bin_dir),allValidFeatures)
	return names, features

def computeFeatures(mol):
	numRings = rdMolDescriptors.CalcNumRings(mol)
	numRotBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
	nitrogenCount = countNitrogens(mol)
	oxygenCount = countOxygens(mol)
	carbonCount = countCarbons(mol)
	boronCount = countBorons(mol)
	phosCount = countPhos(mol)
	sulfurCount = countSulfurs(mol)
	fluorCount = countFluorine(mol)
	iodCount = countIodine(mol)
	doubleBonds = countDoubleBonds(mol)
	surf_area = rdMolDescriptors.CalcLabuteASA(mol)
	mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
	s_logp = rdMolDescriptors.SlogP_VSA_(mol)
	dist_hs = recurseMolHCount(mol)
	output = [numRings, nitrogenCount, oxygenCount, carbonCount, boronCount, phosCount, sulfurCount, fluorCount, iodCount, doubleBonds, surf_area, mol_weight]
	for s in s_logp:
		output.append(s)
	for d in dist_hs:
		output.append(dist_hs[d])
	return output

def recurseMolHCount(mol):
	# Add H atoms to molecule
	mol = Chem.AddHs(mol)
	# Build hash table of atoms
	numAtoms = len(mol.GetAtoms())
	atom_table = {}
	hbond_table = []
	for atom_index in range(numAtoms):
		atom = mol.GetAtomWithIdx(atom_index)
		atom_table[atom_index] = []
		neighbors = atom.GetNeighbors()
		for n in neighbors:
			atom_table[atom_index].append(n.GetIdx())
		name = atom.GetSymbol()
		if ((name == "N") | (name == "O") | (name == "F") | (name == "S")):
			hbond_table.append(atom_index)
	# atom_table is now a list of lists:  eg ([atom_index: [neighbor_atom_index_1, neighbor_atom_index_2,...]])
	dists = getDists(numAtoms, hbond_table, atom_table)
	return dists

def getDists(numAtoms, hbond_table, atom_table):
	dists = {}
	for i in range(21):
		dists[i] = 0

	for index1 in range(len(hbond_table)):
		for index2 in range(len(hbond_table)):
			if (index1 < index2):
				dist = getDist(hbond_table[index1], hbond_table[index2], atom_table)
				if dist in dists:
					dists[dist] += 1
	return dists

# Use Dijkstra's shortest path algorithm to find shortest path from atom at index1 to atom at index2
def getDist(index1, index2, atom_table):
	start = index1 	# index of one of terminal atoms
	goal = index2 	# index of one of terminal atoms
	distances = {}	# dict of minimum distances to atoms in atom_table
	unvisited = []	# list of indeces of unvisited atoms in atom_table
	goal_found = False		# flag to make computation loop until complete

	# populate the distance table
	for atom in atom_table:
		if (atom == start):
			distances[atom] = 0
		else:
			distances[atom] = 99999
		unvisited.append(atom)		# store the dict index

	closest = None			# closest unvisited atom to starting point
	closest_dist = None		# distance of closest unvisited atom
	visited_index = 0
	failsafe = 0			# on occasion, some molecules don't have correct bond data, resulting in partitioned graphs through which Dijkstra cannot find a path. This cuts off infinite loops
	while (goal_found is not True):			# goal_found is false until the shortest path from start to goal is found
		for atom in unvisited:								# loop through unvisited atoms, find nearest unvisited
			if (closest == None):
				closest = atom
				closest_dist = distances[atom]
				visited_index = unvisited.index(atom)
			elif (atom in distances): 
				if (distances[atom] < closest_dist):
					if (atom in unvisited):
						closest = atom
						closest_dist = distances[atom]
						visited_index = unvisited.index(atom)
				elif (failsafe > 500):
					return distances[goal]

		unvisited[visited_index] = -1
		outline = ""
		d_outline = ""
		for unv in unvisited:
			outline += "{} ".format(unv)
		for d in distances:
			d_outline += "{} ".format(distances[d])

		if (closest == goal):								# determine if the goal has been visited
			goal_found = True
			return distances[goal]
		neighbors = atom_table[closest]
		for n in neighbors:									# update shortest distance to current atom's neighbors
			current_path_length = distances[n]
			if (closest_dist + 1 < current_path_length):
				distances[n] = closest_dist + 1
		closest_dist = 99999								# visited atoms are removed from the unvisited list, so this value is reset
		failsafe += 1

def countNitrogens(mol):
	smile = Chem.MolToSmiles(mol)
	nitrogenCount = 0
	for ch in smile:
		if ch == 'N':
			nitrogenCount += 1
	return nitrogenCount

def countOxygens(mol):
	smile = Chem.MolToSmiles(mol)
	oxygenCount = 0
	for ch in smile:
		if ch == 'O':
			oxygenCount += 1
	return oxygenCount

def countCarbons(mol):
	smile = Chem.MolToSmiles(mol)
	carbonCount = 0
	for ch in smile:
		if ch == 'C':
			carbonCount += 1
	return carbonCount

def countBorons(mol):
	smile = Chem.MolToSmiles(mol)
	boronCount = 0
	for ch in smile:
		if ch == 'B':
			boronCount += 1
	return boronCount

def countIodine(mol):
	smile = Chem.MolToSmiles(mol)
	iodCount = 0
	for ch in smile:
		if ch == 'I':
			iodCount += 1
	return iodCount

def countPhos(mol):
	smile = Chem.MolToSmiles(mol)
	phosCount = 0
	for ch in smile:
		if ch == 'P':
			phosCount += 1
	return phosCount

def countFluorine(mol):
	smile = Chem.MolToSmiles(mol)
	fluorCount = 0
	for ch in smile:
		if ch == 'F':
			fluorCount += 1
	return fluorCount

def countSulfurs(mol):
	smile = Chem.MolToSmiles(mol)
	sulfurCount = 0
	for ch in smile:
		if ch == 'S':
			sulfurCount += 1
	return sulfurCount

def countDoubleBonds(mol):
	smile = Chem.MolToSmiles(mol)
	doubleBonds = 0
	for ch in smile:
		if ch == '=':
			doubleBonds += 1
	return doubleBonds


'''


# Sort input data into training and testing datasets
# Picks n evenly-indexed mols for training, rest go to testing
def sortInputData(mols,deltaGs):
	train = []
	test = []
	trainIndeces = []		# Keep track of molecules used for training; use the rest for testing
	c = 0
	p = 0
	count = 0

	if v:
		print "Assemble training and testing data..."
	while (count < int(dataSize)):
		if (count == 0):									# If no molecules have been sorted, take the first one
			try:	
				if (mols[0][1] == None):
					continue
				else:
					ligand_name = mols[0][0]
					features = extractFeatureData(mols[0][1])
					dg = deltaGs[ligand_name]
					train.append([ligand_name,features,dg])
					trainIndeces.append(0)
			except KeyError:
				continue
		else:												# If more than two molecules have been sorted, evenly distribute the rest
			c += 1
			index = int(len(mols)/2**p) + int(len(mols)*c/2**(p-1)) - 1				# Apply a logarithmic distribution to evenly choose training/testing samples
			if (index >= len(mols)):
				p += 1
				c = 0
				index = int(len(mols)/2**p) + int(len(mols)*c/2**(p-1)) - 1
			try:	
				if (mols[index][1] == None):
					continue
				else:
					ligand_name = mols[index][0]
					features = extractFeatureData(mols[index][1])
					dg = deltaGs[ligand_name]
					train.append([ligand_name,features,dg])
					trainIndeces.append(index)
			except KeyError:
				continue
		count = len(train)

	count = 0
	for mol in mols:										# Use record of training molecule indeces to add non-training molecules to test
		if count in trainIndeces:
			continue
		else:
			try:
				if (mol[1] == None):
					continue
				else:
					ligand_name = mols[count][0]
					features = extractFeatureData(mols[count][1])
					dg = deltaGs[ligand_name]
					test.append([ligand_name,features,dg])
			except KeyError:
				continue
		count += 1
	if v:
		print "Sorted {} molecules into {} training points and {} test points.".format(len(mols),len(train),len(test))
	return train, test


# Sort input data into training and testing datasets
# Places every nth mol into train, rest into test
def sortInputStep(mols,deltaGs):
	train = []
	test = []
	count = 0
	test_frac = int(len(mols)/int(dataSize))

	if v:
		print "Assemble training and testing data..."
	for mol in mols:
		if (mol[1] == None):
			continue
		else:
			ligand = mol[0]									# ligand name (string)
			features = extractFeatureData(mol[1])			# mol[1] = mol object (rdkit)
			try:
				dg = deltaGs[ligand]							# deltaG for this ligand
				if (count % test_frac != 0):
					test.append([ligand,features,dg])
				else:
					train.append([ligand,features,dg])
				count += 1
			except KeyError:
				continue
	return train, test

# Save the compiled data as a binary file for faster loading
def saveAsBin(train,test):
	binary_outfile = "sorted_ligs_{}.npz".format(dataSize)
	np.savez(binary_outfile,train,test)

# Takes a mol object and draws it to a PNG file
def drawMolToPng(mol):
	m2 = Chem.AddHs(mol[1])
	AllChem.EmbedMolecule(m2)
	AllChem.Compute2DCoords(m2)
	Draw.MolToFile(m2,"{}.png".format(mol[0]))

'''
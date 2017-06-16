import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import operator
import os, sys

v = True
energies_file = "energies_sorted.dat"

def makeSinglePlot(x_data,y_data,title='Plot',x_label='x',y_label='y',axes_on=True,marker_type='x',add_lobf=True,x_min=None,x_max=None,y_min=None,y_max=None,axis_equal=False):
	plt.figure()																	# make a plot figure
	plt.plot(x_data,y_data,marker_type)												# add the data to the plot
	if add_lobf:																	# add a line of best fit
		plt.plot(x_data, np.poly1d(np.polyfit(x_data, y_data, 1))(x_data))
	plt.suptitle(title)																# add plot title, labels
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if (x_min != None):																# fix boundaries of the plot
		plt.xlim([x_min,x_max])
	if (y_min != None):
		plt.ylim([y_min,y_max])
	if axes_on:																		# enable grid axes
		plt.grid(True)
	if axis_equal:
		plt.axis('equal')
	plt.show()

def compileFeaturesAndLabels():
	ligands = np.load("unsorted_ligs.npy")
	
	[features, labels] = getAllFeatures(ligands)
	np.save("features.npy",features)
	np.save("labels.npy",labels)

# Compile all ligand name, coordinate, and energy data into one comprehensive binary file for faster accessing
def makeUnsortedBinary(coords_dir,out_dir):
	allValidMols = []
	ligand_list = os.listdir(coords_dir)
	fails = 0
	for ligand_file in ligand_list:
		if ligand_file[-4:] == "mol2":
			ligand_name = ligand_file[:-4]
			print(ligand_file,ligand_name)
			try:
				mol = Chem.MolFromMol2File("{}{}".format(coords_dir,ligand_file))
				allValidMols.append([ligand_name,mol])
			except IOError:
				fails += 1
				continue
	if v:
		print " Read in all {} molecules, encountered {} failures.".format(len(ligand_list),fails)
	
	np.save("{}ligand_name_rdkit_mol.npy".format(out_dir),allValidMols)

# Splits the unsorted binary into multiple smaller binaries.
# Binaries are <100 MB for GitHub purposes.
def makeSplitUnsortedBinary():
	mols_per_bin = 100000				# number of ligands to store in each binary file (to regulate file size)
	binary = loadUnsortedBinary()
	bin_count = 0
	binr = []
	for mol in binary:
		binr.append(mol)				# add ligands to binr array
		if (len(binr) >= mols_per_bin):										# if max no. of mols is reached,
			print ("Binr size: {} mols".format(len(binr)))					# save out the binary and start a
			binr = np.asarray(binr)											# new one.
			np.save("unsorted_ligs_part_{}.npy".format(bin_count),binr)
			bin_count += 1
			binr = []
	if (len(binr) >= 0):													# once all mols have been read through,
		binr = np.asarray(binr)												# save out whatever mols are left in 
		np.save("unsorted_ligs_part_{}.npy".format(bin_count),binr)			# memory

# Assembles split binaries into one single ligand db binary
def combineSplitBinaries():
	files = os.listdir("./")
	bins = []
	master_bin = []
	for f in files:
		if (f[:-5] == "unsorted_ligs_part_"):			# find any partial ligand binaries in the directory
			bins.append(f)
	for b in bins:
		t = np.load(b)
		for mol in t:
			master_bin.append(mol)
	master_bin = np.asarray(master_bin)
	np.save("unsorted_ligs.npy",master_bin)

# Reads in energy valus for each ligand
def readInputEnergies():
	deltaGs = {}
	engs = open(energies_file,'r')
	for line in engs:
		line = line.split()
		if (len(line) >= 2):
			name = line[0]
			if (name[-1] == '.'):		# trim off trailing periods
				name = name[:-1]
			dg = float(line[1])
			try:
				deltaGs[name] = dg
			except KeyError:
				continue
	return deltaGs

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

def loadUnsortedBinary():
	t = np.load("unsorted_ligs.npy")
	return t

# Takes an entry from unsorted_ligs.npy and draws it to a PNG file
def drawMolToPng(mol):
	m2 = Chem.AddHs(mol[1])
	AllChem.EmbedMolecule(m2)
	AllChem.Compute2DCoords(m2)
	Draw.MolToFile(m2,"{}.png".format(mol[0]))

# Iterate through autodock outputs and obtain list of energies
def getRigidDockingEnergies(autodock_output_directory):
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
						except:
							print("FILE NOT WELL-FORMED: {}\n{}".format(filename,split))
	return energies

def getNamesMols(input_ligands_path):
	ligands = os.listdir(input_ligands_path)
	for filename in ligands:
		#try:
		mol = Chem.MolFromMolFile("{}{}".format(input_ligands_path,filename))

def getAllFeatures(ligands):
	features = []
	labels = []
	count = 0
	#print("Starting with library of {} ligands".format(len(ligands)))
	for ligand in ligands:
		if (ligand[mol_index] != None):
			if (count % 10000 == 0):
				print("Ligand No: {} / {}".format(count,len(ligands)))
			f_data = computeFeatures(ligand[mol_index])
			#print("{} features computed; f_data[2] = {}".format(len(f_data),f_data[2]))
			#d_hist = f_data[13:33]					# UNCOMMENT
			'''
			dout = ""
			for d in dout:
				dout += "{} ".format(d)
			print(dout)
			'''
			all_zero = True
			keep = True
			for d in d_hist:
				if d != 0:
					all_zero = False
				elif d == 99999:
					keep = False
					break
			if all_zero:
				keep = False
			if keep:
				features.append(f_data)
				labels.append(ligand[ki_index])
			count += 1
	print("{} labels".format(len(labels)))
	return [np.asarray(features), np.asarray(labels)]

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

# Use Dijkstra's SP algorithm to find shortest path from atom at index1 to atom at index2
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

# combineSplitBinaries()
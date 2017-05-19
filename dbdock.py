import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os, sys
import matplotlib
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

name_index = 0
mol_index = 1
ki_index = 2

C = 10.0
epsilon = 0.1
train_size = 0.005
test_size = 1 - train_size
batches = 5

def main():
	compileFeaturesAndLabels()
	X = np.load("features.npy")
	y = np.load("labels.npy")

	models = []
	X_val = X[0:10000]
	y_val = y[0:10000]
	X = X[10001:]
	y = y[10001:]
	for batch in range(batches):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=batch)
		#X_test = X_test[0:4000]
		#y_test = y_test[0:4000]
		print("X_train contains {} samples; y_train contains {} labels.".format(len(X_train),len(y_train)))
		print("X_test contains {} samples; y_test contains {} labels.".format(len(X_test),len(y_test)))
		clf = SVR(C=C, epsilon=epsilon,kernel='rbf')
		clf.fit(X_train,y_train)
		models.append(clf)
		predicted = clf.predict(X_test)
		errors = abs(predicted - y_test)
		print(clf.score(X_test,y_test))
		print("E: {}".format(np.mean(errors)))

	pred = []
	mean_pred = []
	for b in range(batches):
		predictions = models[b].predict(X_val)
		pred.append([predictions])
	for feature in range(len(pred[0])):
		mean_pred.append(np.mean(predictions[:][feature]))
	errors = abs(mean_pred - y_val)
	print("FINAL: {}".format(models[0].score(X_val, y_val)))

	plt.figure()
	plt.grid(True)
	plt.suptitle("SVM - C={} epsilon={}\n  Training Set: {} ligands".format(C,epsilon,len(X_train)))
	plt.xlabel("Predicted (kcal/mol)")
	plt.ylabel("Actual (kcal/mol)")
	plt.plot(predicted,y_test,'x',ms=2,mew=3)
	plt.show()


def compileFeaturesAndLabels():
	ligs_part_0 = np.load("unsorted_ligs_part_0.npy")
	ligs_part_1 = np.load("unsorted_ligs_part_1.npy")

	ligands = []

	for lig in ligs_part_0:
		ligands.append(lig)

	for lig in ligs_part_1:
		ligands.append(lig)

	[features, labels] = getAllFeatures(ligands)
	np.save("features.npy",features)
	np.save("labels.npy",labels)

def getAllFeatures(ligands):
	features = []
	labels = []
	count = 0
	for ligand in ligands:
		if (ligand[mol_index] != None):
			print("Ligand No: {} / {}".format(count,len(ligands)))
			features.append(computeFeatures(ligand[mol_index]))
			labels.append(ligand[ki_index])
		count += 1
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
	tpsa = rdMolDescriptors.CalcTPSA(mol)
	dist_hs = recurseMolHCount(mol)
	output = [numRings, nitrogenCount, oxygenCount, carbonCount, boronCount, phosCount, sulfurCount, fluorCount, iodCount, doubleBonds, surf_area, mol_weight, tpsa]
	for d in dist_hs:
		output.append(d)
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
	print(" Getting distances between hba/ds... (Hbond table contains {} entries)".format(len(hbond_table)))
	dists = getDists(numAtoms, hbond_table, atom_table)
	print(" All distances computed.")
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

	print("   Finding shortest path from [{}] to [{}]...".format(start,goal))
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
				elif (failsafe > 10000):
					for d in distances:
						if distances[d] == 99999:
							distances[d] = closest_dist + 1
					return distances[goal]

		unvisited[visited_index] = -1
		outline = ""
		d_outline = ""
		for unv in unvisited:
			outline += "{} ".format(unv)
		for d in distances:
			d_outline += "{} ".format(distances[d])

		if (closest == goal):			
			print("   Shortest path computed: From [{}] to [{}] - path is length: {}".format(start,goal,distances[goal]))					# determine if the goal has been visited
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


main()

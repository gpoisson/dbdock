import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os, sys
import matplotlib
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

name_index = 0
mol_index = 1
ki_index = 2

C = 10.0
epsilon = 0.1
test_size = 0.05

def main():
	#compileFeaturesAndLabels()
	X = np.load("features.npy")
	y = np.load("labels.npy")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
	print("X_train contains {} samples; y_train contains {} labels.".format(len(X_train),len(y_train)))
	print("X_test contains {} samples; y_test contains {} labels.".format(len(X_test),len(y_test)))
	clf = SVR(C=C, epsilon=epsilon,kernel='rbf')
	clf.fit(X_train,y_train)
	predicted = clf.predict(X_test)
	errors = abs(predicted - y_test)
	print(clf.score(X_test,y_test))

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
	for ligand in ligands:
		if (ligand[mol_index] != None):
			features.append(computeFeatures(ligand[mol_index]))
			labels.append(ligand[ki_index])
	return [np.asarray(features), np.asarray(labels)]

def computeFeatures(mol):
	numRings = rdMolDescriptors.CalcNumRings(mol)
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
	return [numRings, nitrogenCount, oxygenCount, carbonCount, boronCount, phosCount, sulfurCount, fluorCount, doubleBonds, surf_area, mol_weight, tpsa]

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

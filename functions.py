import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
import MDAnalysis as md
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

# Produce multiple plots on one firgure
# 
#def makeMultiPlot(x_data,y_data,title='Plot',x_label='x',y_label='y',axes_on=True,marker_type='x',add_lobf=True,x_min=None,x_max=None,y_min=None,y_max=None):

# Compile all ligand name, coordinate, and energy data into one comprehensive binary file for faster accessing
def makeUnsortedBinary():
	coords_dir = "lig_mols/"
	allValidMols = []
	ligand_list = os.listdir(coords_dir)
	fails = 0
	for ligand_file in ligand_list:
		ligand_name = ligand_file[:-4]
		try:
				mol = Chem.MolFromMolFile("{}{}".format(coords_dir,ligand_file))
				allValidMols.append([ligand_name,mol])
		except IOError:
			fails += 1
			continue
	if v:
		print " Read in all {} molecules, encountered {} failures.".format(len(ligand_list),fails)
	deltaGs = readInputEnergies()
	data = []
	for mol in allValidMols:
		try:
			dg = deltaGs[mol[0]]
			data.append([mol[0],mol[1],dg])
		except KeyError:
			continue
	data = np.asarray(data)
	np.save("unsorted_ligs.npy",data)
	if v:
		print " Read in {} mols, {} names, {} feature sets, {} dgs.".format(len(allValidMols),len(data[:][0]),len(data[:][1]),len(data[:][2]))
	return data

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

# combineSplitBinaries()
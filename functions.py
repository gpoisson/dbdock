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
	np.save("unsorted.npy",data)
	if v:
		print " Read in {} mols, {} names, {} feature sets, {} dgs.".format(len(allValidMols),len(data[:][0]),len(data[:][1]),len(data[:][2]))
	return data

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

def loadUnsortedBinary():
	t = np.load("unsorted.npy")
	return t
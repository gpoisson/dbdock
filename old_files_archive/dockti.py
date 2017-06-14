'''
Input info:		
Protein XYZ		--------------		---------------		-------------		Drug | KI  |Ligand
Binding			|	Rigid	  | x%	|	Flexible  | y%	|	Thermo	|		-----|-----|-------
	Pocket	-->	|	Docking	  |	-->	|	Docking	  | -->	|	Integr	| -->		 |     |
Drug DB			--------------		---------------		-------------			 |	   |
'''
import sys
import os
import numpy
from split_mol2_convert_receptor import parseConfigFile
from split_mol2_convert_receptor import split_mol2

configFile = ""
flexDockingCriteria = []
thermoIntegCriteria = []
results = [[],[],[]]
inProtein = []
bindingPocket = []
drugDb = []
v = False
vv = False
split = True

# Give program usage
def usage():
	print("    drugPy Usage\n\n\t$ python drugPy.py run.cfg <options>")
	print("\n\t -v\t\tverbose console output\n\t-vv\t\tall debug info + verbose console output\n\t-n\t\tskip splitting mol2 library")

# Check input for selected options
def options():
	global v, vv, split
	if ((len(sys.argv) != 2) & (len(sys.argv) != 3) & (len(sys.argv) != 4)):
		usage()
		return False
	if (len(sys.argv) >= 3):
		if (sys.argv[2] == "-v"):
			v = True
		elif (sys.argv[2] == "-vv"):
			v = True
			vv = True
	if (len(sys.argv) >= 4):
		if (sys.argv[3] == "-n"):
			if v:
				print(" Mol2 split has been skipped.")
			split = False
	return True

# Establish that a config file is present
def checkConfig():
	global configFile
	if (len(sys.argv) > 1):
		configFile = sys.argv[1]
		if v:
			print ("Config file: {}".format(configFile))
		return True
	else:
		print ("No config file given.")
		return False

# Parse config file for program data
def getConfigData():
	if (options() & checkConfig()):
		if vv:
			print("Reading in config data...")
		# Get config data
		parseConfigFile(configFile, vv)
		if split:
			split_mol2(v, vv)
		if v:
			print("Execute:  multi_proc_convert.sh... (may prompt for sudo password)")
		os.system("./multi_proc_convert.sh")

		'''
			Encountering 'broken pipe' error here
		'''

		if vv:
			print("Config data read successfully.")
		return True
	else:
		return False

#def prepareData():

# Execute rigid docking sequence
def rigidDocking():
	print("Rigid Docking...")
	# Apply rigid docking
	print("Rigid Docking applied.")

# Execute flexible docking sequence
def flexibleDocking(flexDockingCriteria):
	print("Flexible Docking...")
	# Apply flexible docking
	print("Flexible Docking applied.")

# Execute thermodynamic integration sequence
def thermoIntegration(thermoIntegCriteria):
	print("Thermodynamic Integration...")
	# Apply thermodynamic integration
	print("Thermodynamic Integration appied.")

# Exectue main drugPy sequence
def main():
	if (getConfigData()):
		rigidDocking()
		flexibleDocking(flexDockingCriteria)
		thermoIntegration(thermoIntegCriteria)

main()
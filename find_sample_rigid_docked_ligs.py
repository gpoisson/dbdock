import os

samples_dir = "sample/"
samples_rigid_dir = "/mnt/sda/drug_binding_data/ligands_after_rigid_docking_10_modes"

samples_list = os.listdir(samples_dir)
samples_rigid_list = os.listdir(samples_rigid_dir)

raw_sample_names = []
for file in samples_list:
	name = file[:-4]
	raw_sample_names.append(name)

for file in raw_sample_names:
	for rigid_file in samples_rigid_list:
		rigid_name = rigid_file[:-6]
		if file == rigid_name:
			os.system("cp {}/{} sample_rigid/".format(samples_rigid_dir,rigid_file))


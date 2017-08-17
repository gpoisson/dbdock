#!/bin/bash

SAMPLES=$1			# dbdock populates this with the path to the ligand files
RECEPTOR=$2			# dbdock populates this with the path to the receptor file
SAMPLES_RIGID=$3	# dbdock populates this with the path to the rigid docking output directory

# This script should contain the commands to run autodock on the .pdb files contained in the directories in the CONFIG file
echo " RIGID DOCKING SCRIPT"

echo "   <  INSERT AUTODOCK COMMANDS FOR RIGID DOCKING HERE  file: $0   line: $#"

# Example:
# autodock $SAMPLES $RECEPTOR $SAMPLES_RIGID
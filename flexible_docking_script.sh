#!/bin/bash

SAMPLES_RIGID=$1	# dbdock populates this with the path to the rigid docking output directory
SAMPLES_FLEXIBLE=$2	# dbdock populates this with the path to the flexible docking output directory
RECEPTOR=$3			# dbdock populates this with the path to the receptor file

# This script should contain the commands to run autodock on the .pdb files contained in the directories in the CONFIG file
echo " FLEXIBLE DOCKING SCRIPT"

echo "   <  INSERT AUTODOCK COMMANDS FOR FLEXIBLE DOCKING HERE  file: $0   line: $#"

# Example:
# autodock $SAMPLES_RIGID $RECEPTOR $SAMPLES_FLEXIBLE
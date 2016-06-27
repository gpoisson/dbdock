# dbdock
Scripts related to machine learning drug binding project

Currently, dbdock.py implements a support vector machine to compare experimentally-determined protein-ligand binding affinities to the structural properties of ligands in a database. The goal is to produce a model that can accurately predict binding affinity for a new ligand with known structural properties. 

# Usage

$ python dbdock.py (verbose) (training size)

> Example:

  $ python dbdock.py True 20000    

> Builds SVR model on 20,000 training samples with verbose output ON

dbdock also uses features.py, an auxillary library containing wrappers for generating matplotlib plots

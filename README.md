# dbdock
Scripts related to machine learning drug binding project

svmodel.py is the script which imports the data binary and fits a support vector machine to predict binding affinity.
functions.py contains a number of auxilliary functions, including graphing and feature extracting functions.
dockti.py is intended to wrap the functionality of svmodel and functions into a script which will screen large databases of ligands for potential drug candidates, and then run the selected samples through autodock.

# Usage

> Currently, svmodel works as a standalone script, and will illustrate prediction error with respect to training iterations.
>   Suggested training sample count: 150
$ python svmodel.py <training_sample_count>

> Example:

  $ python svmodel.py 150

> Builds SVR model on a naive sample across all available delta G values, then adaptively samples 150 additional ligands


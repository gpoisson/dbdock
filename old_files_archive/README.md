# dbdock
Scripts related to machine learning drug binding project

The dbdock.py script uses a support vector machine (SVM) to predict the how readily a given drug molecule will bind to a given protein molecule, using physical characteristics of the drug as the feature space of its prediction. This information is useful when developing potential new treatements for viruses such as Zika or Dengue.

Traditionally, this calculation is computationally expensive and takes many hours to run for a large database of drugs. This algorithm uses machine learning to find a pattern relating physical traits of a drug to its binding affinity. After predicting the free energy of binding for each drug, a more refined calculation can be performed to verify the results of the drugs which are predicted to bind most readily, without needing to iterate over the entire database.

Initially a sample set of drugs is chosen and the true binding affinities are computed and used as target values with which to train a SVM. Then, this model is used to quickly predict the values of the other drugs, and a new sample is chosen using this prediction. By iteratively sampling new drugs based on previous predictions, the model's accuracy is efficiently refined while time spent on expensive computation is minimized.

# Usage

$ python dbdock.py 800 1.0

Trains a model using 800 samples past the initial sample set, using a non-linear kernelized SVM with a C parameter = 1.0

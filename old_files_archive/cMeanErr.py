import numpy as np
import matplotlib.pyplot as plt
import os, sys

training_set_prediction_errs = []
unk_sample_prediction_errs = []
mean_errs_train = []
mean_errs_test = []

def getErrLists():
	global training_set_prediction_errs
	files = os.listdir(".")
	for f in files:
		if f[:6] == "errors":
			training_set_prediction_errs.append(np.load(f))
		elif f[:6] == "model_":
			unk_sample_prediction_errs.append(np.load(f))

def getMeanErrs():
	global mean_errs
	for e in training_set_prediction_errs:
		mean_errs.append(np.mean(e))

def plotTrainingMeanError():
	log_c = range(len(mean_errs))
	plt.figure()
	plt.suptitle("Mean Error of Model After 800 Samples vs Log(C Paramter)")
	plt.plot(log_c,mean_errs,'x',ms=3,mew=3)
	plt.grid(True)
	plt.xlabel("Log(SVM C Parameter)")
	plt.ylabel("Mean Model Error After 800 Samples")
	plt.show()

def main():
	getErrLists()
	getMeanErrs()
	plotTrainingMeanError()

main()
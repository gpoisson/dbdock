#!/bin/bash

for COUNT in {1..50}
	do
		python3 xor.py $COUNT >> test_nn_errs
	done
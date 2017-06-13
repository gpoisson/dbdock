#!/bin/bash

# iterate through single-layer with varying sizes
for SIZE in {1..20000..50}
	do
		python xor.py $SIZE >> res.dat
	done

# iterate through two-layer with varying sizes
for SIZE1 in {1..20000..100}
	do
		for SIZE2 in {1..20000..100}
		do
			python xor.py $SIZE1 $SIZE2 >> res.dat
		done
	done

# iterate through three-layer with varying sizes
for SIZE1 in {1..20000..100}
	do
		for SIZE2 in {1..20000..100}
		do
			for SIZE3 in {1..20000..100}
			do
				python xor.py $SIZE1 $SIZE2 $SIZE3 >> res.dat
			done
		done
	done

# iterate through varying sizes of training sets
for SIZE in {1..60000..1000}
	do
		python xor_train.py $SIZE >> res.dat
	done

# iterate through varying numbers of epochs per model
for COUNT in {1..25}
	do
		python xor_epochs.py $COUNT >> res.dat
	done
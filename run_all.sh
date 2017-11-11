#!/bin/bash

for alens in 1.0 0.5
do
    for shft in 0.005 0.01 0.02
    do
	addqueue -q cmb -s -n 1 -m 0.5 /usr/local/shared/python/2.7.6-gcc/bin/python study_shifts.py ${alens} 0 0 ${shft} 1001

	for i in {0..100}
	do
	    seed=$((1000+$i))
	    addqueue -q cmb -s -n 1 -m 0.5 /usr/local/shared/python/2.7.6-gcc/bin/python study_shifts.py ${alens} 0 1 ${shft} $seed
	done
    done
done

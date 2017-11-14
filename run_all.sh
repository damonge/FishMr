#!/bin/bash

for alens in 1.0 0.5
do
    for rdusy in 1
    do
	python study_shifts.py ${alens} ${rdusy} 0 0 0 0 1001 0
	for shft in 0.005 0.01 0.02
	do
	    for i in {0..100}
	    do
		seed=$((1000+$i))
		python study_shifts.py ${alens} ${rdusy} 1 ${shft} 0 0 $seed 0
		python study_shifts.py ${alens} ${rdusy} 0 0 1 ${shft} $seed 0
	    done
	done
    done
done

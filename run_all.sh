#!/bin/bash

for alens in 1.0 0.5 0.2
#for alens in 1.0
do
    for rdusy in 0 1
#    for rdusy in 0
    do
	for s4 in 1
	do
	    python study_shifts.py ${alens} ${rdusy} 0 0 0 0 1001 1 ${s4}
	    for shft in 0.1 0.05 0.02 0.01 0.005
	    do
		python study_shifts.py ${alens} ${rdusy} 1 ${shft} 0 0 1001 1 ${s4}
		python study_shifts.py ${alens} ${rdusy} 0 0 1 ${shft} 1001 1 ${s4}
	    done
	done
    done
done

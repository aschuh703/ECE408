#!/bin/bash

mkdir -p bench

for i in 0 1 2 3 4 5;
do
	echo "--------------";
	echo "Dataset " $i 
	./template -e ./data/${i}/output.dat -i ./data/${i}/input.dat,./data/${i}/kernel.dat -t vector
done

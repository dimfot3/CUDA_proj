#!/bin/bash

echo "--------------------------------------------Cuda v2 b search--------------------------------------------"

for b in {1..20}
do
	for i in {1..10}
	do
		./main_program 2000 10 3 $b
	done
done

echo "--------------------------------------------End of testing--------------------------------------------"

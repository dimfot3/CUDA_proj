#!/bin/bash

echo "--------------------------------------------Cuda v2 Testing--------------------------------------------"

for dim in {10..10000..999}
do
	for i in {1..10}
	do
		./main_program $dim 10 2 2
	done
done

echo "--------------------------------------------End of testing--------------------------------------------"

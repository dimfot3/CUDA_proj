#!/bin/bash

echo "--------------------------------------------Sequential Testing--------------------------------------------"

for dim in {10..10000..999}
do
	for i in {1..10}
	do
		./main_program $dim 10 0 1000
	done
done

echo "--------------------------------------------End of testing--------------------------------------------"

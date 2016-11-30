#!/bin/bash

#
# USAGE: ./generate_test_data.sh [number_points=100] [max=1000]
#

SIZE=${1-100}
MAX=${2-1000}

for i in `seq 0 1 $SIZE`
do
	echo "`expr $RANDOM % $MAX`,`expr $RANDOM % $MAX`"
done


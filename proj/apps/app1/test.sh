#!/bin/bash
if test -z "$1"
	then
		echo "Please give the times to execute"
		exit 1
	else
		echo ".:Start:."
fi

for (( c=1; c<=100; c++ ))
do
	myvar=$(( $c * 100 ))
	time release/uncompressedIdx $myvar
done

echo ".:End:."

exit 0
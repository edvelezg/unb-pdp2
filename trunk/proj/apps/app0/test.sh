#!/bin/bash
# if test -z "$1"
# 	then
# 		echo "Please give the times to execute"
# 		exit 1
# 	else
# 		echo ".:Start:."
# 		aVar= echo $1
# 		
# fi
# echo $aVar

COUNT=6
while [ $COUNT -gt 0 ]; do
	echo $COUNT
	for (( c=1; c<=10; c++ ))
	do
		myvar=$(( $c * 500 ))
		# time release/compressedIdx $myvar
		release/compressedIdx $myvar
	done
	let COUNT=COUNT-1
done


echo ".:End:."

exit 0
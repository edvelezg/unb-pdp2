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

echo ".:Start:."

COUNT=1
while [ $COUNT -gt 0 ]; do
	echo $COUNT
	for (( c=1; c<=50; c++ ))
	do
		myvar=$(( $c * 100 ))
		# time release/compressedIdx $myvar
		release/uncompressedIdx $myvar
	done
	let COUNT=COUNT-1
done


echo ".:End:."

exit 0
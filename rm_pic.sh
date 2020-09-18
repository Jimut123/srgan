#!/bin/bash
for i in {1..200}
do
	if (( $i % 9 == 0))
	then
		for j in {1..34}
		do
			#epoch_190_index_9.png
			mv epoch_${i}_index_${j}.png keep

		done
	fi
done



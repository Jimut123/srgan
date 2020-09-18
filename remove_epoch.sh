#!/bin/bash
for i in {1..100}
do
	if (( $i % 10 == 0)) 
	then
		mv netD_epoch_4_$i.pth mod_10
		echo $i
	fi
	# netD_epoch_4_195.pth
	# echo $i
done

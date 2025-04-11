#!/bin/bash

ulimit -s unlimited

#define parameters

#main program
for i in $(seq 1 $cycle) #The number of cycles
do
    echo "begin $i cycle"
    cp ./restart.xyz ./model.xyz
    b=$(grep -v '^$' ./restart.xyz | wc -l)
	b=$(($b-2)) #The number of atoms
	for j in $(seq 1 $number) #The number of clusters
	do
        a=$(($RANDOM%$species)) #Select the clusters at random
        cp ./cluster/$a.txt ./cluster/add$j.txt
    done
    python ./cluster/add.py
	for j in $(seq 1 $number)
	do 
	    cat ./cluster/add$j.txt >> ./cluster/add.txt
	done
	c=$(grep -c "" ./cluster/add.txt) 
    d=$(($b+$c))
    cat ./cluster/add.txt >> ./model.xyz
    sed -i '1d' ./model.xyz
    sed -i "1i $d" ./model.xyz
    out=
	python ./cluster/z.py
	e=$(grep -c "" ./restart.xyz)
	f=2
	e=$(($e-$f))
	sed -i '1d' ./restart.xyz
    sed -i "1i $e" ./restart.xyz
	rm ./cluster/add.txt
	cat ./delete.txt >> ./count.txt
	rm ./delete.txt
done
echo "Calculation finished"

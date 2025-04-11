#!/bin/bash

ulimit -s unlimited

cp ./model.xyz ./original.xyz
cp ./model.xyz ./restart.xyz
cp ./cluster/add.py ./cluster/add0.py
cp ./cluster/z.py ./cluster/z0.py
cp ./main.sh ./main0.sh
python ./cluster/parameters.py
bash ./main.sh
cp ./cluster/add0.py ./cluster/add.py
cp ./cluster/z0.py ./cluster/z.py
cp ./main0.sh ./main.sh
rm ./cluster/add0.py
rm ./cluster/z0.py
rm ./main0.sh
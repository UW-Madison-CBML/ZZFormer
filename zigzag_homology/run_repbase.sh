#!/bin/bash

echo $HOSTNAME

#filename="homo_sapiens_only"

name=$1
filename="${name%.txt}"

echo $filename

python preML.py $filename 

mkdir $filename
mv *.pkl *json $filename
tar -czvf $filename.tar.gz ./$filename

echo DONE

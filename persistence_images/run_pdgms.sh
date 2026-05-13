#!/bin/bash

echo $HOSTNAME

#source /opt/conda/bin/activate giotto-tda
#conda install -c conda-forge dionysus

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
/opt/conda/bin/conda create --prefix ~/my_env --clone giotto-tda
/opt/conda/bin/conda install -c conda-forge -p ~/my_env dionysus
/opt/conda/bin/conda install -c conda-forge -p ~/my_env pandas


name=$1
filename="${name%.tar.gz}"

echo $filename

#python global_PI.py "./"  $name 2
#~/my_env/bin/python global_PI.py "./" $name 4
~/my_env/bin/python get_cc.py "./" $name

mkdir $filename
mv *.pkl *json $filename
tar -czvf ${filename}.tar.gz ./$filename

echo DONE

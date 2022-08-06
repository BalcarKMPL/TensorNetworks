#!/bin/bash
#PBS -N sudden_19.6_7_30
#PBS -l cput=3000:00:00
#PBS -q mp16
#PBS -o shiva_sudden_19.6_7_30.out
#PBS -e shiva_sudden_19.6_7_30.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-7 python -u createFullBH_PEPS.py 4 7 1.0 4.9 0.005 30 10000000 < /dev/null > shortPEPS_py.out & 2> shortPEPS_py.err &

wait
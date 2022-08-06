#!/bin/bash
#PBS -N sudden_19.6_11_25
#PBS -l cput=5000:00:00
#PBS -q mp16
#PBS -o shiva_sudden_19.6_11_25.out
#PBS -e shiva_sudden_19.6_11_25.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-15 python -u createFullBH_PEPS.py 4 11 1.0 4.9 0.02 25 1000 < /dev/null > shortPEPS_11_25.out & 2> shortPEPS_11_25.err &

wait
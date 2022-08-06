#!/bin/bash
#PBS -N sudden_4.5_6_24
#PBS -l cput=5000:00:00
#PBS -q mp16
#PBS -o shiva_sudden_4.5_6_24.out
#PBS -e shiva_sudden_4.5_6_24.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-15 python -u createFullBH_PEPS.py 24 6 1.0 4.5 0.005 100 1000 < /dev/null > shortPEPS_4.5.out & 2> shortPEPS_4.5.err &

wait
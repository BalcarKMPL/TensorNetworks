#!/bin/bash
#PBS -N sudden_19.6_4_7
#PBS -l cput=5000:00:00
#PBS -q mp16
#PBS -o shiva_sudden_19.6_4_7.out
#PBS -e shiva_sudden_19.6_4_7.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-15 python -u createFullBH_PEPS.py 7 4 1 19.6 0.001 500 < /dev/null > shortPEPS_4_7.out & 2> shortPEPS_4_7.err &

wait
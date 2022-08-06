#!/bin/bash
#PBS -N sudden_19.6_7_5000
#PBS -l cput=1500:00:00
#PBS -q mp8
#PBS -o shiva_sudden_19.6_7_5000.out
#PBS -e shiva_sudden_19.6_7_5000.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-7 python -u createFullBH_PEPS.py 28 7 1.0 4.9 0.0001 5000 200 < /dev/null > shortPEPS_7_5000.out & 2> shortPEPS_7_5000.err &

wait
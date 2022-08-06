#!/bin/bash
#PBS -N run_short_PEPS
#PBS -l cput=5000:00:00
#PBS -q mp16
#PBS -o shiva_run_short_PEPS.out
#PBS -e shiva_run_short_PEPS.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-7 python -u createFullBH_PEPS_2.py 5 9 0.01 1 0.001 20 < /dev/null > shortPEPS_0.01_1.out & 2> shortPEPS_0.01_1.err &
nohup taskset -c 8-15 python -u createFullBH_PEPS_2.py 5 9 0.36 1 0.001 20 < /dev/null > shortPEPS_0.36_1.out & 2> shortPEPS_0.36_1.err &

wait

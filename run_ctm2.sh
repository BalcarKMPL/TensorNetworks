#!/bin/bash
#PBS -N ctm2
#PBS -l cput=5000:00:00
#PBS -q mp16
#PBS -o shiva_ctm2.out
#PBS -e shiva_ctm2.err
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-15 python -u ctmrg_of_peps2.py < /dev/null > ctm2.out & 2> ctm2.err &

wait
#!/bin/bash
#PBS -N jmazur_test
#PBS -l cput=50000:00:00
#PBS -q mp64
#PBS -o out_${NAME}
#PBS -e err_${NAME}
#PBS -M mazur.jakub05@gmail.com
#PBS -m abe
cd $HOME
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
nohup taskset -c 0-63 python -u test.py < /dev/null > test.out & 2> test.err &

wait
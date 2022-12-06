import os
import sys
import Ising
import Tools
import numpy as np
from time import time
import BoseHubbard
import NTU_NEW_4 as NTU
import sympy as smp
import CTMRG_better_4 as CTM
import Corelations_working as Corelations
from ncon import ncon

maxiter = 50
NTUprecision = 1e-20
NTUprecisionspeed = 0
ifsvdu = False
iffast = False

dirs = list(os.scandir("./BH-cubicramp"))

def ResetWorkingStatus(d):
    for fileprev in os.scandir(d.path):
        if (not 'PEPS' in fileprev.path):
            continue
        PEPSprev = dict(np.load(fileprev.path))
        if (not 'status' in PEPSprev):
            continue
        if (PEPSprev['status'] != 'working'):
            continue
        np.savez(fileprev.path, GA=PEPSprev['GA'], GB=PEPSprev['GB'],
                 status='waiting', iter=PEPSprev['iter'], dt=PEPSprev['dt'], t=PEPSprev['t'],
                 J_exact=PEPSprev['J_exact'], J_average=PEPSprev['J_average'], U=PEPSprev['U'])


for d in dirs:
    files = os.scandir(d.path)
    print(d.path)
    while ResetWorkingStatus(d): pass
print("\n\nDONE")
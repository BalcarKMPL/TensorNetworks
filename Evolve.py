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

def EvolveOnce(d):
    # for fileprev in os.scandir(d.path):
    #     for filenext in os.scandir(d.path):
    #         if (not 'PEPS' in filenext.path):
    #             continue
    #         if (not 'PEPS' in fileprev.path):
    #             continue
    #         PEPSnext = dict(np.load(filenext.path))
    #         PEPSprev = dict(np.load(fileprev.path))

    for __i in range(1000000):
        if True:
            PEPSprev = dict(np.load(d.path + f'/PEPS_{__i:05d}.npz'))
            PEPSnext = dict(np.load(d.path + f'/PEPS_{__i+1:05d}.npz'))
            if (PEPSprev['iter'] + 1 != PEPSnext['iter']):
                continue
            if (not 'status' in PEPSnext):
                continue
            if (PEPSnext['status'] != 'waiting'):
                continue
            if ('status' in PEPSprev):
                if (PEPSprev['status'] != 'done'):
                    continue

            print("evolving", PEPSprev['iter'], " -> ", PEPSnext['iter'])
            print(list(PEPSprev))

            delta = 0
            PEPS = {'A':PEPSprev['A'], 'B':PEPSprev['B']}
            # GATES = {'GA':PEPSprev['GA'], 'GB':PEPSprev['GB']}
            GATES = BoseHubbard.TrotterGate(PEPSprev['A'].shape[-1],PEPSprev['A'].shape[-1]**2,PEPSprev['dt'].max()/2,PEPSprev['J_average'].max(),1/4,0)

            np.savez(filenext.path, GA=PEPSnext['GA'], GB=PEPSnext['GB'],
                     status='working', iter=PEPSnext['iter'], dt=PEPSnext['dt'], t=PEPSnext['t'], J_exact=PEPSnext['J_exact'], J_average=PEPSnext['J_average'], U=PEPSnext['U'])

            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            # PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            delta = np.max([delta, np.sqrt(np.abs(PEPS['NTUerror']))])

            print("delta/dt = ", delta / PEPSnext['dt'])
            np.savez(filenext.path, status='done', A=PEPS['A'], B=PEPS['B'], GA=PEPSnext['GA'], GB=PEPSnext['GB'], iter=PEPSnext['iter'], dt=PEPSnext['dt'], t=PEPSnext['t'], J_exact=PEPSnext['J_exact'],
                     J_average=PEPSnext['J_average'], U=1, NTUdelta=delta, NTUerror=PEPS['NTUerror'], SVDUerror=PEPS['SVDUerror'])
            print("done", PEPSprev['iter'], " -> ", PEPSnext['iter'])
            return True
    return False

ifchanged = True
while ifchanged:
    ifchanged = False
    for d in dirs:
        files = os.scandir(d.path)
        if (not "3_4_30_0.1" in d.path): continue
        print(d.path)
        while EvolveOnce(d):
            ifchanged = True
print("\n\nDONE")
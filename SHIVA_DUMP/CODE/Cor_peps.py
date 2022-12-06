print("Importing numpy")
import numpy as np

print("Importing CTMRG_better")
import CTMRG_better

print("Importing os")
import os

print("Importing ncon")
from ncon import ncon

print("Importing Corelations")
import Corelations



if __name__ == '__main__':
    env = {}

    indexes = np.arange(0, 1000)

    dir = './BS_19.6_0.01_10_5'

    paths = os.scandir(dir)
    for pepspath in paths:
        if len(pepspath.name) < 10:
            print("Discarding", pepspath.path)
            continue
        if pepspath.name[0:4] != 'RHOA':
            print("Discarding", pepspath.path)
            continue
        index = int(pepspath.name[5:10])
        if not index in indexes:
            print("Discarding", pepspath.path)
            continue

        path = ""
        if not os.path.exists(dir+"/CORR_AA_" + pepspath.name[5:10] + ".npz"):
            print("CORR-ing PEPS nr:", int(pepspath.name[5:10]))
            PEPS = np.load(dir+"/PEPS_"+pepspath.name[5:10]+".npz")
            env = np.load(dir+"/RHOA_"+pepspath.name[5:10]+".npz")

            a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
            ah = a.T

            OL = ah
            OR = a
            nmax = 5

            corA, corB = Corelations.Corelation(env,PEPS,OL,OR,nmax)
            np.savez(dir+"/CORR_AA_"+pepspath.name[5:10]+".npz",corA=corA,corB=corB)



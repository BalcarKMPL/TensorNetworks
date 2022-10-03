print("Importing libs")
import os
import numpy as np
import CTMRG_better
import NTU_NEW_3 as NTU
import sys
import Corelations_working as Corelations

if __name__ == '__main__':
    # i0 = int(float(sys.argv[1]))
    dirs = ['./BHNEW5swapABcorectederror_16_4_1.0_4.9_0.005_90']

    ts = []
    E = []
    E_n = []
    E_nn = []
    E_n_n = []
    E_a_ah = []
    E_ah_a = []

    for dir in dirs:
        print(dir)
        for iter in range(0, 100, 1):
            print("CTMRG-ing PEPS nr:", iter)
            try:
                PEPS = dict(np.load(dir + "/PEPS_{:05d}.npz".format(iter)))
            except:
                continue
            A = PEPS['A']
            B = PEPS['B']

            i = np.eye(A.shape[-1])
            a = np.diag(np.sqrt(np.arange(1, A.shape[-1])), k=1)
            ah = a.T
            n = ah @ a
            nn = n @ n

            ts.append(iter * 0.005)
            print("0 / 6")
            E.append([NTU.__NTUobs1(A, B, i, i), NTU.__NTUobs2(A, B, i, i), NTU.__NTUobs3(A, B, i, i), NTU.__NTUobs4(A, B, i, i)])
            print("1 / 6")
            E_n.append([NTU.__NTUobs1(A, B, n, i), NTU.__NTUobs2(A, B, n, i), NTU.__NTUobs3(A, B, n, i), NTU.__NTUobs4(A, B, n, i)])
            print("2 / 6")
            E_nn.append([NTU.__NTUobs1(A, B, nn, i), NTU.__NTUobs2(A, B, nn, i), NTU.__NTUobs3(A, B, nn, i), NTU.__NTUobs4(A, B, nn, i)])
            print("3 / 6")
            E_n_n.append([NTU.__NTUobs1(A, B, n, n), NTU.__NTUobs2(A, B, n, n), NTU.__NTUobs3(A, B, n, n), NTU.__NTUobs4(A, B, n, n)])
            print("4 / 6")
            E_ah_a.append([NTU.__NTUobs1(A, B, ah, a), NTU.__NTUobs2(A, B, ah, a), NTU.__NTUobs3(A, B, ah, a), NTU.__NTUobs4(A, B, ah, a)])
            print("5 / 6")
            E_a_ah.append([NTU.__NTUobs1(A, B, a, ah), NTU.__NTUobs2(A, B, a, ah), NTU.__NTUobs3(A, B, a, ah), NTU.__NTUobs4(A, B, a, ah)])
            print("6 / 6")

        np.savez(dir + "/NTUOBS.npz", ts=ts, E=E, E_n=E_n, E_nn=E_nn, E_n_n=E_n_n, E_ah_a=E_ah_a, E_a_ah=E_a_ah)

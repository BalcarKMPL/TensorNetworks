print("Importing numpy")
import numpy as np

print("Importing BH")
import BoseHubbard

print("Importing NTU")
import NTU

print("Importing time")
from time import time

print("Libs imported")

if __name__ == '__main__':
    # J=1/19.6
    # J*dt = 0.05
    dt = 0.01 * 19.6 / 2
    t = 25

    d = 3
    D = 20 # tak o rzucone
    r = 14 # wartości singularne do dt^3

    ifsvdu = False

    print("Init params created")
    J = 1
    PEPS = BoseHubbard.TrotterGate(d, r, dt, J=1/19.6, U=1, mu=0)

    A0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    A0[0, 0, 0, 0, 1] = 1
    PEPS['A'] = A0
    PEPS['B'] = A0
    PEPS['time_steps'] = 0
    PEPS['NTUerror'] = 0

    n = int(abs(t) / abs(2 * dt))

    print("Init state created")
    maxiter = 300

    for i in range(n):
        print("#####", i, "/", n, "#####")
        t0=time()
        np.savez('./BS_19.6_0.01_20/PEPS_{:05d}.npz'.format(i), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'])
        PEPS = NTU.NTUstep(PEPS, method='L', ifsvdu=ifsvdu, maxiter=maxiter, ifprint=True)
        print("#####", i, "/", n, " took ", time()-t0, "s #####")

    print("DONE")

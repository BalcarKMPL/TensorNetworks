print("Importing numpy")
import numpy as np
print("Importing Ising")
import Ising
print("Importing CTMRG")
import CTMRG
print("Importing NTU")
import NTU
print("Importing ncon")
from ncon import ncon

if __name__ == '__main__':
    print("Libs imported")
    dt = 0.001
    t = 2
    g0 = 3.04438

    D = 8

    ifsvdu = False

    print("Init params created")
    v0 = np.array([1, 1])
    gx = g0 / 10
    gz = g0 * 0
    J = 1
    PEPS = Ising.TrotterGate(dt, gx=gx, gz=gz, J=J)

    A0 = np.zeros((D, D, D, D, 2), dtype=np.complex128)
    A0[0, 0, 0, 0, 0] = v0[0]
    A0[0, 0, 0, 0, 1] = v0[1]
    PEPS['A'] = A0
    PEPS['B'] = A0
    PEPS['time_steps'] = 0
    PEPS['NTUerror'] = 0

    n = int(abs(t) / abs(dt))

    print("Init state created")
    maxiter = 300

    for i in range(n):
        print("#####",i,"/",n,"#####")
        np.savez('PEPS_{:05d}.npz'.format(i), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'])
        PEPS = NTU.NTUstep(PEPS, method='L', ifsvdu=ifsvdu, maxiter=maxiter, ifprint=True)

    print("DONE")
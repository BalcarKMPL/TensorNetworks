import os.path
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

from scipy.stats import unitary_group

print("Libs imported")

if __name__ == '__main__':
    t000 = time()

    D = int(float(sys.argv[1]))
    d = int(float(sys.argv[2]))
    tQ_exp = float(sys.argv[3])
    dt = float(sys.argv[4])

    r = d**2  # <14 <- wartości singularne do dt^3
    tQ = 0.1 * (2 ** (tQ_exp / 10))
    Jc = 1 / 16.7
    Jc_s, tQ_s, t_s, t0_s, dt_s = smp.symbols('J_c t_Q t t_0 dt', real=True)
    ramp   = Jc_s * (1 + t_s / tQ_s - smp.Rational(4, 27) * t_s ** 3 / tQ_s ** 3)
    linear = Jc_s * (1 + t_s / tQ_s)
    H = smp.Piecewise((0,t_s<0),(1,t_s>0),(1/2,True))
    J_exact_s = ramp * (1 - H) + linear * H
    J_average_s = smp.integrate(J_exact_s / dt_s, (t_s, t0_s, t0_s + dt_s))
    J_exact = smp.lambdify(t_s, J_exact_s.evalf(subs={Jc_s: Jc, tQ_s: tQ}), 'numpy')
    J_average = smp.lambdify(t0_s, J_average_s.evalf(subs={Jc_s: Jc, tQ_s: tQ, dt_s: dt}), 'numpy')

    print("d =", d)
    print("D =", D)
    print("tQ =", tQ)
    print("dt =", dt)

    dir = "./BH-cubicramp_" + str(d) + str("_") + str(D) + str("_") + str(tQ_exp) + str("_") + str(dt)
    if not os.path.isdir(dir): os.mkdir(dir)
    print("Saving in ", dir)

    A0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    A0[0, 0, 0, 0, 1] = 1
    PEPS = {'A': A0, 'B': A0}

    a = np.diag(np.sqrt(np.arange(1, d)), k=1)
    ah = a.T

    maxiter = 50
    NTUprecision = 1e-15
    NTUprecisionspeed = 0
    ifsvdu = False
    iffast = False

    env = {}
    delta = 0
    np.savez(dir + ('/SPECS.npz'), NTUprecision=NTUprecision, NTUprecisionspeed=NTUprecisionspeed, maxiter=maxiter, dt=dt, d=d, D=D, r=r, tQ=tQ)
    np.savez(dir + ('/PEPS_{:05d}.npz'.format(0)), A=PEPS['A'], B=PEPS['B'], NTUerror=0, SVDUerror=0, iter=0, dt=dt, t=0, J_exact=0, J_average=0, U=1, NTUdelta=delta)

    for i,t in enumerate(np.arange(-3/2*tQ, 10*tQ, dt)):
        print("#####", t/tQ, " NTU:")
        GATES = BoseHubbard.TrotterGate(d, r, dt / 2, J_average(t), 1 / 4, 0)

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

        print("delta/dt = ",delta / dt)
        np.savez(dir + ('/PEPS_{:05d}.npz'.format(i+1)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'], SVDUerror=PEPS['SVDUerror'], iter=i+1, dt=dt, t=t, J_exact=J_exact(t), J_average=J_average(t), U=1, NTUdelta=delta)

        if delta >= 0.001 * dt:
            break

    print("Whole sim took:", time() - t000, "s")

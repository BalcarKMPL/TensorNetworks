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

ks = np.arange(0, 40, 10)
Ds = np.arange(4, 8, 2)
d = 3
dt = 0.1
params=[]
for k in ks:
    for D in Ds:
        if (D>14 and k>=70) or (k<70 and D<=14): params.append({'k':k, 'D':D})

masterdirname = "./BH-cubicramp"
def dirname(k, D): return f"{masterdirname}/BH-cubicramp_{d}_{D}_{k}_{dt}"
if not os.path.exists(masterdirname): os.mkdir(masterdirname)

for p in params:
    if not os.path.exists(dirname(p['k'],p['D'])): os.mkdir(dirname(p['k'],p['D']))
    tQ = 0.1 * (2 ** (k / 10))
    N = int(3/2 * (2 ** (k / 10)))+3

    D = p['D']
    k = p['k']
    d = 3
    dt = 0.1
    tQ = 0.1 * (2 ** (k / 10))

    print("D, k =", D, k)

    r = d ** 2  # <14 <- wartości singularne do dt^3
    tQ = 0.1 * (2 ** (k / 10))
    Jc = 1 / 16.7
    Jc_s, tQ_s, t_s, t0_s, dt_s = smp.symbols('J_c t_Q t t_0 dt', real=True)
    ramp = Jc_s * (1 + t_s / tQ_s - smp.Rational(4, 27) * t_s ** 3 / tQ_s ** 3)
    linear = Jc_s * (1 + t_s / tQ_s)
    H = smp.Piecewise((0, t_s < 0), (1, t_s > 0), (1 / 2, True))
    J_exact_s = ramp * (1 - H) + linear * H
    J_average_s = smp.integrate(J_exact_s / dt_s, (t_s, t0_s, t0_s + dt_s))
    J_exact = smp.lambdify(t_s, J_exact_s.evalf(subs={Jc_s: Jc, tQ_s: tQ}), 'numpy')
    J_average = smp.lambdify(t0_s, J_average_s.evalf(subs={Jc_s: Jc, tQ_s: tQ, dt_s: dt}), 'numpy')

    T0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    T0[0, 0, 0, 0, 1] = 1

    maxiter = 50
    NTUprecision = 1e-20
    NTUprecisionspeed = 0
    ifsvdu = False
    iffast = False
    np.savez(dirname(k,D) + ('/SPECS.npz'), NTUprecision=NTUprecision, NTUprecisionspeed=NTUprecisionspeed, maxiter=maxiter, dt=dt, d=d, D=D, r=r, tQ=tQ)

    for i,t in enumerate(np.arange(-3/2*tQ,2*dt,dt)):
        GATES = BoseHubbard.TrotterGate(d,d*d,dt/2,J_average(t),1/4,0)
        if i==0: np.savez(dirname(k,D) + ('/PEPS_{:05d}.npz'.format(i)), A=T0, B=T0, GA=GATES['GA'], GB=GATES['GB'], status='done', iter=i, dt=dt, t=t, J_exact=J_exact(t), J_average=J_average(t), U=1, NTUdelta=0, NTUerror=0, SVDUerror=0)
        else: np.savez(dirname(k,D) + ('/PEPS_{:05d}.npz'.format(i)), GA=GATES['GA'], GB=GATES['GB'], status='waiting', iter=i, dt=dt, t=t, J_exact=J_exact(t), J_average=J_average(t), U=1)

print("DONE")

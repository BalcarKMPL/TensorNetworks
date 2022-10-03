import os.path
import sys
import Ising
import Tools

print("Importing numpy, time")
import numpy as np
from time import time

print("Importing BH, NTU, CTMRG, Corelations_working")
import BoseHubbard
import NTU_NEW_4 as NTU
import CTMRG_better_4 as CTM
import Corelations_working as Corelations
from ncon import ncon

from scipy.stats import unitary_group

print("Libs imported")

if __name__ == '__main__':
    t000 = time()
    
    chi = int(float(sys.argv[1]))
    D = int(float(sys.argv[2]))
    J = float(sys.argv[3])
    U = float(sys.argv[4])
    dt = float(sys.argv[5])
    n = int(float(sys.argv[6]))
    CTMstep = int(float(sys.argv[7]))

    d = 3
    r = d**2  # <14 <- wartości singularne do dt^3

    print("d =", d)
    print("D =", D)
    print("chi =", chi)
    print("dt =", dt)
    print("n =", n)
    print("t =", n * dt)
    print("J =", J)
    print("U =", U)
    print("CTM step =", CTMstep)

    dir = "./BH-NTU4-CTM4_" + str(chi) + str("_") + str(D) + str("_") + str(J) + str("_") + str(U) + str("_") + str(dt) + str(
        "_") + str(n)
    print("Saving in ", dir)
    print("Init params created")
    GATES = BoseHubbard.TrotterGate(d, r, dt / 2, J=J, U=U, mu=0)
    A0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    A0[0, 0, 0, 0, 1] = 1
    PEPS = {'A':A0, 'B':A0}

    if not os.path.isdir(dir): os.mkdir(dir)

    a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
    ah = a.T

    print("Init state created")
    maxiter = 50
    CTMRGprecision = 1e-10
    INVprecision = 1e-10
    NTUprecision = 1e-20
    NTUprecisionspeed = 0

    env = {}
    np.savez(dir + ('/SPECS.npz'), INVprecision=INVprecision, NTUprecision=NTUprecision, CTMRGprecision=CTMRGprecision, maxiter=maxiter, n=n, dt=dt, d=d, D=D, r=r, chi=chi, J=J, U=U)
    np.savez(dir + ('/PEPS_{:05d}.npz'.format(0)), A=PEPS['A'], B=PEPS['B'], NTUerror=0, SVDUerror=0, iter=0, dt=dt, J=J, U=U)

    for i in range(0, n + 1):
        print("#####", i, "/", n, " NTU:")
        t0 = time()
        if i > 0:
            t0 = time()
            ifsvdu = False
            iffast = False

            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            # PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)

            np.savez(dir + ('/PEPS_{:05d}.npz'.format(i)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'], SVDUerror=PEPS['SVDUerror'], iter=i, dt=dt, J=J, U=U)

            GATES['GA'], GATES['GB'] = GATES['GB'], GATES['GA']

        # if i % CTMstep != 0: continue
        print("#####", i, "/", n, " CTMRG:")
        env = CTM.CTMRGstepLtest(PEPS['A'], PEPS['B'], chi, maxiter=maxiter, invprecision=INVprecision, precision=CTMRGprecision,ifrandom=True, tests1=[{'name':'N','A':ah @ a,'B':ah @ a},{'name':'NN','A':ah @ a@ah @ a,'B':ah @ a@ah @ a}], tests2=[{'name':'AhA','A':ah,'B':a},{'name':'AAh','A':a,'B':ah},{'name':'NN','A':ah @ a,'B':ah @ a}])

        np.savez(dir + ('/RHOA_{:05d}.npz'.format(i)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
                 E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'],
                 E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
                 C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
                 C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=i, dt=dt, A=PEPS['A'],
                 B=PEPS['B'], J=J, U=U, names1=env['names1'], names2=env['names2'], vals1=env['vals1'], vals2=env['vals2'], errors1=env['errors1'], errors2=env['errors2'])
        print("#####", i, "/", n, " CORELATIONS:")

        envrot = CTM.__rot1env(env)
        PEPSrot = NTU.__rotinv(PEPS)

        corrWE = Corelations.Corelation(env, PEPS, ah, a, 1)
        corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 1)
        np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        corrWE = Corelations.Corelation(env, PEPS, a, ah, 1)
        corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 1)
        np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 1)
        corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 1)
        np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)

    print("Whole sim took:", time() - t000, "s")

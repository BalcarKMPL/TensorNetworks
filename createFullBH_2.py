import os.path

import Ising

print("Importing numpy, time")
import numpy as np
from time import time

print("Importing BH, NTU, CTMRG, Corelations")
import BoseHubbard
import NTU
import CTMRG_better
import Corelations
from ncon import ncon

from scipy.stats import unitary_group

print("Libs imported")

if __name__ == '__main__':
    # J=1/19.6
    # J*dt = 0.05
    dt = 0.01 * 19.6
    t = 200

    d = 3
    D = 9  # tak o rzucone
    r = 9  # <14 <- wartości singularne do dt^3
    chis = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    # U1, Vh1, U2, Vh2 = unitary_group.rvs(D), unitary_group.rvs(D), unitary_group.rvs(D), unitary_group.rvs(D)
    # d1, d2 = np.exp(np.random.randn(D)), np.exp(np.random.randn(D))
    # D1, D1inv = np.diag(d1), np.diag(1/d1)
    # D2, D2inv = np.diag(d2), np.diag(1/d2)
    # G1 = U1 @ D1 @ Vh1
    # G2 = U2 @ D2 @ Vh2
    # G1inv = Vh1.conj().T @ D1inv @ U1.conj().T
    # G2inv = Vh2.conj().T @ D2inv @ U2.conj().T

    ifsvdu = False

    dir = "./FULL_TEST_SWAP_1"
    print("Init params created")
    J = 1
    PEPS = BoseHubbard.TrotterGate(d, r, dt, J=1 / 19.6, U=0, mu=0)
    # G = np.random.randn(d,d,r)+1j*np.random.randn(d,d,r)
    # PEPS = {'GA': G, 'GB': G}
    A0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    A0[0, 0, 0, 0, 1] = 1
    PEPS['A'] = A0
    PEPS['B'] = A0
    # PEPS['A'] = ncon([G2inv,A0,G1],([-4,1],[-1,2,-3,1,-5],[2,-2]))
    # PEPS['B'] = ncon([G1inv,A0,G2],([-4,1],[-1,2,-3,1,-5],[2,-2]))
    PEPS['time_steps'] = 0
    PEPS['NTUerror'] = 0

    if not os.path.isdir(dir): os.mkdir(dir)

    a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
    ah = a.T

    n = 2

    print("Init state created")
    maxiter = 300
    CTMRGprecision = 1e-12
    INVprecision = 1e-10
    NTUprecision = 1e-15

    env = {}
    print("#####", 0, "/", n+1, "CTMRG:")
    env = CTMRG_better.CTMRGstepL(PEPS['A'], PEPS['B'], chis[0], maxiter=maxiter, env0={}, invprecision=INVprecision,
                                  precision=CTMRGprecision, ifprint=False, ifrandom=False)
    np.savez(dir + ('/PEPS_{:05d}.npz'.format(0)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'])
    np.savez(dir + ('/RHOA_{:05d}.npz'.format(0)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
             E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'],
             E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
             C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
             C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])
    print("#####", 0, "/", n+1, "CORELATIONS:")

    envrot = CTMRG_better.__rot1env(env)
    PEPSrot = NTU.__rotinv(PEPS)

    corrWE = Corelations.Corelation(env, PEPS, ah, a, 5)
    corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 5)
    np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(0)), corA=corrNS['corA'], corB=corrNS['corB'])
    np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(0)), corA=corrWE['corA'], corB=corrWE['corB'])
    corrWE = Corelations.Corelation(env, PEPS, a, ah, 5)
    corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 5)
    np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(0)), corA=corrNS['corA'], corB=corrNS['corB'])
    np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(0)), corA=corrWE['corA'], corB=corrWE['corB'])
    corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 5)
    corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 5)
    np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(0)), corA=corrNS['corA'], corB=corrNS['corB'])
    np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(0)), corA=corrWE['corA'], corB=corrWE['corB'])

    for i in range(n+1):
        print("#####", i + 1, "/", n+1, " NTU:")
        t0 = time()
        # PEPS = NTU.NTUstep(PEPS, method='L', ifsvdu=ifsvdu, maxiter=maxiter, ifprint=True, precision=NTUprecision)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rot(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rot(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rot(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rotinv(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rotinv(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rotinv(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)

        # for iiiiiiter in range(16):
        #     PEPS = NTU.__step(PEPS, ifprint=True, precision=1e-10)
        #     PEPS = NTU.__rot(PEPS)
        #     PEPS = NTU.__rot(PEPS)
        #     PEPS = NTU.__step(PEPS, ifprint=True, precision=1e-10)
        #     PEPS = NTU.__rotinv(PEPS)
        #     PEPS = NTU.__rotinv(PEPS)

        for iiiiiiter in range(2):
            PEPS = NTU.__step(PEPS, ifprint=True, precision=1e-10, ifsvdu=True)
            buff = PEPS['A']
            PEPS['A'] = PEPS['B']
            PEPS['B'] = buff
            # buff = PEPS['GA']
            # PEPS['GA'] = PEPS['GB']
            # PEPS['GB'] = buff

        print("#####", i + 1, "/", n+1, " CTMRG:")
        env = CTMRG_better.CTMRGstepL(PEPS['A'], PEPS['B'], chis[i + 1], maxiter=maxiter, env0={},
                                      invprecision=INVprecision, precision=CTMRGprecision, ifprint=False,
                                      ifrandom=False)
        np.savez(dir + ('/PEPS_{:05d}.npz'.format(i + 1)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'])
        np.savez(dir + ('/RHOA_{:05d}.npz'.format(i + 1)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
                 E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'],
                 E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
                 C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
                 C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])
        print("#####", i + 1, "/", n, " CORELATIONS:")

        envrot = CTMRG_better.__rot1env(env)
        PEPSrot = NTU.__rotinv(PEPS)

        corrWE = Corelations.Corelation(env, PEPS, ah, a, 5)
        corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 5)
        np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(i + 1)), corA=corrNS['corA'], corB=corrNS['corB'])
        np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(i + 1)), corA=corrWE['corA'], corB=corrWE['corB'])
        corrWE = Corelations.Corelation(env, PEPS, a, ah, 5)
        corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 5)
        np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(i + 1)), corA=corrNS['corA'], corB=corrNS['corB'])
        np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(i + 1)), corA=corrWE['corA'], corB=corrWE['corB'])
        corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 5)
        corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 5)
        np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(i + 1)), corA=corrNS['corA'], corB=corrNS['corB'])
        np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(i + 1)), corA=corrWE['corA'], corB=corrWE['corB'])
        print("#####", i + 1, "/", n+1, " took ", time() - t0, "s #####")

    print("DONE")

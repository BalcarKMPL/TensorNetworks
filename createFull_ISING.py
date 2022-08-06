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

print("Libs imported")

if __name__ == '__main__':
    # J=1/19.6
    # J*dt = 0.05
    dt = 0.05

    z=np.array([[1,0],[0,-1]])
    x=np.array([[0,1],[1,0]])

    d = 2
    D = 10  # tak o rzucone
    r = 3  # niezerowe wartości singularne
    chis = [2, 2, 5, 10, 10, 10]

    ifsvdu = False

    print("Init params created")
    PEPS = Ising.TrotterGate(dbeta=1j * dt, J=1, gx=0.5)
    # G = np.random.randn(d,d,r)+1j*np.random.randn(d,d,r)
    # PEPS={'GA':G,'GB':G}
    A0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    A0[0, 0, 0, 0, 0] = 1
    A0[0, 0, 0, 0, 1] = 1
    PEPS['A'] = A0
    PEPS['B'] = A0
    PEPS['time_steps'] = 0
    PEPS['NTUerror'] = 0

    # D=4
    # d=2
    # PEPS = Ising.TrotterGate(0.01*1j, gx=0.3)
    # A0 = np.zeros((D,D,D,D,d))
    # A0[0,0,0,0,0]=1
    # A0[0,0,0,0,1]=1
    # PEPS['A'] = A0
    # PEPS['B'] = A0
    # PEPS['time_steps'] = 0
    # PEPS['NTUerror'] = 0

    dir = "./FULL_ISING_TEST"
    if not os.path.isdir(dir): os.mkdir(dir)

    a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
    ah = a.T

    n = 1

    print("Init state created")
    maxiter = 300
    CTMRGprecision = 1e-12
    INVprecision = 1e-10
    NTUprecision = 1e-15

    env = {}
    print("#####", 0, "/", n, "CTMRG:")
    env = CTMRG_better.CTMRGstepL(PEPS['A'], PEPS['B'], chis[0], maxiter=maxiter, env={}, invprecision=INVprecision,
                                  precision=CTMRGprecision, ifprint=False, ifrandom=False)
    np.savez(dir + ('/PEPS_{:05d}.npz'.format(0)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'])
    np.savez(dir + ('/RHOA_{:05d}.npz'.format(0)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
             E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'],
             E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
             C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
             C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])
    print("#####", 0, "/", n, "CORELATIONS:")

    envrot = CTMRG_better.__rot1env(env)
    PEPSrot = NTU.__rotinv(PEPS)

    corrWE = Corelations.Corelation(env, PEPS, x,x, 5)
    corrNS = Corelations.Corelation(envrot, PEPSrot, x,x, 5)
    np.savez(dir + ('/CORR_XX_NS_{:05d}.npz'.format(0)), corA=corrNS['corA'], corB=corrNS['corB'])
    np.savez(dir + ('/CORR_XX_WE_{:05d}.npz'.format(0)), corA=corrWE['corA'], corB=corrWE['corB'])
    corrWE = Corelations.Corelation(env, PEPS, z,z, 5)
    corrNS = Corelations.Corelation(envrot, PEPSrot, z,z, 5)
    np.savez(dir + ('/CORR_ZZ_NS_{:05d}.npz'.format(0)), corA=corrNS['corA'], corB=corrNS['corB'])
    np.savez(dir + ('/CORR_ZZ_WE_{:05d}.npz'.format(0)), corA=corrWE['corA'], corB=corrWE['corB'])

    for i in range(3):
        print("#####", i + 1, "/", n, " NTU:")
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

        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rot(PEPS)
        # PEPS = NTU.__rot(PEPS)
        # PEPS = NTU.__step(PEPS,ifprint=True,precision=1e-10)
        # PEPS = NTU.__rotinv(PEPS)
        # PEPS = NTU.__rotinv(PEPS)

        for iiiiiiter in range(2*2):
            PEPS = NTU.__step(PEPS, ifprint=True, precision=1e-10)
            buff = PEPS['A']
            PEPS['A'] = PEPS['B']
            PEPS['B'] = buff
            buff = PEPS['GA']
            PEPS['GA'] = PEPS['GB']
            PEPS['GB'] = buff

        print("#####", i + 1, "/", n, " CTMRG:")
        env = CTMRG_better.CTMRGstepL(PEPS['A'], PEPS['B'], chis[i + 1], maxiter=maxiter, env={},
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

        corrWE = Corelations.Corelation(env, PEPS, x,x, 5)
        corrNS = Corelations.Corelation(envrot, PEPSrot, x,x, 5)
        np.savez(dir + ('/CORR_XX_NS_{:05d}.npz'.format(i + 1)), corA=corrNS['corA'], corB=corrNS['corB'])
        np.savez(dir + ('/CORR_XX_WE_{:05d}.npz'.format(i + 1)), corA=corrWE['corA'], corB=corrWE['corB'])
        corrWE = Corelations.Corelation(env, PEPS, z,z, 5)
        corrNS = Corelations.Corelation(envrot, PEPSrot, z,z, 5)
        np.savez(dir + ('/CORR_ZZ_NS_{:05d}.npz'.format(i + 1)), corA=corrNS['corA'], corB=corrNS['corB'])
        np.savez(dir + ('/CORR_ZZ_WE_{:05d}.npz'.format(i + 1)), corA=corrWE['corA'], corB=corrWE['corB'])
        print("#####", i + 1, "/", n, " took ", time() - t0, "s #####")

    print("DONE")

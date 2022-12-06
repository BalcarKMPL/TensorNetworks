import os.path
import sys
import Ising
import Tools

print("Importing numpy, time")
import numpy as np
from time import time

print("Importing BH, NTU, CTMRG, Corelations_working")
import BoseHubbard
import NTU_NEW as NTU
import CTMRG_better
import Corelations_working as Corelations
from ncon import ncon

from scipy.stats import unitary_group

print("Libs imported")

if __name__ == '__main__':
    t000 = time()

    while True:
        for dir in ['./BH/asd/BH_NTUNEW_sud_24_6_1.0_0.03828125_0.0025_141','./BH/asd/BH_NTUNEW_sud_24_6_1.0_0.0765625_0.0025_100','./BH/asd/BH_NTUNEW_sud_24_6_1.0_0.153125_0.0025_70']:
            SPECS = dict(np.load(dir + '/SPECS.npz'))
            chi = SPECS['chi'].max()
            D = SPECS['D'].max()
            J = SPECS['J'].max()
            U = SPECS['U'].max()
            dt = SPECS['dt'].max()
            n = SPECS['n'].max() * 10
            CTMstep = 1

            d = 3
            r = 9  # <14 <- wartości singularne do dt^3

            print("d =", d)
            print("D =", D)
            print("chi =", chi)
            print("dt =", dt)
            print("n =", n)
            print("t =", n * dt)
            print("J =", J)
            print("U =", U)

            print("Saving in ", dir)
            print("Init params created")
            GATES = BoseHubbard.TrotterGate(d, r, dt / 2, J=J, U=U, mu=0)

            imax = 0
            ctmstep = []
            for i in os.scandir(dir):
                if i.name[:4] == 'PEPS':
                    dd = int(i.name[5:10])
                    imax = np.max((imax, dd))
                if i.name[:4] == 'RHOA':
                    ctmstep.append(int(i.name[5:10]))
            s = (np.sort(ctmstep))
            CTMstep = s[1] - s[0]
            print("CTMstep =",CTMstep)

            PEPS = dict(np.load(dir + '/PEPS_{:05d}.npz'.format(imax)))
            print('Evolving', 'PEPS_{:05d}.npz'.format(imax))

            PEPS['GA'] = GATES['GA']
            PEPS['GB'] = GATES['GB']

            a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
            ah = a.T

            print("Init state created")
            maxiter = 100
            CTMRGprecision = 1e-12
            INVprecision = 1e-10
            NTUprecision = 1e-15
            NTUprecisionspeed = 0 * 1e-5
            ifsvdu = False

            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            # PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, ifprint=True, precision=NTUprecision, ifsvdu=ifsvdu, precisionspeed=NTUprecisionspeed, maxiter=maxiter)

            np.savez(dir + ('/PEPS_{:05d}.npz'.format(imax + 1)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'], SVDUerror=PEPS['SVDUerror'], iter=imax + 1, dt=dt, J=J, U=U)

            if (imax + 1) % CTMstep != 0: continue

            print("#####", imax + 1, "/", n, " CTMRG:")
            env = CTMRG_better.CTMRGstepLR(PEPS['A'], PEPS['B'], chi, maxiter=maxiter, env0={}, invprecision=INVprecision, precision=CTMRGprecision, ifprint=False, ifrandom=False)
            np.savez(dir + ('/RHOA_{:05d}.npz'.format(imax + 1)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=imax+1, dt=dt, A=PEPS['A'], B=PEPS['B'], J=J, U=U)

            print("#####", imax + 1, "/", n, " CORELATIONS:")
            envrot = CTMRG_better.__rot1env(env)
            PEPSrot = NTU.__rotinv(PEPS)

            corrWE = Corelations.Corelation(env, PEPS, ah, a, 1)
            corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 1)
            np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(imax + 1)), corA=corrNS['corA'], corB=corrNS['corB'], iter=imax + 1, dt=dt)
            np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(imax + 1)), corA=corrWE['corA'], corB=corrWE['corB'], iter=imax + 1, dt=dt)
            corrWE = Corelations.Corelation(env, PEPS, a, ah, 1)
            corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 1)
            np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(imax + 1)), corA=corrNS['corA'], corB=corrNS['corB'], iter=imax + 1, dt=dt)
            np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(imax + 1)), corA=corrWE['corA'], corB=corrWE['corB'], iter=imax + 1, dt=dt)
            corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 1)
            corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 1)
            np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(imax + 1)), corA=corrNS['corA'], corB=corrNS['corB'], iter=imax + 1, dt=dt)
            np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(imax + 1)), corA=corrWE['corA'], corB=corrWE['corB'], iter=imax + 1, dt=dt)



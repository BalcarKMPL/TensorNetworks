import os.path
import sys
import Ising
import Tools

print("Importing numpy, time")
import numpy as np
from time import time

print("Importing BH, NTU, CTMRG, Corelations_working")
import BoseHubbard
import NTU_NEW_4 as NTU_OBS
import NTU_NEW_4 as NTU
# import SVDU as NTU
import CTMRG_better_4 as CTM
import Corelations_working as Corelations
from ncon import ncon

from scipy.stats import unitary_group

print("Libs imported")


def RandomGaugeGates(G):
    xd = G.shape[-1]
    while True:
        u, s, vh = np.linalg.svd(np.random.randn(xd, xd) + 1j * np.random.randn(xd, xd))
        s = np.random.rand(xd) + 0.0001
        M1 = u @ np.diag(s) @ vh
        M2 = vh.conj().T @ np.diag(1 / s) @ u.conj().T
        if np.linalg.norm(M1 @ M2 - np.eye(xd)) < 1e-10 * np.linalg.norm(M1 @ M2): break
    return G @ M1, G @ M2.T


def RandomGauge(xd=2):
    while True:
        u, s, vh = np.linalg.svd(np.random.randn(xd, xd) + 1j * np.random.randn(xd, xd))
        s = np.random.rand(xd) + 0.0001
        M1 = u @ np.diag(s) @ vh
        M2 = vh.conj().T @ np.diag(1 / s) @ u.conj().T
        if np.linalg.norm(M1 @ M2 - np.eye(xd)) < 1e-10 * np.linalg.norm(M1 @ M2): break
    return M1, M2.T


def RandomOrtho():
    t = np.random.rand() * 2 * np.pi
    return np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])


def RandomGaugePEPS(PEPS):
    A = PEPS['A']
    B = PEPS['B']

    M0A, M2B = RandomGauge(A.shape[0])
    M1A, M3B = RandomGauge(A.shape[1])
    M2A, M0B = RandomGauge(A.shape[2])
    M3A, M1B = RandomGauge(A.shape[3])
    An = ncon([A, M0A, M1A, M2A, M3A], ([1, 2, 3, 4, -5], [1, -1], [2, -2], [3, -3], [4, -4]))
    Bn = ncon([B, M0B, M1B, M2B, M3B], ([1, 2, 3, 4, -5], [1, -1], [2, -2], [3, -3], [4, -4]))

    PEPS['A'] = An
    PEPS['B'] = Bn
    return PEPS


if __name__ == '__main__':
    t000 = time()

    chi = int(float(sys.argv[1]))
    D = int(float(sys.argv[2]))
    J = float(sys.argv[3])
    gx = float(sys.argv[4])
    dt = float(sys.argv[5])
    n = int(float(sys.argv[6]))
    CTMstep = int(float(sys.argv[7]))

    d = 2
    r = 4

    print("d =", d)
    print("D =", D)
    print("chi =", chi)
    print("dt =", dt)
    print("n =", n)
    print("t =", n * dt)
    print("J =", J)
    print("gx =", gx)

    X = np.array([[0, 1], [1, 0]])
    I = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])

    dir = "./ISING-NTU4_" + str(chi) + str("_") + str(D) + str("_") + str(J) + str("_") + str(gx) + str("_") + str(dt) + str("_") + str(n)
    print("Saving in ", dir)
    print("Init params created")

    GATES = Ising.TrotterGate(dt / 2, J, gx, 0)
    A0 = np.zeros((D, D, D, D, 2))
    B0 = np.zeros((D, D, D, D, 2))
    A0[0, 0, 0, 0, 0] = 1
    A0[0, 0, 0, 0, 1] = 1
    B0[0, 0, 0, 0, 0] = 1
    B0[0, 0, 0, 0, 1] = 1
    PEPS = {'A': A0, 'B': B0, 'NTUerror': 0, 'SVDUerror': 0}

    if not os.path.isdir(dir): os.mkdir(dir)

    maxiter = 50
    CTMRGprecision = 1e-12
    INVprecision = 1e-10
    NTUprecision = 1e-15
    NTUprecisionspeed = 0
    ifsvdu = False
    iffast = False

    env = {}
    np.savez(dir + ('/SPECS.npz'), maxiter=maxiter, n=n, dt=dt, d=d, D=D, r=r, chi=chi, J=J, gx=gx, ifsvdu=ifsvdu, INVprecision=INVprecision, NTUprecision=NTUprecision, NTUprecisionspeed=NTUprecisionspeed, CTMRGprecision=CTMRGprecision, iffast=iffast)

    for i in range(0, n + 1):
        t0 = time()
        print("#####", i, "/", n, " NTU:")
        if i > 0:
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rot(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)
            PEPS = NTU.__rotinv(PEPS)
            PEPS = NTU.__step(PEPS, GATES, NTUprecision, ifsvdu, maxiter, True, NTUprecisionspeed, iffast)

        np.savez(dir + ('/PEPS_{:05d}.npz'.format(i)), A=PEPS['A'], B=PEPS['B'], NTUerror=PEPS['NTUerror'], SVDUerror=PEPS['SVDUerror'], iter=i, dt=dt, J=J, gx=gx)

        if i % CTMstep != 0: continue
        print("#####", i, "/", n, " CTMRG:")
        env = CTM.CTMRGstepLtest(PEPS['A'], PEPS['B'], chi, maxiter=maxiter, env0=env, invprecision=INVprecision, precision=CTMRGprecision, ifrandom=True, tests1=[{'name': 'X', 'A': X, 'B': X}, {'name': 'Z', 'A': Z, 'B': Z}], tests2=[{'name': 'ZZ', 'A': Z, 'B': Z}, {'name': 'XX', 'A': X, 'B': X}])
        np.savez(dir + ('/RHOA_{:05d}.npz'.format(i)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=i, dt=dt, A=PEPS['A'], B=PEPS['B'], J=J, gx=gx, names1=env['names1'], names2=env['names2'], vals1=env['vals1'], vals2=env['vals2'], errors1=env['errors1'], errors2=env['errors2'])

        print("#####", i, "/", n, " OBS & CORS:")
        envrot = CTM.__rot1env(env)
        PEPSrot = NTU.__rotinv(PEPS)

        nn = NTU_OBS.__NTUobs1(PEPS['A'], PEPS['B'], I, I)
        np.savez(dir + ('/NTUOBS_{:05}.npz').format(i), XA=NTU_OBS.__NTUobs1(PEPS['A'], PEPS['B'], X, I) / nn, XB=NTU_OBS.__NTUobs1(PEPS['A'], PEPS['B'], I, X) / nn, ZA=NTU_OBS.__NTUobs1(PEPS['A'], PEPS['B'], Z, I) / nn, ZB=NTU_OBS.__NTUobs1(PEPS['A'], PEPS['B'], I, Z) / nn)

        nnA, nnB = np.trace(env['rhoA']), np.trace(env['rhoB'])
        np.savez(dir + ('/OBS_{:05d}.npz').format(i), XA=np.trace(X @ env['rhoA']) / nnA, XB=np.trace(X @ env['rhoB']) / nnB, ZA=np.trace(Z @ env['rhoA']) / nnA, ZB=np.trace(Z @ env['rhoB']) / nnB)

        corrWE = Corelations.Corelation(env, PEPS, Z, Z, 1)
        corrNS = Corelations.Corelation(envrot, PEPSrot, Z, Z, 1)
        np.savez(dir + ('/CORR_ZZ_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        np.savez(dir + ('/CORR_ZZ_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        corrWE = Corelations.Corelation(env, PEPS, X, X, 1)
        corrNS = Corelations.Corelation(envrot, PEPSrot, X, X, 1)
        np.savez(dir + ('/CORR_XX_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        np.savez(dir + ('/CORR_XX_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        print("#####", i, "/", n, " took ", time() - t0, "s #####")

    print("Whole sim took:", time() - t000, "s")

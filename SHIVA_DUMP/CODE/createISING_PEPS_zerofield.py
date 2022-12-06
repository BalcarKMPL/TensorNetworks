import os.path
import sys
import Ising
import Tools

print("Importing numpy, time")
import numpy as np
from time import time

print("Importing BH, NTU, CTMRG, Corelations_working")
import BoseHubbard
import NTU_NEW_3 as NTU
# import SVDU as NTU
import CTMRG_better_3 as CTM
import Corelations_working as Corelations
from ncon import ncon

from scipy.stats import unitary_group

print("Libs imported")


def RandomGaugeGates(G,hermitian=False):
    xd = G.shape[-1]
    while True:
        u,s,vh = np.linalg.svd(np.random.randn(xd,xd)+1j*np.random.randn(xd,xd))
        s = np.random.rand(xd) + 0.0001
        M1 = u @ np.diag(s) @ vh
        M2 = vh.conj().T @ np.diag(1/s) @ u.conj().T
        if np.linalg.norm(M1 @ M2 - np.eye(xd)) < 1e-10 * np.linalg.norm(M1 @ M2): break
    return G@M1, G@M2.T
def RandomOrtho():
    t = np.random.rand() * 2 * np.pi
    return np.array([[np.cos(t),np.sin(t)],[np.sin(t),-np.cos(t)]])

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

    dir = "./ISING_" + str(chi) + str("_") + str(D) + str("_") + str(J) + str("_") + str(gx) + str("_") + str(dt) + str("_") + str(n)
    print("Saving in ", dir)
    print("Init params created")

    if not os.path.isdir(dir): os.mkdir(dir)

    maxiter = 100
    CTMRGprecision = 1e-12
    INVprecision = 1e-10
    NTUprecision = 1e-15
    NTUprecisionspeed = 0

    env = {}
    np.savez(dir + ('/SPECS.npz'), INVprecision=INVprecision, NTUprecision=NTUprecision, CTMRGprecision=CTMRGprecision, maxiter=maxiter, n=n, dt=dt, d=d, D=D, r=r, chi=chi, J=J, gx=gx)

    for i in range(0, n + 1):
        t0 = time()
        PEPS = {}
        db = 2 * i * dt * 1j
        G2 = (np.sqrt(np.sinh(db))) * np.einsum('ij,k->ijk', Z, np.array([0, 1])) + (np.sqrt(np.cosh(db))) * np.einsum('ij,k->ijk', I, np.array([1, 0]))
        G3 = np.zeros((2,2,3), dtype=np.complex128); G3[:,:,:2] = G2
        G4 = np.zeros((2,2,4), dtype=np.complex128); G4[:,:,:2] = G2
        G5 = np.zeros((2,2,5), dtype=np.complex128); G5[:,:,:2] = G2

        G2A, G2B = RandomGaugeGates(G2)
        G3A, G3B = RandomGaugeGates(G3)
        G4A, G4B = RandomGaugeGates(G4)
        G5A, G5B = RandomGaugeGates(G5)

        A = ncon([np.array([1, -1]), G2A, G3A, G4A, G5A], ([1], [1, 2, -1], [2, 3, -2], [3, 4, -3], [4, -5, -4]))
        B = ncon([np.array([1, -1]), G4B, G5B, G2B, G3B], ([1], [1, 2, -1], [2, 3, -2], [3, 4, -3], [4, -5, -4]))
        print(A)

        np.savez(dir + ('/PEPS_{:05d}.npz'.format(i)), A=A, B=B, NTUerror=0, SVDUerror=0, iter=i, dt=dt, J=J, gx=gx)

        print("#####", i, "/", n, " CTMRG:")
        env = CTM.CTMRGstepLtest(A, B, chi, maxiter=maxiter, env0={}, invprecision=INVprecision, precision=CTMRGprecision, ifrandom=True)
        np.savez(dir + ('/RHOA_{:05d}.npz'.format(i)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=i, dt=dt, A=A, B=B, J=J, gx=gx)
        print("#####", i, "/", n, " CORELATIONS:")

        PEPS = {'A': A, 'B': B}
        envrot = CTM.__rot1env(env)
        PEPSrot = NTU.__rotinv(PEPS)

        nn = NTU.__NTUobs1(A,B,I,I)
        np.savez(dir + ('/NTUOBS_{:05}.npz').format(i), XA = NTU.__NTUobs1(A,B,X,I)/nn, XB = NTU.__NTUobs1(A,B,I,X)/nn, ZA = NTU.__NTUobs1(A,B,Z,I)/nn, ZB = NTU.__NTUobs1(A,B,I,Z)/nn)

        np.savez(dir + ('/OBS_{:05d}.npz').format(i), XA=np.trace(X @ env['rhoA'])/np.trace(env['rhoA']), XB=np.trace(X @ env['rhoB'])/np.trace(env['rhoB']), ZA=np.trace(Z @ env['rhoA'])/np.trace(env['rhoA']), ZB=np.trace(Z @ env['rhoB'])/np.trace(env['rhoB']))

        # corrWE = Corelations.Corelation(env, PEPS, Z, Z, 1)
        # corrNS = Corelations.Corelation(envrot, PEPSrot, Z, Z, 1)
        # np.savez(dir + ('/CORR_ZZ_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        # np.savez(dir + ('/CORR_ZZ_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        # corrWE = Corelations.Corelation(env, PEPS, X, X, 1)
        # corrNS = Corelations.Corelation(envrot, PEPSrot, X, X, 1)
        # np.savez(dir + ('/CORR_XX_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        # np.savez(dir + ('/CORR_XX_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        print("#####", i, "/", n, " took ", time() - t0, "s #####")

    print("Whole sim took:", time() - t000, "s")

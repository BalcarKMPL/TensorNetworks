import os.path
import sys
import Ising
import Tools

print("Importing numpy, time")
import numpy as np
from time import time
from scipy.linalg import expm, norm

print("Importing BH, NTU, CTMRG, Corelations_working")
import BoseHubbard
import NTU_NEW_2 as NTU
import CTMRG_better
import Corelations_working as Corelations
from ncon import ncon

from scipy.stats import unitary_group

print("Libs imported")

if __name__ == '__main__':
    t000 = time()
    
    chi = int(float(sys.argv[1]))
    D = int(float(sys.argv[2]))*0+2
    J = float(sys.argv[3])
    U = float(sys.argv[4])
    dt = float(sys.argv[5])
    n = int(float(sys.argv[6]))
    CTMstep = int(float(sys.argv[7]))

    d = 3
    r = 7  # <14 <- wartości singularne do dt^3

    print("d =", d)
    print("D =", D)
    print("chi =", chi)
    print("dt =", dt)
    print("n =", n)
    print("t =", n * dt)
    print("J =", J)
    print("U =", U)
    print("CTM step =", CTMstep)

    dir = "./BHEXACT_" + str(chi) + str("_") + str(D) + str("_") + str(J) + str("_") + str(U) + str("_") + str(dt) + str(
        "_") + str(n)
    print("Saving in ", dir)
    print("Init params created")
    PEPS = BoseHubbard.TrotterGate(d, r, dt, J=J, U=2*U, mu=0)
    A0 = np.zeros((D, D, D, D, d), dtype=np.complex128)
    A0[0, 0, 0, 0, 1] = 1
    PEPS['A'] = A0
    PEPS['B'] = A0

    if not os.path.isdir(dir): os.mkdir(dir)

    a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
    ah = a.T
    nop = ah @ a
    nnop = nop @ nop

    print("Init state created")
    maxiter = 50
    CTMRGprecision = 1e-12
    INVprecision = 1e-10
    NTUprecision = 1e-15
    NTUprecisionspeed = 1e-4

    env = {}
    np.savez(dir + ('/SPECS.npz'), INVprecision=INVprecision, NTUprecision=NTUprecision, CTMRGprecision=CTMRGprecision, maxiter=maxiter, n=n, dt=dt, d=d, D=D, r=r, chi=chi, J=J, U=U)
    np.savez(dir + ('/PEPS_{:05d}.npz'.format(0)), A=PEPS['A'], B=PEPS['B'], NTUerror=0, SVDUerror=0, iter=0, dt=dt, J=J, U=U)

    for i in range(0, n + 1):
        print(Tools.estimated_time(i, n + 1, t000))
        print("#####", i, "/", n, " NTU:")
        t0 = time()
        Dtr = 2
        if i > 0:
            GA = PEPS['GA']
            GB = PEPS['GB']
            t0 = time()
            ifsvdu = False
            iffast = False

            A,B=PEPS['A'],PEPS['B']
            Dtr = A.shape[0] * r
            A = ncon([A,GA,GA,GA,GA],([-1,-3,-5,-7,1],[1,2,-2],[2,3,-4],[3,4,-6],[4,-9,-8])).reshape(Dtr,Dtr,Dtr,Dtr,d)
            B = ncon([B,GB,GB,GB,GB],([-1,-3,-5,-7,1],[1,2,-6],[2,3,-8],[3,4,-2],[4,-9,-4])).reshape(Dtr,Dtr,Dtr,Dtr,d)

            A = A/norm(A)
            B = B/norm(B)
            precprec = 1e-2
            for iiifdsijisd in range(Dtr):
                if norm(A[-1, :, :, :, :]) <= precprec and norm(A[:, -1, :, :, :]) <= precprec and norm(A[:, :, -1, :, :]) <= precprec and norm(A[:, :, :, -1, :]) <= precprec and norm(B[-1, :, :, :, :]) <= precprec and norm(B[:, -1, :, :, :]) <= precprec and norm(B[:, :, -1, :, :]) <= precprec and norm(B[:, :, :, -1, :]) <= precprec:
                    A = A[:-1, :-1, :-1, :-1, :]
                    A = A / norm(A)
                    B = B[:-1, :-1, :-1, :-1, :]
                    B = B / norm(B)
            print(A.shape)

            PEPS['A'] = A
            PEPS['B'] = B


            print("######")
            I = np.eye(A.shape[-1])
            obid = NTU.__NTUobs(A,B,I,I)
            print("1/8\t\tobid =",obid)
            aah = NTU.__NTUobs(A,B,a,ah)/obid
            print("2/8\t\taah =",aah)
            aha = NTU.__NTUobs(A,B,ah,a)/obid
            print("3/8\t\taha =",aha)
            nA = NTU.__NTUobs(A,B,nop,I)/obid
            print("4/8\t\tnA =",nA)
            nnA = NTU.__NTUobs(A,B,nnop,I)/obid
            print("5/8\t\tnnA =",nnA)
            nB = NTU.__NTUobs(A,B,I,nop)/obid
            print("6/8\t\tnB =",nB)
            nnB = NTU.__NTUobs(A,B,I,nnop)/obid
            print("7/8\t\tnnB =",nnB)
            nAnB = NTU.__NTUobs(A,B,nop,nop)/obid
            print("8/8\t\tnAnB =",nAnB)
            print("######")
            # np.savez(dir + ('/PEPS_{:05d}.npz'.format(i)), A=PEPS['A'], B=PEPS['B'], iter=i, dt=dt, J=J, U=U, obid=obid, aah=aah,aha=aha,nA=nA,nB=nB,nnA=nnA,nnB=nnB,nAnB=nAnB)


        # if i % CTMstep != 0: continue
        # print("#####", i, "/", n, " CTMRG:")
        # env = CTMRG_better.CTMRGstepLR(PEPS['A'], PEPS['B'], 5, maxiter=maxiter, env0={},
        #                                invprecision=INVprecision, precision=CTMRGprecision, ifprint=False,
        #                                ifrandom=False)
        # np.savez(dir + ('/RHOA_{:05d}.npz'.format(i)), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
        #          E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'],
        #          E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
        #          C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
        #          C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=i, dt=dt, A=PEPS['A'],
        #          B=PEPS['B'], J=J, U=U)
        # print("#####", i, "/", n, " CORELATIONS:")
        #
        # envrot = CTMRG_better.__rot1env(env)
        # PEPSrot = NTU.__rotinv(PEPS)
        #
        # corrWE = Corelations.Corelation(env, PEPS, ah, a, 0)
        # corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 0)
        # np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        # np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        # corrWE = Corelations.Corelation(env, PEPS, a, ah, 0)
        # corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 0)
        # np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        # np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        # corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 0)
        # corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 0)
        # np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)
        # np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)
        # print("#####", i, "/", n, " took ", time() - t0, "s #####")

    # E_AB_aah_0, E_AB_aha_0, E_BA_aah_0, E_BA_aha_0 = [], [], [], []
    # E_AB_aah_1, E_AB_aha_1, E_BA_aah_1, E_BA_aha_1 = [], [], [], []
    # E_AB_aah_2, E_AB_aha_2, E_BA_aah_2, E_BA_aha_2 = [], [], [], []
    # for i in range(0, n + 1):
    #     print("#####", i, "/", n, " Exact Corelations:")
    #     PEPS = np.load(dir + ('/PEPS_{:05d}.npz'.format(i)))
    #     env = np.load(dir + ('/RHOA_{:05d}.npz').format(i))
    #
    #     # Korelacje dist = 0
    #
    #     E_AB_aah_0.append(np.trace(env['rhoA'] @ (a @ ah)))
    #     E_AB_aha_0.append(np.trace(env['rhoA'] @ (ah @ a)))
    #     E_BA_aah_0.append(np.trace(env['rhoB'] @ (a @ ah)))
    #     E_BA_aha_0.append(np.trace(env['rhoB'] @ (ah @ a)))
    #
    #     print("Dist 0\t\tDONE")
    #
    #     # Korelacje dist = 1
    #
    #     A = PEPS['A']
    #     B = PEPS['B']
    #
    #     cons = ([-1, -2, -3, -4, 1], [-5, 1])
    #     Aa = ncon([A, a], cons)
    #     Bah = ncon([B, ah], cons)
    #     Aah = ncon([A, ah], cons)
    #     Ba = ncon([B, a], cons)
    #     Ac = A.conj()
    #     Bc = B.conj()
    #
    #     cons1 = ([2, 11], [11, 19, 9, 10], [19, 24, 17, 18], [24, 25], [25, 21, 22, 23], [21, 20], [20, 12, 13, 14],
    #              [12, 3, 4, 6], [3, 1], [1, 2, 5, 7], [4, 15, 9, 5, 8], [6, 16, 10, 7, 8], [13, 22, 17, 15, 26],
    #              [14, 23, 18, 16, 26])
    #     E_AB_aah_1.append(ncon(
    #         [env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'], env['E_E_A'], env['C_NE_B'], env['E_N_A'],
    #          env['E_N_B'], env['C_NW_A'], env['E_W_B'], Aa, Ac, Bah, Bc],
    #         cons1) / ncon(
    #         [env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'], env['E_E_A'], env['C_NE_B'], env['E_N_A'],
    #          env['E_N_B'], env['C_NW_A'], env['E_W_B'], A, Ac, B, Bc],
    #         cons1))
    #     E_AB_aha_1.append(ncon(
    #         [env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'], env['E_E_A'], env['C_NE_B'], env['E_N_A'],
    #          env['E_N_B'], env['C_NW_A'], env['E_W_B'], Aah, Ac, Ba, Bc],
    #         cons1) / ncon(
    #         [env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'], env['E_E_A'], env['C_NE_B'], env['E_N_A'],
    #          env['E_N_B'], env['C_NW_A'], env['E_W_B'], A, Ac, B, Bc],
    #         cons1))
    #     E_BA_aah_1.append(ncon(
    #         [env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'], env['E_E_B'], env['C_NE_A'], env['E_N_B'],
    #          env['E_N_A'], env['C_NW_B'], env['E_W_A'], Ba, Bc, Aah, Ac],
    #         cons1) / ncon(
    #         [env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'], env['E_E_B'], env['C_NE_A'], env['E_N_B'],
    #          env['E_N_A'], env['C_NW_B'], env['E_W_A'], B, Bc, A, Ac],
    #         cons1))
    #     E_BA_aha_1.append(ncon(
    #         [env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'], env['E_E_B'], env['C_NE_A'], env['E_N_B'],
    #          env['E_N_A'], env['C_NW_B'], env['E_W_A'], Bah, Bc, Aa, Ac],
    #         cons1) / ncon(
    #         [env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'], env['E_E_B'], env['C_NE_A'], env['E_N_B'],
    #          env['E_N_A'], env['C_NW_B'], env['E_W_A'], B, Bc, A, Ac],
    #         cons1))
    #     print("Dist 1\t\tDONE")
    #
    #     # Korelacje 2
    #
    #     cons = ([-1, -2, -3, -4, 1], [-5, 1])
    #     Aa = ncon([A, a], cons)
    #     Bah = ncon([B, ah], cons)
    #     Aah = ncon([A, ah], cons)
    #     Ba = ncon([B, a], cons)
    #     Ac = A.conj()
    #     Bc = B.conj()
    #
    #     cons2 = ([3, 1], [1, 2, 6, 7], [2, 11], [11, 20, 9, 10], [20, 29, 18, 19], [29, 35, 27, 28], [35, 34],
    #              [34, 31, 32, 33], [31, 30], [30, 21, 22, 23], [21, 12, 13, 14], [12, 3, 4, 5], [4, 15, 9, 6, 8],
    #              [5, 16, 10, 7, 8], [13, 24, 18, 15, 17], [14, 25, 19, 16, 17], [22, 32, 27, 24, 26],
    #              [23, 33, 28, 25, 26])
    #     E_BA_aah_2.append(ncon(
    #         [env['C_NW_B'], env['E_W_A'], env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'],
    #          env['E_E_A'], env['C_NE_B'], env['E_N_A'], env['E_N_B'], env['E_N_A'], Ba, Bc, A, Ac, Bah, Bc],
    #         cons2) / ncon(
    #         [env['C_NW_B'], env['E_W_A'], env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'],
    #          env['E_E_A'], env['C_NE_B'], env['E_N_A'], env['E_N_B'], env['E_N_A'], B, Bc, A, Ac, B, Bc], cons2))
    #     E_AB_aah_2.append(ncon(
    #         [env['C_NW_A'], env['E_W_B'], env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'],
    #          env['E_E_B'], env['C_NE_A'], env['E_N_B'], env['E_N_A'], env['E_N_B'], Aa, Ac, B, Bc, Aah, Ac],
    #         cons2) / ncon(
    #         [env['C_NW_A'], env['E_W_B'], env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'],
    #          env['E_E_B'], env['C_NE_A'], env['E_N_B'], env['E_N_A'], env['E_N_B'], A, Ac, B, Bc, A, Ac], cons2))
    #     E_BA_aha_2.append(ncon(
    #         [env['C_NW_B'], env['E_W_A'], env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'],
    #          env['E_E_A'], env['C_NE_B'], env['E_N_A'], env['E_N_B'], env['E_N_A'], Bah, Bc, A, Ac, Ba, Bc],
    #         cons2) / ncon(
    #         [env['C_NW_B'], env['E_W_A'], env['C_SW_B'], env['E_S_A'], env['E_S_B'], env['E_S_A'], env['C_SE_B'],
    #          env['E_E_A'], env['C_NE_B'], env['E_N_A'], env['E_N_B'], env['E_N_A'], B, Bc, A, Ac, B, Bc], cons2))
    #     E_AB_aha_2.append(ncon(
    #         [env['C_NW_A'], env['E_W_B'], env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'],
    #          env['E_E_B'], env['C_NE_A'], env['E_N_B'], env['E_N_A'], env['E_N_B'], Aah, Ac, B, Bc, Aa, Ac],
    #         cons2) / ncon(
    #         [env['C_NW_A'], env['E_W_B'], env['C_SW_A'], env['E_S_B'], env['E_S_A'], env['E_S_B'], env['C_SE_A'],
    #          env['E_E_B'], env['C_NE_A'], env['E_N_B'], env['E_N_A'], env['E_N_B'], A, Ac, B, Bc, A, Ac], cons2))
    #
    #     print("Dist 2\t\tDONE")
    #     print("#####", i, "/", n, " took ", time() - t0, "s #####")
    #
    # np.savez(dir + ('/EXACT_CORS.npz'), AB_aah_0=E_AB_aah_0, AB_aha_0=E_AB_aha_0,
    #          BA_aah_0=E_BA_aah_0, BA_aha_0=E_BA_aha_0, AB_aah_1=E_AB_aah_1, AB_aha_1=E_AB_aha_1, BA_aah_1=E_BA_aah_1,
    #          BA_aha_1=E_BA_aha_1, AB_aah_2=E_AB_aah_2, AB_aha_2=E_AB_aha_2, BA_aah_2=E_BA_aah_2, BA_aha_2=E_BA_aha_2)

    print("Whole sim took:", time() - t000, "s")
from ncon import ncon
import numpy as np
import CTMRG
from time import time


def Corelation(env, PEPS, OL, OR, nmax):
    print("############################")
    t0 = time()
    # corelationsA, corelationsB = [np.trace(env['rhoA'] @ OL @ OR)], [np.trace(env['rhoB'] @ OL @ OR)]
    corelationsA, corelationsB = [], []

    A = PEPS['A']
    B = PEPS['B']
    AOL = A @ OL.T
    AOR = A @ OR.T
    BOL = B @ OL.T
    BOR = B @ OR.T
    E_E_A = env['E_E_A']
    E_E_B = env['E_E_B']
    E_W_A = env['E_W_A']
    E_W_B = env['E_W_B']
    E_S_A = env['E_S_A']
    E_S_B = env['E_S_B']
    E_N_A = env['E_N_A']
    E_N_B = env['E_N_B']
    C_NW_A = env['C_NW_A']
    C_SE_A = env['C_SE_A']
    C_NE_A = env['C_NE_A']
    C_SW_A = env['C_SW_A']
    C_NE_B = env['C_NE_B']
    C_SW_B = env['C_SW_B']
    C_NW_B = env['C_NW_B']
    C_SE_B = env['C_SE_B']

    def AddLayerLeft(L0, EdgeUp, X, Xc, EdgeDown):
        chi = EdgeUp.shape[0]
        D = X.shape[0]
        d = X.shape[-1]

        L1 = ncon([L0, EdgeUp], ([1, -3, -5, -6], [-1, 1, -2, -4]))

        L2 = ncon([L1.reshape(chi, D * D, D, D, chi), X.swapaxes(3, 2).swapaxes(2, 1).reshape(D * D, D, D, d)],
                  ([-1, 1, -5, -6, -7], [1, -2, -3, -4]))

        L3 = ncon([L2.reshape(chi, D, D, d * D * D, chi),
                   Xc.swapaxes(4, 3).swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0).swapaxes(4, 3).swapaxes(3, 2).reshape(
                       d * D * D, D, D)], ([-1, -2, -4, 1, -6], [1, -3, -5]))

        L4 = ncon([L3.reshape(chi, D, D, D * D * chi), EdgeDown.swapaxes(0, 1).reshape(chi, chi * D * D)],
                  ([-1, -2, -3, 1], [-4, 1]))

        return ncon([L0, EdgeUp, EdgeDown, X, Xc],
                    ([1, 2, 3, 9], [-1, 1, 5, 6], [9, -4, 7, 8], [5, -2, 7, 2, 4], [6, -3, 8, 3, 4]))

        return L4

    def AddLayerRight(R0, EdgeUp, X, Xc, EdgeDown):
        chi = EdgeUp.shape[0]
        D = X.shape[0]
        d = X.shape[-1]

        R1 = ncon([R0, EdgeUp], ([1, -3, -5, -6], [1, -1, -2, -4]))

        R2 = ncon([R1.reshape(chi, D * D, D, D, chi), X.reshape(D * D, D, D, d)],
                  ([-1, 1, -2, -3, -7], [1, -6, -5, -4]))

        R3 = ncon([R2.reshape(chi, D * D * d, D, D, chi), Xc.swapaxes(4, 3).swapaxes(3, 2).reshape(D * D * d, D, D)],
                  ([-1, 1, -2, -5, -6], [1, -4, -3]))

        R4 = ncon([R3.reshape(chi, D, D, D * D * chi), EdgeDown.reshape(chi, D * D * chi)], ([-1, -2, -3, 1], [-4, 1]))

        return ncon([R0, EdgeUp, EdgeDown, X, Xc],
                    ([1, 4, 5, 9], [1, -1, 2, 3], [-4, 9, 7, 8], [2, 4, 7, -2, 6], [3, 5, 8, -3, 6]))

        return R4

    def CreateLeft(Cup, Edge, Cdown):
        return ncon([Cup, Edge, Cdown], ([-1, 1], [1, 2, -2, -3], [2, -4]))

    def CreateRight(Cup, Edge, Cdown):
        return ncon([Cup, Edge, Cdown], ([1, -1], [2, 1, -2, -3], [-4, 2]))

    def FinishLeftRight(Left, Right):
        return Left.flatten() @ Right.flatten()

    LeftOP_A = CreateLeft(C_NW_A, E_W_B, C_SW_A)
    LeftEM_A = LeftOP_A
    LeftOP_B = CreateLeft(C_NW_B, E_W_A, C_SW_B)
    LeftEM_B = LeftOP_B
    RightOP_A = CreateRight(C_NE_A, E_E_B, C_SE_A)
    RightEM_A = RightOP_A
    RightOP_B = CreateRight(C_NE_B, E_E_A, C_SE_B)
    RightEM_B = RightOP_B

    LeftOP_A = AddLayerLeft(LeftOP_A, E_N_B, A @ OL.T @ OR.T, A.conj(), E_S_B)
    LeftOP_B = AddLayerLeft(LeftOP_B, E_N_A, B @ OL.T @ OR.T, B.conj(), E_S_A)
    LeftEM_A = AddLayerLeft(LeftEM_A, E_N_B, A, A.conj(), E_S_B)
    LeftEM_B = AddLayerLeft(LeftEM_B, E_N_A, B, B.conj(), E_S_A)
    corelationsA.append(FinishLeftRight(LeftOP_A, RightOP_A) / FinishLeftRight(LeftEM_A, RightEM_A))
    corelationsB.append(FinishLeftRight(LeftOP_B, RightOP_B) / FinishLeftRight(LeftEM_B, RightEM_B))

    LeftOP_A = CreateLeft(C_NW_A, E_W_B, C_SW_A)
    LeftEM_A = LeftOP_A
    LeftOP_B = CreateLeft(C_NW_B, E_W_A, C_SW_B)
    LeftEM_B = LeftOP_B
    RightOP_A = CreateRight(C_NE_A, E_E_B, C_SE_A)
    RightEM_A = RightOP_A
    RightOP_B = CreateRight(C_NE_B, E_E_A, C_SE_B)
    RightEM_B = RightOP_B

    LeftOP_A = AddLayerLeft(LeftOP_A, E_N_B, A @ OL.T, A.conj(), E_S_B)
    LeftOP_B = AddLayerLeft(LeftOP_B, E_N_A, B @ OL.T, B.conj(), E_S_A)
    LeftEM_A = AddLayerLeft(LeftEM_A, E_N_B, A, A.conj(), E_S_B)
    LeftEM_B = AddLayerLeft(LeftEM_B, E_N_A, B, B.conj(), E_S_A)
    RightOP_A = AddLayerRight(RightOP_A, E_N_B, A @ OR.T, A.conj(), E_S_B)
    RightOP_B = AddLayerRight(RightOP_B, E_N_A, B @ OR.T, B.conj(), E_S_A)
    RightEM_A = AddLayerRight(RightEM_A, E_N_B, A, A.conj(), E_S_B)
    RightEM_B = AddLayerRight(RightEM_B, E_N_A, B, B.conj(), E_S_A)
    corelationsA.append(FinishLeftRight(LeftOP_A, RightOP_B) / FinishLeftRight(LeftEM_A, RightEM_B))
    corelationsB.append(FinishLeftRight(LeftOP_B, RightOP_A) / FinishLeftRight(LeftEM_B, RightEM_A))
    print("\t", 1, "/", 2 * nmax+1)

    for iter in range(nmax):
        LeftOP_A = AddLayerLeft(LeftOP_A, E_N_A, B, B.conj(), E_S_A)
        LeftOP_B = AddLayerLeft(LeftOP_B, E_N_B, A, A.conj(), E_S_B)
        LeftEM_A = AddLayerLeft(LeftEM_A, E_N_A, B, B.conj(), E_S_A)
        LeftEM_B = AddLayerLeft(LeftEM_B, E_N_B, A, A.conj(), E_S_B)
        corelationsA.append(FinishLeftRight(LeftOP_A, RightOP_A) / FinishLeftRight(LeftEM_A, RightEM_A))
        corelationsB.append(FinishLeftRight(LeftOP_B, RightOP_B) / FinishLeftRight(LeftEM_B, RightEM_B))
        print("\t", 2 * iter + 2, "/", 2 * nmax+1)

        LeftOP_A = AddLayerLeft(LeftOP_A, E_N_B, A, A.conj(), E_S_B)
        LeftOP_B = AddLayerLeft(LeftOP_B, E_N_A, B, B.conj(), E_S_A)
        LeftEM_A = AddLayerLeft(LeftEM_A, E_N_B, A, A.conj(), E_S_B)
        LeftEM_B = AddLayerLeft(LeftEM_B, E_N_A, B, B.conj(), E_S_A)
        corelationsA.append(FinishLeftRight(LeftOP_A, RightOP_B) / FinishLeftRight(LeftEM_A, RightEM_B))
        corelationsB.append(FinishLeftRight(LeftOP_B, RightOP_A) / FinishLeftRight(LeftEM_B, RightEM_A))
        print("\t", 2 * iter + 3, "/", 2 * nmax+1)

    return {'corA': np.array(corelationsA), 'corB': np.array(corelationsB)}

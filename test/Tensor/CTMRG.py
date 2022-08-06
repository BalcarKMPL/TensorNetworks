import numpy as np
from ncon import ncon
from scipy.linalg import svd, norm, qr
from Tools import truncate3


def __rot0(A):
    return A


def __rot1(A):
    return A.swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)


def __rot2(A):
    return __rot1(__rot1(A))


def __rot3(A):
    return __rot1(__rot1(__rot1(A)))


def Rho1(A, B):
    # Leading order cost:(D^10)*(d^1)
    tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), B,
               B.conj(), A, A.conj(), B, B.conj(), A, A.conj()]
    connects = [[40, 24, 9, 41, 1], [40, 21, 10, 41, 1], [39, 23, 15, 24, 2], [39, 22, 16, 21, 2],
                [9, 26, 12, 42, 3], [10, 25, 11, 42, 3], [38, 37, 17, 23, 4], [38, 37, 18, 22, 4],
                [12, 32, 43, 44, 5], [11, 31, 43, 44, 5], [13, 30, 33, 32, 6], [14, 29, 33, 31, 6],
                [19, 35, 34, 30, 7], [20, 35, 34, 29, 7], [17, 36, 19, 28, 8], [18, 36, 20, 27, 8],
                [15, 28, 13, 26, -1], [16, 27, 14, 25, -2]]
    con_order = [40, 41, 1, 38, 37, 4, 42, 3, 35, 34, 7, 43, 44, 5, 32, 39, 2, 36, 8, 29, 23, 22, 31,
                 30, 33, 6, 9, 10, 19, 20, 14, 27, 24, 21, 12, 25, 11, 16, 17, 18, 26, 15, 13, 28]
    return ncon(tensors, connects, con_order)


def Rho1E(A, B):
    tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj(), A, A.conj(), B, B.conj(),
               A, A.conj(), B, B.conj(), A, A.conj(), B, B.conj()]
    connects = [[1, 2, 3, 4, 5], [1, 6, 7, 4, 5], [8, 9, 10, 2, 11], [8, 12, 13, 6, 11], [3, 14, 15, 16, 17],
                [7, 18, 19, 16, 17], [20, 21, 22, 9, 23], [20, 57, 24, 12, 23], [15, 25, 26, 27, 28],
                [19, 29, 26, 27, 28], [30, 31, 32, 25, 33], [34, 35, 32, 29, 33], [36, 53, 38, 31, 39],
                [40, 37, 38, 35, 39], [22, 58, 36, 42, -2], [24, 41, 40, 43, -4], [10, 42, 30, 14, -1],
                [13, 43, 34, 18, -3], [44, 56, 45, 21, 46], [44, 56, 47, 57, 46], [45, 55, 48, 58, 49],
                [47, 55, 50, 41, 49], [48, 54, 51, 53, 52], [50, 54, 51, 37, 52]]
    con_order = [1, 4, 5, 16, 17, 8, 11, 44, 56, 46, 54, 51, 52, 20, 23, 45, 50, 32, 33, 26, 27, 28, 36, 40, 15, 19, 2,
                 6, 47, 55, 48, 49, 10, 25, 29, 21, 57, 22, 58, 53, 18, 34, 24, 41, 37, 38, 39, 3, 7, 13, 30, 14, 9, 12,
                 42, 31, 35, 43]
    T = ncon(tensors, connects, con_order)
    return T.reshape(A.shape[-1] * B.shape[-1], A.shape[-1] * B.shape[-1])


def FullCTMRG(A, B, chi, maxiter=1000, env={}, invprecision=1e-10, precision=1e-20, ifprint=False, ifrandom=False):
    D = A.shape[0]
    d = A.shape[-1]

    if ifrandom:
        env = {'E_E_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_E_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_W_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_W_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_S_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_S_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_N_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'E_N_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D),
               'C_NW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi),
               'C_SW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi),
               'C_NE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi),
               'C_SE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi)}

    if len(env) == 0:
        tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
        connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2],
                    [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
        con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
        C_NW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
        tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
        connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2],
                    [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
        con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
        C_SW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
        tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
        connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2],
                    [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
        con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
        C_SE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
        tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
        connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2],
                    [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
        con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
        C_NE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
        tensors = [B, B.conj(), A, A.conj()]
        connects = [[-1, 3, -5, 5, 2], [-3, 4, -7, 5, 2], [-2, -9, -6, 3, 1], [-4, -10, -8, 4, 1]]
        con_order = [5, 2, 4, 3, 1]
        E_W_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [A, A.conj(), B, B.conj()]
        connects = [[-1, 3, -5, 5, 1], [-3, 4, -7, 5, 1], [-2, -9, -6, 3, 2], [-4, -10, -8, 4, 2]]
        con_order = [5, 1, 4, 3, 2]
        E_W_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [A, A.conj(), B, B.conj()]
        connects = [[-6, 3, -2, -9, 1], [-8, 4, -4, -10, 1], [-5, 5, -1, 3, 2], [-7, 5, -3, 4, 2]]
        con_order = [5, 2, 3, 1, 4]
        E_E_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [B, B.conj(), A, A.conj()]
        connects = [[-6, 3, -2, -9, 2], [-8, 4, -4, -10, 2], [-5, 5, -1, 3, 1], [-7, 5, -3, 4, 1]]
        con_order = [5, 1, 3, 2, 4]
        E_E_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [B, B.conj(), A, A.conj()]
        connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
        con_order = [5, 1, 4, 3, 2]
        E_N_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [B, B.conj(), A, A.conj()]
        connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
        con_order = [5, 2, 4, 3, 1]
        E_S_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [A, A.conj(), B, B.conj()]
        connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
        con_order = [5, 2, 3, 1, 4]
        E_S_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
        tensors = [A, A.conj(), B, B.conj()]
        connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
        con_order = [5, 1, 3, 4, 2]
        E_N_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)

        E_E_A = E_E_A[:chi, :chi, :, :] / norm(E_E_A)
        E_E_B = E_E_B[:chi, :chi, :, :] / norm(E_E_B)
        E_W_A = E_W_A[:chi, :chi, :, :] / norm(E_W_A)
        E_W_B = E_W_B[:chi, :chi, :, :] / norm(E_W_B)
        E_S_A = E_S_A[:chi, :chi, :, :] / norm(E_S_A)
        E_S_B = E_S_B[:chi, :chi, :, :] / norm(E_S_B)
        E_N_A = E_N_A[:chi, :chi, :, :] / norm(E_N_A)
        E_N_B = E_N_B[:chi, :chi, :, :] / norm(E_N_B)
        C_NW_A = C_NW_A[:chi, :chi] / norm(C_NW_A)
        C_NE_B = C_NE_B[:chi, :chi] / norm(C_NE_B)
        C_SE_A = C_SE_A[:chi, :chi] / norm(C_SE_A)
        C_SW_B = C_SW_B[:chi, :chi] / norm(C_SW_B)

        # # Xh.shape = (chi, D**4)
        # # X.shape  = (D**4, chi)
        # ih = np.eye(D * D * D * D)[:chi]
        # i = ih.T
        # u1, s1, vh1 = truncate3(C_NW_A, chi)
        # u2, s2, vh2 = truncate3(C_NE_B, chi)
        # u3, s3, vh3 = truncate3(C_SE_A, chi)
        # u4, s4, vh4 = truncate3(C_SW_B, chi)
        # v1 = vh1.T
        # v2 = vh2.T
        # v3 = vh3.T
        # v4 = vh4.T
        # C_NW_A = np.diag(s1)
        # C_NE_B = np.diag(s2)
        # C_SE_A = np.diag(s3)
        # C_SW_B = np.diag(s4)
        #
        # E_E_A = ncon([E_E_A, i, v2], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_E_B = ncon([E_E_B, u3, i], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_W_A = ncon([E_W_A, i, v4], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_W_B = ncon([E_W_B, u1, i], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_S_A = ncon([E_S_A, i, u4], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_S_B = ncon([E_S_B, v3, i], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_N_A = ncon([E_N_A, u2, i], ([1, 2, -3, -4], [1, -1], [2, -2]))
        # E_N_B = ncon([E_N_B, i, v1], ([1, 2, -3, -4], [1, -1], [2, -2]))

        # return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B,
        #         'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A}
    else:
        E_E_A = env['E_E_A']
        E_E_B = env['E_E_B']
        E_W_A = env['E_W_A']
        E_W_B = env['E_W_B']
        E_S_A = env['E_S_A']
        E_S_B = env['E_S_B']
        E_N_A = env['E_N_A']
        E_N_B = env['E_N_B']
        C_NW_A = env['C_NW_A']
        C_NE_B = env['C_NE_B']
        C_SE_A = env['C_SE_A']
        C_SW_B = env['C_SW_B']

    prevsall = 0
    preverror = 1000000000
    error = preverror / 2
    tensors = [C_NW_A, A, E_N_B, E_N_A, C_SW_B, E_W_A, E_W_B, C_SE_A, E_S_B, E_S_A, C_NE_B, E_E_A, E_E_B, A.conj(),
               B, B.conj(), B, B.conj(), A, A.conj()]
    connects = [[3, 4], [21, 12, 13, 22, -1], [2, 3, 21, 25], [1, 2, 20, 26], [6, 7], [5, 6, 23, 27],
                [4, 5, 22, 24], [9, 10], [8, 9, 31, 30], [7, 8, 28, 29], [35, 1], [11, 35, 34, 33],
                [10, 11, 32, 38], [25, 17, 18, 24, -2], [20, 34, 15, 12, 39], [26, 33, 16, 17, 39],
                [13, 14, 28, 23, 37], [18, 19, 29, 27, 37], [15, 32, 31, 14, 36], [16, 38, 30, 19, 36]]
    con_order = [36, 10, 6, 4, 37, 39, 35, 9, 21, 24, 7, 1, 23, 27, 28, 29, 20, 26, 34, 33, 32, 38, 31, 30, 8, 14,
                 19, 11, 15, 16, 5, 17, 18, 3, 22, 25, 2, 12, 13]
    rhoA = ncon(tensors, connects, con_order)
    rhoA = rhoA / np.trace(rhoA)
    rhoA = (rhoA + rhoA.conj().T)/2
    for iter in range(maxiter):
        print("##################################################")
        print(iter)

        tensors = [C_NW_A, A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, A.conj(), B, B.conj()]
        connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1],
                    [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
        con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
        UpperHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
        tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, B.conj(), A, A.conj()]
        connects = [[1, 2], [-1, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -4, 17, 16],
                    [-2, 8, 12, 10, 7], [-3, 9, 13, 11, 7], [-5, 17, 15, 8, 6], [-6, 16, 14, 9, 6]]
        con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
        BottomHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
        tensors = [C_NW_A, A, E_N_B, C_SW_B, E_W_A, E_W_B, E_S_A, A.conj(), B, B.conj()]
        connects = [[1, 2], [10, -2, 8, 11, 6], [-1, 1, 10, 14], [4, 5], [3, 4, 12, 15], [2, 3, 11, 13],
                    [5, -4, 16, 17], [14, -3, 9, 13, 6], [8, -5, 16, 12, 7], [9, -6, 17, 15, 7]]
        con_order = [5, 2, 1, 11, 10, 4, 17, 15, 16, 12, 7, 13, 14, 6, 3, 8, 9]
        LeftHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
        tensors = [E_N_A, C_SE_A, E_S_B, C_NE_B, E_E_A, E_E_B, B, B.conj(), A, A.conj()]
        connects = [[1, -4, 9, 10], [2, 3], [-1, 2, 12, 11], [17, 1], [4, 17, 16, 15], [3, 4, 14, 13],
                    [9, 16, 7, -5, 5], [10, 15, 8, -6, 5], [7, 14, 12, -2, 6], [8, 13, 11, -3, 6]]
        con_order = [1, 3, 17, 9, 16, 2, 14, 12, 13, 11, 6, 10, 15, 5, 4, 7, 8]
        RightHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)

        # -R=
        RU1 = qr(UpperHalf.T, mode='r')
        RU2 = qr(UpperHalf, mode='r')
        RB1 = qr(BottomHalf.T, mode='r')
        RB2 = qr(BottomHalf, mode='r')
        RL1 = qr(LeftHalf.T, mode='r')
        RL2 = qr(LeftHalf, mode='r')
        RL2 = qr(LeftHalf, mode='r')
        RR1 = qr(RightHalf.T, mode='r')
        RR2 = qr(RightHalf, mode='r')

        # =P-
        def create_isometries(X1, X2):
            u, s, vh = truncate3(X1 @ X2.T, chi)
            s = np.where(s < s[0] * invprecision, 0, 1 / np.sqrt(s))
            # if ifprint: print(s)
            s = np.diag(np.sqrt(s))
            p1 = (s @ u.conj().T @ X1).T.reshape(chi, D, D, chi)
            p2 = (s @ vh.conj() @ X2).T.reshape(chi, D, D, chi)
            return p1, p2

        # u, s, vh = svd(RU2 @ RB1.T)
        # s = np.diag(1 / np.sqrt(s))
        # P2 = s @ vh.conj().T @ RB1
        # P3 = s @ u.conj().T @ RU2

        P1, P8 = create_isometries(RR2[0], RL1[0])
        P7, P6 = create_isometries(RB2[0], RU1[0])
        P5, P4 = create_isometries(RL2[0], RR1[0])
        P3, P2 = create_isometries(RU2[0], RB1[0])

        tensors = [C_NW_A, A, E_N_B, E_W_B, A.conj(), P2, P1]
        connects = [[1, 2], [4, 13, 11, 5, 3], [8, 1, 4, 7], [2, 9, 5, 6], [7, 12, 10, 6, 3], [9, 11, 10, -2],
                    [8, 13, 12, -1]]
        con_order = [1, 8, 4, 13, 7, 12, 3, 2, 5, 6, 11, 10, 9]
        C_NW_A_new = ncon(tensors, connects, con_order)
        tensors = [C_SW_B, E_W_A, E_S_A, B, B.conj(), P4, P3]
        connects = [[1, 2], [8, 1, 4, 5], [2, 9, 6, 7], [12, 11, 6, 4, 3], [13, 10, 7, 5, 3], [9, 11, 10, -2],
                    [8, 12, 13, -1]]
        con_order = [2, 1, 6, 4, 7, 5, 3, 8, 12, 13, 9, 11, 10]
        C_SW_B_new = ncon(tensors, connects, con_order)
        tensors = [C_SE_A, E_S_B, E_E_B, A, A.conj(), P5, P6]
        connects = [[1, 2], [8, 1, 5, 4], [2, 9, 7, 6], [10, 7, 5, 12, 3], [11, 6, 4, 13, 3], [8, 12, 13, -1],
                    [9, 10, 11, -2]]
        con_order = [2, 1, 7, 5, 6, 4, 3, 9, 10, 11, 8, 12, 13]
        C_SE_A_new = ncon(tensors, connects, con_order)
        tensors = [E_N_A, C_NE_B, E_E_A, B, B.conj(), P7, P8]
        connects = [[1, 9, 3, 4], [7, 1], [8, 7, 6, 5], [3, 6, 10, 12, 2], [4, 5, 11, 13, 2], [8, 10, 11, -1],
                    [9, 12, 13, -2]]
        con_order = [1, 7, 3, 6, 4, 5, 2, 8, 10, 11, 9, 12, 13]
        C_NE_B_new = ncon(tensors, connects, con_order)
        tensors = [A, E_W_B, A.conj(), P3, P2]
        connects = [[3, -3, 9, 4, 2], [1, 7, 4, 5], [6, -4, 8, 5, 2], [1, 3, 6, -1], [7, 9, 8, -2]]
        con_order = [1, 5, 6, 3, 4, 2, 9, 7, 8]
        E_W_A_new = ncon(tensors, connects, con_order)
        tensors = [E_W_A, B, B.conj(), P2, P3]
        connects = [[7, 1, 3, 4], [8, -3, 5, 3, 2], [9, -4, 6, 4, 2], [1, 5, 6, -2], [7, 8, 9, -1]]
        con_order = [1, 3, 5, 4, 6, 2, 7, 8, 9]
        E_W_B_new = ncon(tensors, connects, con_order)
        tensors = [E_E_B, A, A.conj(), P6, P7]
        connects = [[1, 7, 6, 5], [8, 6, 4, -3, 2], [9, 5, 3, -4, 2], [7, 8, 9, -2], [1, 4, 3, -1]]
        con_order = [7, 5, 9, 6, 8, 2, 1, 3, 4]
        E_E_A_new = ncon(tensors, connects, con_order)
        tensors = [E_E_A, B, B.conj(), P7, P6]
        connects = [[7, 6, 5, 4], [2, 5, 9, -3, 1], [3, 4, 8, -4, 1], [7, 9, 8, -1], [6, 2, 3, -2]]
        con_order = [6, 4, 3, 5, 2, 1, 7, 8, 9]
        E_E_B_new = ncon(tensors, connects, con_order)
        tensors = [A, E_N_B, A.conj(), P1, P8]
        connects = [[3, 8, -3, 4, 2], [7, 1, 3, 6], [6, 9, -4, 5, 2], [7, 8, 9, -1], [1, 4, 5, -2]]
        con_order = [7, 3, 8, 2, 6, 9, 4, 1, 5]
        E_N_A_new = ncon(tensors, connects, con_order)
        tensors = [E_N_A, B, B.conj(), P1, P8]
        connects = [[1, 7, 3, 4], [3, 6, -3, 9, 2], [4, 5, -4, 8, 2], [1, 6, 5, -1], [7, 9, 8, -2]]
        con_order = [1, 3, 6, 4, 5, 2, 7, 9, 8]
        E_N_B_new = ncon(tensors, connects, con_order)
        tensors = [E_S_A, B, B.conj(), P5, P4]
        connects = [[1, 7, 5, 6], [-3, 8, 5, 3, 2], [-4, 9, 6, 4, 2], [1, 3, 4, -1], [7, 8, 9, -2]]
        con_order = [1, 6, 4, 5, 3, 2, 7, 9, 8]
        E_S_B_new = ncon(tensors, connects, con_order)
        tensors = [E_S_B, A, A.conj(), P5, P4]
        connects = [[7, 1, 4, 3], [-3, 6, 4, 9, 2], [-4, 5, 3, 8, 2], [7, 9, 8, -1], [1, 6, 5, -2]]
        con_order = [7, 3, 8, 4, 9, 2, 1, 5, 6]
        E_S_A_new = ncon(tensors, connects, con_order)

        E_E_A = E_E_A_new / norm(E_E_A_new)
        E_E_B = E_E_B_new / norm(E_E_B_new)
        E_W_A = E_W_A_new / norm(E_W_A_new)
        E_W_B = E_W_B_new / norm(E_W_B_new)
        E_S_A = E_S_A_new / norm(E_S_A_new)
        E_S_B = E_S_B_new / norm(E_S_B_new)
        E_N_A = E_N_A_new / norm(E_N_A_new)
        E_N_B = E_N_B_new / norm(E_N_B_new)
        C_NW_A = C_NW_A_new / norm(C_NW_A_new)
        C_NE_B = C_NE_B_new / norm(C_NE_B_new)
        C_SE_A = C_SE_A_new / norm(C_SE_A_new)
        C_SW_B = C_SW_B_new / norm(C_SW_B_new)

        prevrhoA = rhoA
        tensors = [C_NW_A, A, E_N_B, E_N_A, C_SW_B, E_W_A, E_W_B, C_SE_A, E_S_B, E_S_A, C_NE_B, E_E_A, E_E_B, A.conj(),
                   B, B.conj(), B, B.conj(), A, A.conj()]
        connects = [[3, 4], [21, 12, 13, 22, -1], [2, 3, 21, 25], [1, 2, 20, 26], [6, 7], [5, 6, 23, 27],
                    [4, 5, 22, 24], [9, 10], [8, 9, 31, 30], [7, 8, 28, 29], [35, 1], [11, 35, 34, 33],
                    [10, 11, 32, 38], [25, 17, 18, 24, -2], [20, 34, 15, 12, 39], [26, 33, 16, 17, 39],
                    [13, 14, 28, 23, 37], [18, 19, 29, 27, 37], [15, 32, 31, 14, 36], [16, 38, 30, 19, 36]]
        con_order = [36, 10, 6, 4, 37, 39, 35, 9, 21, 24, 7, 1, 23, 27, 28, 29, 20, 26, 34, 33, 32, 38, 31, 30, 8, 14,
                     19, 11, 15, 16, 5, 17, 18, 3, 22, 25, 2, 12, 13]
        rhoA = ncon(tensors, connects, con_order)
        rhoA = rhoA / np.trace(rhoA)
        rhoA = (rhoA + rhoA.conj().T)/2
        print(rhoA)
        preverror = error
        error = norm(rhoA - prevrhoA)
        print(error)

        if error < precision:
            break

        # print("Calculating error")
        # CTMRG_A(A,B,1,env={'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B,
        #     'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A})
        # if error <= preverror:
        # break
        # pass

    print(error)
    return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B,
            'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A,
            'rhoA': rhoA}


def CTMRG_4(A, B, OA1, OB1, OA2, OB2, env):
    E_E_A = env['E_E_A']
    E_E_B = env['E_E_B']
    E_W_A = env['E_W_A']
    E_W_B = env['E_W_B']
    E_S_A = env['E_S_A']
    E_S_B = env['E_S_B']
    E_N_A = env['E_N_A']
    E_N_B = env['E_N_B']
    C_NW_A = env['C_NW_A']
    C_NE_B = env['C_NE_B']
    C_SE_A = env['C_SE_A']
    C_SW_B = env['C_SW_B']

    tensors = [C_NW_A, A, E_N_B, E_N_A, C_SW_B, E_W_A, E_W_B, C_SE_A, E_S_B, E_S_A, C_NE_B, E_E_A, E_E_B, A.conj(), B,
               B.conj(), B, B.conj(), A, A.conj(), OA1, OB2, OA2, OB1]
    connects = [[3, 4], [21, 12, 13, 22, 43], [2, 3, 21, 25], [1, 2, 20, 26], [6, 7], [5, 6, 23, 27], [4, 5, 22, 24],
                [9, 10], [8, 9, 31, 30], [7, 8, 28, 29], [36, 1], [11, 36, 35, 34], [10, 11, 33, 32],
                [25, 17, 18, 24, 44], [20, 35, 15, 12, 37], [26, 34, 16, 17, 38], [13, 14, 28, 23, 41],
                [18, 19, 29, 27, 42], [15, 33, 31, 14, 39], [16, 32, 30, 19, 40], [43, 44], [41, 42], [39, 40],
                [37, 38]]
    con_order = [42, 44, 17, 3, 1, 37, 7, 40, 9, 19, 35, 31, 28, 2, 6, 23, 10, 33, 30, 39, 32, 20, 36, 8, 29, 14, 41,
                 27, 5, 25, 26, 34, 38, 4, 11, 15, 18, 24, 16, 21, 12, 43, 13, 22]
    return ncon(tensors, connects, con_order)


def CTMRG_AB(A, B, OA, OB, env):
    I = np.eye(A.shape[-1])
    return CTMRG_4(A, B, OA, OB, I, I, env)


def CTMRG_AA(A, B, OA1, OA2, env):
    I = np.eye(A.shape[-1])
    return CTMRG_4(A, B, OA1, I, OA2, I, env)


def CTMRG_A(A, B, OA, env):
    I = np.eye(A.shape[-1])
    return CTMRG_4(A, B, OA, I, I, I, env)


def CTMRG_B(A, B, OB, env):
    I = np.eye(A.shape[-1])
    return CTMRG_4(A, B, I, OB, I, I, env)


def CTMRG_1(A, B, env):
    I = np.eye(A.shape[-1])
    return CTMRG_4(A, B, I, I, I, I, env)

import numpy as np
from ncon import ncon
from scipy.linalg import svd, norm, qr
from Tools import truncate3
from time import time
import copy


def __rot1env(env):
    # obrót odwrotny tożsamy z rot1
    env2 = {}
    for key in env.keys():
        env2[key] = env[key]

    env2['C_NW_A'] = np.copy(env['C_NE_A'])
    env2['C_NE_A'] = np.copy(env['C_SE_A'])
    env2['C_SE_A'] = np.copy(env['C_SW_A'])
    env2['C_SW_A'] = np.copy(env['C_NW_A'])

    env2['C_NW_B'] = np.copy(env['C_NE_B'])
    env2['C_NE_B'] = np.copy(env['C_SE_B'])
    env2['C_SE_B'] = np.copy(env['C_SW_B'])
    env2['C_SW_B'] = np.copy(env['C_NW_B'])

    env2['E_E_A'] = np.copy(env['E_S_A'])
    env2['E_S_A'] = np.copy(env['E_W_A'])
    env2['E_W_A'] = np.copy(env['E_N_A'])
    env2['E_N_A'] = np.copy(env['E_E_A'])

    env2['E_E_B'] = np.copy(env['E_S_B'])
    env2['E_S_B'] = np.copy(env['E_W_B'])
    env2['E_W_B'] = np.copy(env['E_N_B'])
    env2['E_N_B'] = np.copy(env['E_E_B'])

    return env2


def __rot3env(env):
    # obrót odwrotny tożsamy z rot3
    env2 = {}
    for key in env.keys():
        env2[key] = env[key]

    env2['C_SW_A'] = np.copy(env['C_SE_A'])
    env2['C_SE_A'] = np.copy(env['C_NE_A'])
    env2['C_NE_A'] = np.copy(env['C_NW_A'])
    env2['C_NW_A'] = np.copy(env['C_SW_A'])

    env2['C_SW_B'] = np.copy(env['C_SE_B'])
    env2['C_SE_B'] = np.copy(env['C_NE_B'])
    env2['C_NE_B'] = np.copy(env['C_NW_B'])
    env2['C_NW_B'] = np.copy(env['C_SW_B'])

    env2['E_N_A'] = np.copy(env['E_W_A'])
    env2['E_W_A'] = np.copy(env['E_S_A'])
    env2['E_S_A'] = np.copy(env['E_E_A'])
    env2['E_E_A'] = np.copy(env['E_N_A'])

    env2['E_N_B'] = np.copy(env['E_W_B'])
    env2['E_W_B'] = np.copy(env['E_S_B'])
    env2['E_S_B'] = np.copy(env['E_E_B'])
    env2['E_E_B'] = np.copy(env['E_N_B'])

    return env2


def __rot0(A):
    return np.copy(A)


def __rot1(A):
    return np.copy(A).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)


def __rot2(A):
    return __rot1(__rot1(A))


def __rot3(A):
    return __rot1(__rot1(__rot1(A)))


def Rho1(A, B):
    # Leading order cost:(D^10)*(d^1)
    tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj(), A, A.conj(), B, B.conj(), A, A.conj()]
    connects = [[40, 24, 9, 41, 1], [40, 21, 10, 41, 1], [39, 23, 15, 24, 2], [39, 22, 16, 21, 2], [9, 26, 12, 42, 3], [10, 25, 11, 42, 3], [38, 37, 17, 23, 4], [38, 37, 18, 22, 4], [12, 32, 43, 44, 5], [11, 31, 43, 44, 5], [13, 30, 33, 32, 6], [14, 29, 33, 31, 6], [19, 35, 34, 30, 7], [20, 35, 34, 29, 7], [17, 36, 19, 28, 8], [18, 36, 20, 27, 8], [15, 28, 13, 26, -1], [16, 27, 14, 25, -2]]
    con_order = [40, 41, 1, 38, 37, 4, 42, 3, 35, 34, 7, 43, 44, 5, 32, 39, 2, 36, 8, 29, 23, 22, 31, 30, 33, 6, 9, 10, 19, 20, 14, 27, 24, 21, 12, 25, 11, 16, 17, 18, 26, 15, 13, 28]
    return ncon(tensors, connects, con_order)


def Rho1E(A, B):
    tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj(), A, A.conj(), B, B.conj(), A, A.conj(), B, B.conj(), A, A.conj(), B, B.conj()]
    connects = [[1, 2, 3, 4, 5], [1, 6, 7, 4, 5], [8, 9, 10, 2, 11], [8, 12, 13, 6, 11], [3, 14, 15, 16, 17], [7, 18, 19, 16, 17], [20, 21, 22, 9, 23], [20, 57, 24, 12, 23], [15, 25, 26, 27, 28], [19, 29, 26, 27, 28], [30, 31, 32, 25, 33], [34, 35, 32, 29, 33], [36, 53, 38, 31, 39], [40, 37, 38, 35, 39], [22, 58, 36, 42, -2], [24, 41, 40, 43, -4], [10, 42, 30, 14, -1], [13, 43, 34, 18, -3], [44, 56, 45, 21, 46], [44, 56, 47, 57, 46], [45, 55, 48, 58, 49], [47, 55, 50, 41, 49], [48, 54, 51, 53, 52], [50, 54, 51, 37, 52]]
    con_order = [1, 4, 5, 16, 17, 8, 11, 44, 56, 46, 54, 51, 52, 20, 23, 45, 50, 32, 33, 26, 27, 28, 36, 40, 15, 19, 2, 6, 47, 55, 48, 49, 10, 25, 29, 21, 57, 22, 58, 53, 18, 34, 24, 41, 37, 38, 39, 3, 7, 13, 30, 14, 9, 12, 42, 31, 35, 43]
    T = ncon(tensors, connects, con_order)
    return T.reshape(A.shape[-1] * B.shape[-1], A.shape[-1] * B.shape[-1])

#
# def CTMRGstepLR(A, B, chi, maxiter=1000, env0={}, invprecision=1e-10, precision=1e-20, ifprint=False, ifrandom=False):
#     env = __CTMRG_chi_change(env0, chi)
#     if A.shape[0] == 1:
#         return {'C_SW_A': np.array([1]).reshape(1, 1), 'C_SW_B': np.array([1]).reshape(1, 1), 'C_SE_A': np.array([1]).reshape(1, 1), 'C_SE_B': np.array([1]).reshape(1, 1), 'C_NW_A': np.array([1]).reshape(1, 1), 'C_NW_B': np.array([1]).reshape(1, 1), 'C_NE_A': np.array([1]).reshape(1, 1), 'C_NE_B': np.array([1]).reshape(1, 1), 'E_N_A': np.array([1]).reshape(1, 1, 1, 1), 'E_N_B': np.array([1]).reshape(1, 1, 1, 1), 'E_S_A': np.array([1]).reshape(1, 1, 1, 1), 'E_S_B': np.array([1]).reshape(1, 1, 1, 1), 'E_E_A': np.array([1]).reshape(1, 1, 1, 1), 'E_E_B': np.array([1]).reshape(1, 1, 1, 1), 'E_W_A': np.array([1]).reshape(1, 1, 1, 1), 'E_W_B': np.array([1]).reshape(1, 1, 1, 1), 'rhoA': ncon([A, A.conj()], ([1, 2, 3, 4, -1], [1, 2, 3, 4, -2])), 'rhoB': ncon([B, B.conj()], ([1, 2, 3, 4, -1], [1, 2, 3, 4, -2])), 'error': 0}
#
#     errora, errorb = np.inf, np.inf
#     for iter in range(maxiter):
#         if "rhoA" in env and "rhoB" in env:
#             prevrhoA = env["rhoA"]
#             prevrhoB = env["rhoB"]
#         else:
#             prevrhoA = 0
#             prevrhoB = 0
#             errora, errorb = np.inf, np.inf
#
#         print("\t", iter, "  ", 1)
#         env = __CTMRT_left_right(A, B, chi, env, invprecision, ifprint=ifprint, ifrandom=ifrandom)
#         env = __rot1env(__rot1env(__rot1env(env)))  # __rot3env(env)
#         A = __rot1(__rot1(__rot1(A)))  # __rot3(A)
#         B = __rot1(__rot1(__rot1(B)))  # __rot3(B)
#         print("\t", iter, "  ", 2)
#         env = __CTMRT_left_right(A, B, chi, env, invprecision, ifprint=ifprint, ifrandom=ifrandom)
#         env = __rot1env(env)
#         A = __rot1(A)
#         B = __rot1(B)
#
#         preverrora, preverrorb = errora, errorb
#         errora, errorb = norm(env["rhoA"] - prevrhoA), norm(env["rhoB"] - prevrhoB)
#
#         print("Errorz =", errora, "\t", errorb, "\t\t >", precision)
#         if (errora < precision and errorb < precision) or (preverrora < errora and preverrorb < errorb):  # ((np.abs(errorb-preverrorb)/errorb < 1e-4 or np.abs(errora-preverrora)/errora < 1e-4) and False):
#             env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
#             break
#
#     env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
#     return env
#
#
# def CTMRGstepL(A, B, chi, Ac=None, Bc=None, maxiter=1000, env0={}, invprecision=1e-10, precision=1e-20, ifprint=False, ifrandom=False):
#     if np.any(Ac == None): Ac = A.conj()
#     if np.any(Bc == None): Bc = B.conj()
#     D = A.shape[0]
#
#     env = __CTMRG_chi_change(env0, chi)
#     if ifrandom:
#         env = {'E_E_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_E_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_W_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_W_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_S_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_S_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_N_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_N_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'C_NW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi),
#                'C_NW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi)}
#     if len(env) == 0:
#         if chi <= D ** 2:
#             C_NW_A = ncon([A, Ac], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(D * D, D * D)
#             C_NW_B = ncon([B, Bc], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(D * D, D * D)
#             C_NE_A = ncon([A, Ac], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(D * D, D * D)
#             C_NE_B = ncon([B, Bc], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(D * D, D * D)
#             C_SW_A = ncon([A, Ac], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(D * D, D * D)
#             C_SW_B = ncon([B, Bc], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(D * D, D * D)
#             C_SE_A = ncon([A, Ac], ([-1, 1, 2, -3, 3], [-2, 1, 2, -4, 3])).reshape(D * D, D * D)
#             C_SE_B = ncon([B, Bc], ([-1, 1, 2, -3, 3], [-2, 1, 2, -4, 3])).reshape(D * D, D * D)
#
#             E_E_A = ncon([A, Ac], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(D * D, D * D, D, D)
#             E_E_B = ncon([B, Bc], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(D * D, D * D, D, D)
#             E_N_A = ncon([A, Ac], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(D * D, D * D, D, D)
#             E_N_B = ncon([B, Bc], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(D * D, D * D, D, D)
#             E_W_A = ncon([A, Ac], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(D * D, D * D, D, D)
#             E_W_B = ncon([B, Bc], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(D * D, D * D, D, D)
#             E_S_A = ncon([A, Ac], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(D * D, D * D, D, D)
#             E_S_B = ncon([B, Bc], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(D * D, D * D, D, D)
#         if chi <= D ** 4 and chi > D ** 2:
#             tensors = [A, Ac, B, Bc, B, Bc, A, Ac]
#             connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2], [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
#             con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
#             C_NW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, Ac, B, Bc, B, Bc, A, Ac]
#             connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2], [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
#             con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
#             C_SW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, Ac, B, Bc, B, Bc, A, Ac]
#             connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2], [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
#             con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
#             C_SE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, Ac, B, Bc, B, Bc, A, Ac]
#             connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2], [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
#             con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
#             C_NE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, Bc, A, Ac, A, Ac, B, Bc]
#             connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2], [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
#             con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
#             C_NW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, Bc, A, Ac, A, Ac, B, Bc]
#             connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2], [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
#             con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
#             C_SW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, Bc, A, Ac, A, Ac, B, Bc]
#             connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2], [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
#             con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
#             C_SE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, Bc, A, Ac, A, Ac, B, Bc]
#             connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2], [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
#             con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
#             C_NE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, Bc, A, Ac]
#             connects = [[-1, 3, -5, 5, 2], [-3, 4, -7, 5, 2], [-2, -9, -6, 3, 1], [-4, -10, -8, 4, 1]]
#             con_order = [5, 2, 4, 3, 1]
#             E_W_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, Ac, B, Bc]
#             connects = [[-1, 3, -5, 5, 1], [-3, 4, -7, 5, 1], [-2, -9, -6, 3, 2], [-4, -10, -8, 4, 2]]
#             con_order = [5, 1, 4, 3, 2]
#             E_W_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, Ac, B, Bc]
#             connects = [[-6, 3, -2, -9, 1], [-8, 4, -4, -10, 1], [-5, 5, -1, 3, 2], [-7, 5, -3, 4, 2]]
#             con_order = [5, 2, 3, 1, 4]
#             E_E_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, Bc, A, Ac]
#             connects = [[-6, 3, -2, -9, 2], [-8, 4, -4, -10, 2], [-5, 5, -1, 3, 1], [-7, 5, -3, 4, 1]]
#             con_order = [5, 1, 3, 2, 4]
#             E_E_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, Bc, A, Ac]
#             connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
#             con_order = [5, 1, 4, 3, 2]
#             E_N_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, Bc, A, Ac]
#             connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
#             con_order = [5, 2, 4, 3, 1]
#             E_S_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, Ac, B, Bc]
#             connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
#             con_order = [5, 2, 3, 1, 4]
#             E_S_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, Ac, B, Bc]
#             connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
#             con_order = [5, 1, 3, 4, 2]
#             E_N_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#         env['E_E_A'] = E_E_A[:chi, :chi, :, :] / norm(E_E_A)
#         env['E_E_B'] = E_E_B[:chi, :chi, :, :] / norm(E_E_B)
#         env['E_W_A'] = E_W_A[:chi, :chi, :, :] / norm(E_W_A)
#         env['E_W_B'] = E_W_B[:chi, :chi, :, :] / norm(E_W_B)
#         env['E_S_A'] = E_S_A[:chi, :chi, :, :] / norm(E_S_A)
#         env['E_S_B'] = E_S_B[:chi, :chi, :, :] / norm(E_S_B)
#         env['E_N_A'] = E_N_A[:chi, :chi, :, :] / norm(E_N_A)
#         env['E_N_B'] = E_N_B[:chi, :chi, :, :] / norm(E_N_B)
#         env['C_NW_A'] = C_NW_A[:chi, :chi] / norm(C_NW_A)
#         env['C_NE_B'] = C_NE_B[:chi, :chi] / norm(C_NE_B)
#         env['C_SE_A'] = C_SE_A[:chi, :chi] / norm(C_SE_A)
#         env['C_SW_B'] = C_SW_B[:chi, :chi] / norm(C_SW_B)
#         env['C_NW_B'] = C_NW_B[:chi, :chi] / norm(C_NW_B)
#         env['C_NE_A'] = C_NE_A[:chi, :chi] / norm(C_NE_A)
#         env['C_SE_B'] = C_SE_B[:chi, :chi] / norm(C_SE_B)
#         env['C_SW_A'] = C_SW_A[:chi, :chi] / norm(C_SW_A)
#
#     if A.shape[0] == 1:
#         return {'C_SW_A': np.array([1]).reshape(1, 1), 'C_SW_B': np.array([1]).reshape(1, 1), 'C_SE_A': np.array([1]).reshape(1, 1), 'C_SE_B': np.array([1]).reshape(1, 1), 'C_NW_A': np.array([1]).reshape(1, 1), 'C_NW_B': np.array([1]).reshape(1, 1), 'C_NE_A': np.array([1]).reshape(1, 1), 'C_NE_B': np.array([1]).reshape(1, 1), 'E_N_A': np.array([1]).reshape(1, 1, 1, 1), 'E_N_B': np.array([1]).reshape(1, 1, 1, 1), 'E_S_A': np.array([1]).reshape(1, 1, 1, 1), 'E_S_B': np.array([1]).reshape(1, 1, 1, 1), 'E_E_A': np.array([1]).reshape(1, 1, 1, 1), 'E_E_B': np.array([1]).reshape(1, 1, 1, 1), 'E_W_A': np.array([1]).reshape(1, 1, 1, 1), 'E_W_B': np.array([1]).reshape(1, 1, 1, 1), 'rhoA': ncon([A, Ac], ([1, 2, 3, 4, -1], [1, 2, 3, 4, -2])), 'rhoB': ncon([B, Bc], ([1, 2, 3, 4, -1], [1, 2, 3, 4, -2])), 'error': 0}
#
#     errora, errorb = np.inf, np.inf
#     for iter in range(maxiter):
#         if "rhoA" in env and "rhoB" in env:
#             prevrhoA = env["rhoA"]
#             prevrhoB = env["rhoB"]
#         else:
#             prevrhoA = 0
#             prevrhoB = 0
#             errora, errorb = np.inf, np.inf
#
#         for itermin in range(4):
#             print("\t", iter, "  ", itermin)
#             env = __CTMRT_left_test(A, B, chi, Ac, Bc, env, invprecision, ifprint=ifprint)
#             env = __rot1env(env)
#             A = __rot1(A)
#             B = __rot1(B)
#
#         preverrora, preverrorb = errora, errorb
#         errora, errorb = norm(env["rhoA"] - prevrhoA), norm(env["rhoB"] - prevrhoB)
#
#         print("Errorz =", errora, "\t", errorb, "\t\t >", precision)
#         if (errora < precision and errorb < precision) or (preverrora < errora and preverrorb < errorb):
#             env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
#             break
#
#     env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
#     return env


def CTMRGstepLtest(A, B, chi, Ac=None, Bc=None, maxiter=1000, env0={}, invprecision=1e-10, precision=1e-20, ifprint=False, ifrandom=False):
    if np.any(Ac == None): Ac = A.conj()
    if np.any(Bc == None): Bc = B.conj()
    D = A.shape[0]

    env = __CTMRG_chi_change(env0, chi)
    if ifrandom:
        env = {'E_E_A': np.random.randn(chi, chi, B.shape[1], Bc.shape[1]) + 1j * np.random.randn(chi, chi, B.shape[1], Bc.shape[1]), 'E_E_B': np.random.randn(chi, chi, A.shape[1], Ac.shape[1]) + 1j * np.random.randn(chi, chi, A.shape[1], Ac.shape[1]), 'E_W_A': np.random.randn(chi, chi, B.shape[3], Bc.shape[3]) + 1j * np.random.randn(chi, chi, B.shape[3], Bc.shape[3]), 'E_W_B': np.random.randn(chi, chi, A.shape[3], Ac.shape[3]) + 1j * np.random.randn(chi, chi, A.shape[3], Ac.shape[3]), 'E_S_A': np.random.randn(chi, chi, B.shape[2], Bc.shape[2]) + 1j * np.random.randn(chi, chi, B.shape[2], Bc.shape[2]), 'E_S_B': np.random.randn(chi, chi, A.shape[2], Ac.shape[2]) + 1j * np.random.randn(chi, chi, A.shape[2], Ac.shape[2]), 'E_N_A': np.random.randn(chi, chi, B.shape[0], Bc.shape[0]) + 1j * np.random.randn(chi, chi, B.shape[0], Bc.shape[0]), 'E_N_B': np.random.randn(chi, chi, A.shape[0], Ac.shape[0]) + 1j * np.random.randn(chi, chi, A.shape[0], Ac.shape[0]),
               'C_NW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi)}
    else:
        # raise Exception("cannot use nonrandom initial env")
        env = {'C_NW_A': ncon([A, Ac], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(A.shape[1]**2, A.shape[2]**2)[:chi,:chi],
               'C_NW_B': ncon([B, Bc], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(B.shape[1]**2, B.shape[2]**2)[:chi,:chi],
               'C_NE_A': ncon([A, Ac], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(A.shape[2]**2, A.shape[3]**2)[:chi,:chi],
               'C_NE_B': ncon([B, Bc], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(B.shape[2]**2, B.shape[3]**2)[:chi,:chi],
               'C_SW_A': ncon([A, Ac], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(A.shape[0]**2, A.shape[1]**2)[:chi,:chi],
               'C_SW_B': ncon([B, Bc], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(B.shape[0]**2, B.shape[1]**2)[:chi,:chi],
               'C_SE_A': ncon([A, Ac], ([-3, 1, 2, -1, 3], [-4, 1, 2, -2, 3])).reshape(A.shape[3]**2, A.shape[0]**2)[:chi,:chi],
               'C_SE_B': ncon([B, Bc], ([-3, 1, 2, -1, 3], [-4, 1, 2, -2, 3])).reshape(B.shape[3]**2, B.shape[0]**2)[:chi,:chi],
               'E_N_A': ncon([A, Ac], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(A.shape[1]**2,A.shape[3]**2,A.shape[2],A.shape[2])[:chi,:chi,:,:],
               'E_N_B': ncon([B, Bc], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(B.shape[1]**2,B.shape[3]**2,B.shape[2],B.shape[2])[:chi,:chi,:,:],
               'E_E_A': ncon([A, Ac], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(A.shape[2]**2,A.shape[0]**2,A.shape[3],A.shape[3])[:chi,:chi,:,:],
               'E_E_B': ncon([B, Bc], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(B.shape[2]**2,B.shape[0]**2,B.shape[3],B.shape[3])[:chi,:chi,:,:],
               'E_S_A': ncon([A, Ac], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(A.shape[3]**2,A.shape[1]**2,A.shape[0],A.shape[0])[:chi,:chi,:,:],
               'E_S_B': ncon([B, Bc], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(B.shape[3]**2,B.shape[1]**2,B.shape[0],B.shape[0])[:chi,:chi,:,:],
               'E_W_A': ncon([A, Ac], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(A.shape[0]**2,A.shape[2]**2,A.shape[1],A.shape[1])[:chi,:chi,:,:],
               'E_W_B': ncon([B, Bc], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(B.shape[0]**2,B.shape[2]**2,B.shape[1],B.shape[1])[:chi,:chi,:,:]}

    errora, errorb = np.inf, np.inf
    for iter in range(maxiter):
        if "rhoA" in env and "rhoB" in env:
            prevrhoA = env["rhoA"]
            prevrhoB = env["rhoB"]
        else:
            prevrhoA = 0
            prevrhoB = 0
            errora, errorb = np.inf, np.inf

        for itermin in range(4):
            print("\t", iter, "  ", itermin)
            env = __CTMRT_left_test(A, B, chi, Ac, Bc, env, invprecision, ifprint=ifprint)
            env = __rot1env(env)
            A = __rot1(A)
            B = __rot1(B)
            Ac = __rot1(Ac)
            Bc = __rot1(Bc)

        preverrora, preverrorb = errora, errorb
        errora, errorb = norm(env["rhoA"] - prevrhoA), norm(env["rhoB"] - prevrhoB)

        print(env['rhoA'])
        print(env['rhoB'])
        print("Errorz =", errora, "\t", errorb, "\t\t >", precision)
        if (errora < precision and errorb < precision) or (preverrora < errora and preverrorb < errorb and errorb < 1e-5 and errora < 1e-5):
            env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
            break

    env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
    return env

# def __CTMRT_left(A, B, chi, Ac, Bc, env0={}, invprecision=1e-10, ifprint=False):
#     env = copy.deepcopy(env0)
#
#     t0 = time()
#     D = A.shape[0]
#     d = A.shape[-1]
#
#     E_E_A = env['E_E_A']
#     E_E_B = env['E_E_B']
#     E_W_A = env['E_W_A']
#     E_W_B = env['E_W_B']
#     E_S_A = env['E_S_A']
#     E_S_B = env['E_S_B']
#     E_N_A = env['E_N_A']
#     E_N_B = env['E_N_B']
#     C_NW_A = env['C_NW_A']
#     C_NE_B = env['C_NE_B']
#     C_SE_A = env['C_SE_A']
#     C_SW_B = env['C_SW_B']
#     C_NW_B = env['C_NW_B']
#     C_NE_A = env['C_NE_A']
#     C_SE_B = env['C_SE_B']
#     C_SW_A = env['C_SW_A']
#
#     if ifprint: print("\t##################################################")
#
#     if ifprint: print("Timing 1", time() - t0, "s")
#     t0 = time()
#
#     # tensors = [C_NW_A, A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, Ac, B, Bc]
#     # connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1],
#     #             [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
#     # con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
#     # UpperHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     # tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, Bc, A, Ac]
#     # connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16],
#     #             [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
#     # con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
#     # BottomHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     #
#     # tensors = [C_NW_B, B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, Bc, A, Ac]
#     # connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1],
#     #             [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
#     # con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
#     # UpperHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     # tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, Ac, B, Bc]
#     # connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16],
#     #             [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
#     # con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
#     # BottomHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     tensors = [C_NW_A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, A, Ac, B, Bc]
#     connects = [[15, 16], [15, 14, 11, 12], [14, 13, 6, 3], [-4, 16, 9, 10], [17, 13], [17, -1, 4, 7], [11, 8, -5, 9, 2], [12, 5, -6, 10, 2], [6, 7, -2, 8, 1], [3, 4, -3, 5, 1]]
#     con_order = [15, 13, 16, 12, 10, 11, 9, 2, 17, 3, 4, 6, 7, 1, 14, 5, 8]
#     UpperHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, Bc, A, Ac]
#     connects = [[13, 14], [13, -4, 9, 10], [16, 17], [16, 15, 5, 6], [15, 14, 12, 11], [-1, 17, 3, 4], [-5, 8, 11, 9, 1], [-6, 7, 12, 10, 1], [-2, 4, 6, 8, 2], [-3, 3, 5, 7, 2]]
#     con_order = [13, 14, 16, 9, 11, 17, 6, 4, 10, 12, 1, 5, 3, 2, 15, 8, 7]
#     BottomHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     tensors = [C_NW_B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, B, Bc, A, Ac]
#     connects = [[15, 16], [15, 14, 11, 12], [14, 13, 6, 3], [-4, 16, 9, 10], [17, 13], [17, -1, 4, 7], [11, 8, -5, 9, 2], [12, 5, -6, 10, 2], [6, 7, -2, 8, 1], [3, 4, -3, 5, 1]]
#     con_order = [15, 16, 12, 10, 17, 13, 3, 4, 6, 7, 1, 11, 9, 2, 14, 5, 8]
#     UpperHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, Ac, B, Bc]
#     connects = [[13, 14], [13, -4, 9, 10], [16, 17], [16, 15, 5, 6], [15, 14, 12, 11], [-1, 17, 3, 4], [-5, 8, 11, 9, 1], [-6, 7, 12, 10, 1], [-2, 4, 6, 8, 2], [-3, 3, 5, 7, 2]]
#     con_order = [13, 16, 14, 9, 11, 17, 6, 4, 10, 12, 1, 5, 3, 2, 15, 8, 7]
#     BottomHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     if ifprint: print("\tHalves calculated")
#
#     if ifprint: print("Timing 2", time() - t0, "s")
#     t0 = time()
#
#     # -R=
#     ifqr = False
#     RUA = (UpperHalfA if not ifqr else qr(UpperHalfA, mode='r')[0]) / norm(UpperHalfA)
#     RUB = (UpperHalfB if not ifqr else qr(UpperHalfB, mode='r')[0]) / norm(UpperHalfB)
#     RBA = (BottomHalfA if not ifqr else qr(BottomHalfA, mode='r')[0]) / norm(BottomHalfA)
#     RBB = (BottomHalfB if not ifqr else qr(BottomHalfB, mode='r')[0]) / norm(BottomHalfB)
#
#     if ifprint: print("\tQR done")
#
#     if ifprint: print("Timing 3", time() - t0, "s")
#     t0 = time()
#
#     # =P-
#     def create_isometries(X1, X2):
#         # -X1= to ta górna macierz w CTMRG_BY_QR (R bez tyldy)
#         # P1 to trójkąt podstawą zwrócony w górę
#         vh, s, u = svd(X1 @ X2.T)
#         # print(s)
#         sd = [1] + [0] * (chi - 1)
#         maxnonzeroindex = 1
#         for i in range(1, D * D * chi):
#             if s[i] > s[0] * invprecision:
#                 maxnonzeroindex = i
#                 if i < chi: sd[i] = 1 / np.sqrt(s[i] / s[0])
#             else:
#                 break
#
#         print("\t\t", maxnonzeroindex, " / ", chi, "\t", str(s[chi] / s[0]), "" if maxnonzeroindex < chi else "\t!!!!!!!")
#         if ifprint: print("\t", sd)
#
#         sd = np.diag(sd)
#         P2 = (X1.T @ (u.conj().T)[:, :chi] @ sd).reshape(chi, D, D, chi)
#         P1 = (X2.T @ (vh.conj())[:, :chi] @ sd).reshape(chi, D, D, chi)
#         return P1, P2
#
#     P1A, P2A = create_isometries(RUB, RBB)
#     P1B, P2B = create_isometries(RUA, RBA)
#
#     if ifprint: print("\tIsometries calculated")
#
#     if ifprint: print("Timing 4", time() - t0, "s")
#     t0 = time()
#
#     # Uppercorner
#     C_NW_A_new = (P1B.reshape(chi * D * D, chi).T @ ncon([C_NW_B, E_N_A], ([1, -1], [-4, 1, -2, -3])).reshape(chi * D * D, chi)).T
#     C_NW_B_new = (P1A.reshape(chi * D * D, chi).T @ ncon([C_NW_A, E_N_B], ([1, -1], [-4, 1, -2, -3])).reshape(chi * D * D, chi)).T
#
#     # Lowercorner
#     C_SW_A_new = P2A.reshape(chi * D * D, chi).T @ ncon([C_SW_B, E_S_A], ([-1, 1], [1, -4, -2, -3])).reshape(chi * D * D, chi)
#     C_SW_B_new = P2B.reshape(chi * D * D, chi).T @ ncon([C_SW_A, E_S_B], ([-1, 1], [1, -4, -2, -3])).reshape(chi * D * D, chi)
#
#     # Edge
#     connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
#     con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
#     tensors = [P2A, P1B, E_W_B, A, Ac]
#     E_W_A_new = ncon(tensors, connects, con_order)
#     tensors = [P2B, P1A, E_W_A, B, Bc]
#     E_W_B_new = ncon(tensors, connects, con_order)
#
#     E_W_A = E_W_A_new / norm(E_W_A_new)
#     E_W_B = E_W_B_new / norm(E_W_B_new)
#     C_NW_A = C_NW_A_new / norm(C_NW_A_new)
#     C_SW_B = C_SW_B_new / norm(C_SW_B_new)
#     C_NW_B = C_NW_B_new / norm(C_NW_B_new)
#     C_SW_A = C_SW_A_new / norm(C_SW_A_new)
#
#     if ifprint: print("\tTensors updated")
#
#     if ifprint: print("Timing 5", time() - t0, "s")
#     t0 = time()
#
#     tensors = [C_NW_A, A, E_N_B, C_SW_A, E_W_B, C_SE_A, E_S_B, C_NE_A, E_E_B, Ac]
#     connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     con_order = [4, 3, 2, 7, 10, 11, 16, 6, 5, 1, 14, 13, 9, 8, 15, 12]
#     rhoA = ncon(tensors, connects, con_order)
#
#     tensors = [C_NW_B, B, E_N_A, C_SW_B, E_W_A, C_SE_B, E_S_A, C_NE_B, E_E_A, Bc]
#     connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     con_order = [1, 7, 5, 3, 2, 6, 13, 14, 4, 10, 11, 16, 9, 8, 12, 15]
#     rhoB = ncon(tensors, connects, con_order)
#
#     rhoA = rhoA / np.trace(rhoA)
#     rhoA = (rhoA + rhoA.conj().T) / 2
#     if ifprint: print("\t", rhoA)
#     rhoB = rhoB / np.trace(rhoB)
#     rhoB = (rhoB + rhoB.conj().T) / 2
#     if ifprint: print("\t", rhoB)
#
#     # tensors = [P2B, P1A, E_W_A, B, Bc]
#     if ifprint: print("\tDensity matricies calculated")
#
#     if ifprint: print("Timing 6", time() - t0, "s")
#     t0 = time()
#
#     if ifprint: print("\t ---", time() - t0, "s")
#
#     return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B, 'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A, 'C_NW_B': C_NW_B, 'C_SW_A': C_SW_A, 'C_NE_A': C_NE_A, 'C_SE_B': C_SE_B, 'rhoA': rhoA, 'rhoB': rhoB}

def __CTMRT_left_test(A, B, chi, Ac, Bc, env0={}, invprecision=1e-10, ifprint=False):
    env = copy.deepcopy(env0)

    t0 = time()

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
    C_NW_B = env['C_NW_B']
    C_NE_A = env['C_NE_A']
    C_SE_B = env['C_SE_B']
    C_SW_A = env['C_SW_A']

    if ifprint: print("\t##################################################")

    if ifprint: print("Timing 1", time() - t0, "s")
    t0 = time()

    # tensors = [C_NW_A, A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, Ac, B, Bc]
    # connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1],
    #             [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
    # con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
    # UpperHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
    # tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, Bc, A, Ac]
    # connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16],
    #             [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
    # con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
    # BottomHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
    #
    # tensors = [C_NW_B, B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, Bc, A, Ac]
    # connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1],
    #             [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
    # con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
    # UpperHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
    # tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, Ac, B, Bc]
    # connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16],
    #             [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
    # con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
    # BottomHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)

    tensors = [C_NW_A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, A, Ac, B, Bc]
    connects = [[15, 16], [15, 14, 11, 12], [14, 13, 6, 3], [-4, 16, 9, 10], [17, 13], [17, -1, 4, 7], [11, 8, -5, 9, 2], [12, 5, -6, 10, 2], [6, 7, -2, 8, 1], [3, 4, -3, 5, 1]]
    con_order = [15, 13, 16, 12, 10, 11, 9, 2, 17, 3, 4, 6, 7, 1, 14, 5, 8]
    UpperHalfA = ncon(tensors, connects, con_order).reshape(chi * B.shape[2] ** 2, chi * A.shape[2] ** 2)

    tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, Bc, A, Ac]
    connects = [[13, 14], [13, -4, 9, 10], [16, 17], [16, 15, 5, 6], [15, 14, 12, 11], [-1, 17, 3, 4], [-5, 8, 11, 9, 1], [-6, 7, 12, 10, 1], [-2, 4, 6, 8, 2], [-3, 3, 5, 7, 2]]
    con_order = [13, 14, 16, 9, 11, 17, 6, 4, 10, 12, 1, 5, 3, 2, 15, 8, 7]
    BottomHalfA = ncon(tensors, connects, con_order).reshape(chi * A.shape[0] ** 2, chi * B.shape[0] ** 2)

    tensors = [C_NW_B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, B, Bc, A, Ac]
    connects = [[15, 16], [15, 14, 11, 12], [14, 13, 6, 3], [-4, 16, 9, 10], [17, 13], [17, -1, 4, 7], [11, 8, -5, 9, 2], [12, 5, -6, 10, 2], [6, 7, -2, 8, 1], [3, 4, -3, 5, 1]]
    con_order = [15, 16, 12, 10, 17, 13, 3, 4, 6, 7, 1, 11, 9, 2, 14, 5, 8]
    UpperHalfB = ncon(tensors, connects, con_order).reshape(chi * A.shape[2] ** 2, chi * B.shape[2] ** 2)

    tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, Ac, B, Bc]
    connects = [[13, 14], [13, -4, 9, 10], [16, 17], [16, 15, 5, 6], [15, 14, 12, 11], [-1, 17, 3, 4], [-5, 8, 11, 9, 1], [-6, 7, 12, 10, 1], [-2, 4, 6, 8, 2], [-3, 3, 5, 7, 2]]
    con_order = [13, 16, 14, 9, 11, 17, 6, 4, 10, 12, 1, 5, 3, 2, 15, 8, 7]
    BottomHalfB = ncon(tensors, connects, con_order).reshape(chi * B.shape[0] ** 2, chi * A.shape[0] ** 2)

    if ifprint: print("\tHalves calculated")

    if ifprint: print("Timing 2", time() - t0, "s")
    t0 = time()

    # -R=
    ifqr = False
    RUA = (UpperHalfA if not ifqr else qr(UpperHalfA, mode='r')[0]) / norm(UpperHalfA)
    RUB = (UpperHalfB if not ifqr else qr(UpperHalfB, mode='r')[0]) / norm(UpperHalfB)
    RBA = (BottomHalfA if not ifqr else qr(BottomHalfA, mode='r')[0]) / norm(BottomHalfA)
    RBB = (BottomHalfB if not ifqr else qr(BottomHalfB, mode='r')[0]) / norm(BottomHalfB)

    if ifprint: print("\tQR done")

    if ifprint: print("Timing 3", time() - t0, "s")
    t0 = time()

    # =P-
    def create_isometries(X1, X2):
        # -X1= to ta górna macierz w CTMRG_BY_QR (R bez tyldy)
        # P1 to trójkąt podstawą zwrócony w górę
        vh, s, u = svd(X1 @ X2.T)
        # print(s)
        sd = [1] + [0] * (chi - 1)
        maxnonzeroindex = 1
        for i in range(1, len(s)):
            if s[i] > s[0] * invprecision:
                maxnonzeroindex = i
                if i < chi: sd[i] = 1 / np.sqrt(s[i] / s[0])
            else:
                break

        print("\t\t", maxnonzeroindex, " / ", chi, "\t", str(s[chi] / s[0]), "" if maxnonzeroindex < chi else "\t!!!!!!!")
        if ifprint: print("\t", sd)

        sd = np.diag(sd)
        D1 = int(np.round(np.sqrt(X1.shape[-1] / chi)))
        D2 = int(np.round(np.sqrt(X2.shape[-1] / chi)))
        P2 = (X1.T @ (u.conj().T)[:, :chi] @ sd).reshape(chi, D1, D1, chi)
        P1 = (X2.T @ (vh.conj())[:, :chi] @ sd).reshape(chi, D2, D2, chi)
        return P1, P2

    P1A, P2A = create_isometries(RUB, RBB)
    P1B, P2B = create_isometries(RUA, RBA)

    if ifprint: print("\tIsometries calculated")

    if ifprint: print("Timing 4", time() - t0, "s")
    t0 = time()

    # Uppercorner
    C_NW_A_new = (P1B.reshape(chi * P1B.shape[1] ** 2, chi).T @ ncon([C_NW_B, E_N_A], ([1, -1], [-4, 1, -2, -3])).reshape(chi * E_N_A.shape[2] ** 2, chi)).T
    C_NW_B_new = (P1A.reshape(chi * P1A.shape[1] ** 2, chi).T @ ncon([C_NW_A, E_N_B], ([1, -1], [-4, 1, -2, -3])).reshape(chi * E_N_B.shape[2] ** 2, chi)).T

    # Lowercorner
    C_SW_A_new = P2A.reshape(chi * P2A.shape[1] ** 2, chi).T @ ncon([C_SW_B, E_S_A], ([-1, 1], [1, -4, -2, -3])).reshape(chi * E_S_A.shape[2] ** 2, chi)
    C_SW_B_new = P2B.reshape(chi * P2B.shape[1] ** 2, chi).T @ ncon([C_SW_A, E_S_B], ([-1, 1], [1, -4, -2, -3])).reshape(chi * E_S_B.shape[2] ** 2, chi)

    # Edge
    connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
    con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
    tensors = [P2A, P1B, E_W_B, A, Ac]
    E_W_A_new = ncon(tensors, connects, con_order)
    tensors = [P2B, P1A, E_W_A, B, Bc]
    E_W_B_new = ncon(tensors, connects, con_order)

    E_W_A = E_W_A_new / norm(E_W_A_new)
    E_W_B = E_W_B_new / norm(E_W_B_new)
    C_NW_A = C_NW_A_new / norm(C_NW_A_new)
    C_SW_B = C_SW_B_new / norm(C_SW_B_new)
    C_NW_B = C_NW_B_new / norm(C_NW_B_new)
    C_SW_A = C_SW_A_new / norm(C_SW_A_new)

    if ifprint: print("\tTensors updated")

    if ifprint: print("Timing 5", time() - t0, "s")
    t0 = time()

    tensors = [C_NW_A, A, E_N_B, C_SW_A, E_W_B, C_SE_A, E_S_B, C_NE_A, E_E_B, Ac]
    connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
    con_order = [4, 3, 2, 7, 10, 11, 16, 6, 5, 1, 14, 13, 9, 8, 15, 12]
    rhoA = ncon(tensors, connects, con_order)

    tensors = [C_NW_B, B, E_N_A, C_SW_B, E_W_A, C_SE_B, E_S_A, C_NE_B, E_E_A, Bc]
    connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
    con_order = [1, 7, 5, 3, 2, 6, 13, 14, 4, 10, 11, 16, 9, 8, 12, 15]
    rhoB = ncon(tensors, connects, con_order)

    rhoA = rhoA / np.trace(rhoA)
    rhoA = (rhoA + rhoA.conj().T) / 2
    if ifprint: print("\t", rhoA)
    rhoB = rhoB / np.trace(rhoB)
    rhoB = (rhoB + rhoB.conj().T) / 2
    if ifprint: print("\t", rhoB)

    # tensors = [P2B, P1A, E_W_A, B, Bc]
    if ifprint: print("\tDensity matricies calculated")

    if ifprint: print("Timing 6", time() - t0, "s")
    t0 = time()

    if ifprint: print("\t ---", time() - t0, "s")

    return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B, 'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A, 'C_NW_B': C_NW_B, 'C_SW_A': C_SW_A, 'C_NE_A': C_NE_A, 'C_SE_B': C_SE_B, 'rhoA': rhoA, 'rhoB': rhoB}

# def __CTMRT_left_right(A, B, chi, env0={}, invprecision=1e-10, ifprint=False, ifrandom=False):
#     env = copy.deepcopy(env0)
#
#     t0 = time()
#     D = A.shape[0]
#     d = A.shape[-1]
#
#     if ifrandom:
#         env = {'E_E_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_E_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_W_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_W_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_S_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_S_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_N_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_N_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'C_NW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi),
#                'C_NW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi)}
#
#     if len(env) == 0:
#         if chi <= D ** 2:
#             # temp = A.swapaxes(3,2).swapaxes(2,1).swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NW_A = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).reshape(D*D,D*D)
#             # temp = B.swapaxes(3,2).swapaxes(2,1).swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NW_B = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).reshape(D*D,D*D)
#             # temp = A.swapaxes(4,3).swapaxes(3,2).swapaxes(2,1).reshape(D*d,D*D*D)
#             # E_N_A = (temp.conj().T @ temp).reshape(D,D,D,D,D,D).swapaxes(2,3).swapaxes(3,4).swapaxes(4,5).swapaxes(3,4).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3).swapaxes(1,2).reshape(D**2,D**2,D,D)
#             # temp = B.swapaxes(4,3).swapaxes(3,2).swapaxes(2,1).reshape(D*d,D*D*D)
#             # E_N_B = (temp.conj().T @ temp).reshape(D,D,D,D,D,D).swapaxes(2,3).swapaxes(3,4).swapaxes(4,5).swapaxes(3,4).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3).swapaxes(1,2).reshape(D**2,D**2,D,D)
#             #
#             # # work in progress, watch your step
#             # temp = A.swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NE_A = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).swapaxes(1,3).swapaxes(3,4).reshape(D*D,D*D)
#             # temp = B.swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NE_B = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).swapaxes(1,3).swapaxes(3,4).reshape(D*D,D*D)
#             # temp = A.
#
#             C_NW_A = ncon([A, A.conj()], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(D * D, D * D)
#             C_NW_B = ncon([B, B.conj()], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(D * D, D * D)
#             C_NE_A = ncon([A, A.conj()], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(D * D, D * D)
#             C_NE_B = ncon([B, B.conj()], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(D * D, D * D)
#             C_SW_A = ncon([A, A.conj()], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(D * D, D * D)
#             C_SW_B = ncon([B, B.conj()], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(D * D, D * D)
#             C_SE_A = ncon([A, A.conj()], ([-1, 1, 2, -3, 3], [-2, 1, 2, -4, 3])).reshape(D * D, D * D)
#             C_SE_B = ncon([B, B.conj()], ([-1, 1, 2, -3, 3], [-2, 1, 2, -4, 3])).reshape(D * D, D * D)
#
#             E_E_A = ncon([A, A.conj()], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(D * D, D * D, D, D)
#             E_E_B = ncon([B, B.conj()], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(D * D, D * D, D, D)
#             E_N_A = ncon([A, A.conj()], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(D * D, D * D, D, D)
#             E_N_B = ncon([B, B.conj()], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(D * D, D * D, D, D)
#             E_W_A = ncon([A, A.conj()], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(D * D, D * D, D, D)
#             E_W_B = ncon([B, B.conj()], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(D * D, D * D, D, D)
#             E_S_A = ncon([A, A.conj()], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(D * D, D * D, D, D)
#             E_S_B = ncon([B, B.conj()], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(D * D, D * D, D, D)
#
#         if chi > D ** 2:
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2], [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
#             con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
#             C_NW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2], [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
#             con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
#             C_SW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2], [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
#             con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
#             C_SE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2], [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
#             con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
#             C_NE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2], [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
#             con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
#             C_NW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2], [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
#             con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
#             C_SW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2], [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
#             con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
#             C_SE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2], [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
#             con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
#             C_NE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[-1, 3, -5, 5, 2], [-3, 4, -7, 5, 2], [-2, -9, -6, 3, 1], [-4, -10, -8, 4, 1]]
#             con_order = [5, 2, 4, 3, 1]
#             E_W_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[-1, 3, -5, 5, 1], [-3, 4, -7, 5, 1], [-2, -9, -6, 3, 2], [-4, -10, -8, 4, 2]]
#             con_order = [5, 1, 4, 3, 2]
#             E_W_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[-6, 3, -2, -9, 1], [-8, 4, -4, -10, 1], [-5, 5, -1, 3, 2], [-7, 5, -3, 4, 2]]
#             con_order = [5, 2, 3, 1, 4]
#             E_E_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[-6, 3, -2, -9, 2], [-8, 4, -4, -10, 2], [-5, 5, -1, 3, 1], [-7, 5, -3, 4, 1]]
#             con_order = [5, 1, 3, 2, 4]
#             E_E_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
#             con_order = [5, 1, 4, 3, 2]
#             E_N_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
#             con_order = [5, 2, 4, 3, 1]
#             E_S_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
#             con_order = [5, 2, 3, 1, 4]
#             E_S_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
#             con_order = [5, 1, 3, 4, 2]
#             E_N_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#
#         E_E_A = E_E_A[:chi, :chi, :, :] / norm(E_E_A)
#         E_E_B = E_E_B[:chi, :chi, :, :] / norm(E_E_B)
#         E_W_A = E_W_A[:chi, :chi, :, :] / norm(E_W_A)
#         E_W_B = E_W_B[:chi, :chi, :, :] / norm(E_W_B)
#         E_S_A = E_S_A[:chi, :chi, :, :] / norm(E_S_A)
#         E_S_B = E_S_B[:chi, :chi, :, :] / norm(E_S_B)
#         E_N_A = E_N_A[:chi, :chi, :, :] / norm(E_N_A)
#         E_N_B = E_N_B[:chi, :chi, :, :] / norm(E_N_B)
#         C_NW_A = C_NW_A[:chi, :chi] / norm(C_NW_A)
#         C_NE_B = C_NE_B[:chi, :chi] / norm(C_NE_B)
#         C_SE_A = C_SE_A[:chi, :chi] / norm(C_SE_A)
#         C_SW_B = C_SW_B[:chi, :chi] / norm(C_SW_B)
#         C_NW_B = C_NW_B[:chi, :chi] / norm(C_NW_B)
#         C_NE_A = C_NE_A[:chi, :chi] / norm(C_NE_A)
#         C_SE_B = C_SE_B[:chi, :chi] / norm(C_SE_B)
#         C_SW_A = C_SW_A[:chi, :chi] / norm(C_SW_A)
#
#     else:
#         E_E_A = env['E_E_A']
#         E_E_B = env['E_E_B']
#         E_W_A = env['E_W_A']
#         E_W_B = env['E_W_B']
#         E_S_A = env['E_S_A']
#         E_S_B = env['E_S_B']
#         E_N_A = env['E_N_A']
#         E_N_B = env['E_N_B']
#         C_NW_A = env['C_NW_A']
#         C_NE_B = env['C_NE_B']
#         C_SE_A = env['C_SE_A']
#         C_SW_B = env['C_SW_B']
#         C_NW_B = env['C_NW_B']
#         C_NE_A = env['C_NE_A']
#         C_SE_B = env['C_SE_B']
#         C_SW_A = env['C_SW_A']
#
#     t0 = time()
#     if ifprint: print("\t##################################################")
#
#     if False:
#         print("A:", A.shape)
#         print("B:", B.shape)
#
#         print("C_NW_A:", C_NW_A.shape)
#         print("C_NW_B:", C_NW_B.shape)
#         print("C_SW_A:", C_SW_A.shape)
#         print("C_SW_B:", C_SW_B.shape)
#         print("C_NE_A:", C_NW_A.shape)
#         print("C_NE_B:", C_NW_B.shape)
#         print("C_SE_A:", C_SW_A.shape)
#         print("C_SE_B:", C_SW_B.shape)
#
#         print("E_W_A:", E_W_A.shape)
#         print("E_W_B:", E_W_B.shape)
#         print("E_S_A:", E_S_A.shape)
#         print("E_S_B:", E_S_B.shape)
#         print("E_N_A:", E_N_A.shape)
#         print("E_N_B:", E_N_B.shape)
#         print("E_E_A:", E_E_A.shape)
#         print("E_E_B:", E_E_B.shape)
#
#     if ifprint: print("Timing 1", time() - t0, "s")
#     t0 = time()
#
#     tensors = [C_NW_A, A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, A.conj(), B, B.conj()]
#     connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1], [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
#     con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
#     UpperHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, B.conj(), A, A.conj()]
#     connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16], [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
#     con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
#     BottomHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     tensors = [C_NW_B, B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, B.conj(), A, A.conj()]
#     connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1], [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
#     con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
#     UpperHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, A.conj(), B, B.conj()]
#     connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16], [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
#     con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
#     BottomHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     if ifprint: print("\tHalves calculated")
#
#     if ifprint: print("Timing 2", time() - t0, "s")
#     t0 = time()
#
#     # tensors = [C_NW_A, A, E_N_B, C_SW_B, E_W_A, E_W_B, E_S_A, A.conj(), B, B.conj()]
#     # connects = [[1, 2], [10, -2, 8, 11, 6], [-1, 1, 10, 14], [4, 5], [3, 4, 12, 15], [2, 3, 11, 13],
#     #             [5, -4, 16, 17], [14, -3, 9, 13, 6], [8, -5, 16, 12, 7], [9, -6, 17, 15, 7]]
#     # con_order = [5, 2, 1, 11, 10, 4, 17, 15, 16, 12, 7, 13, 14, 6, 3, 8, 9]
#     # LeftHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     # tensors = [E_N_A, C_SE_A, E_S_B, C_NE_B, E_E_A, E_E_B, B, B.conj(), A, A.conj()]
#     # connects = [[1, -4, 9, 10], [2, 3], [-1, 2, 12, 11], [17, 1], [4, 17, 16, 15], [3, 4, 14, 13],
#     #             [9, 16, 7, -5, 5], [10, 15, 8, -6, 5], [7, 14, 12, -2, 6], [8, 13, 11, -3, 6]]
#     # con_order = [1, 3, 17, 9, 16, 2, 14, 12, 13, 11, 6, 10, 15, 5, 4, 7, 8]
#     # RightHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     # -R=
#     RUA = qr(UpperHalfA.T, mode='r')
#     RUB = qr(UpperHalfB.T, mode='r')
#     RBA = qr(BottomHalfA.T, mode='r')
#     RBB = qr(BottomHalfB.T, mode='r')
#     LUA = qr(UpperHalfA, mode='r')
#     LUB = qr(UpperHalfB, mode='r')
#     LBA = qr(BottomHalfA, mode='r')
#     LBB = qr(BottomHalfB, mode='r')
#     # LBA = qr(UpperHalfA, mode='r')
#     # LBB = qr(UpperHalfB, mode='r')
#     # LUA = qr(BottomHalfA, mode='r')
#     # LUB = qr(BottomHalfB, mode='r')
#
#     if ifprint: print("\tQR done")
#
#     if ifprint: print("Timing 3", time() - t0, "s")
#     t0 = time()
#
#     # =P-
#     def create_isometries(X1, X2):
#         # -X1= to ta górna macierz w CTMRG_BY_QR
#         u, s, vh = truncate3((X1 @ X2.T).T, chi)
#         s = s / s[0]
#         s = np.where(s < s[0] * invprecision, 0, 1 / np.sqrt(s))
#         pr = False
#         for i in range(len(s)):
#             if s[i] < s[0] * invprecision:
#                 s[i] = 0
#                 if not pr:
#                     pr = True
#                     print("\t\t", i)
#             else:
#                 s[i] = 1 / np.sqrt(s[i])
#         if not pr: print("\t\t", i + 1, "\t", "!!!!!!!")
#         if ifprint: print("\t", s)
#         s = np.diag(s)
#         p2 = (s @ u.conj().T @ X1).T.reshape(chi, D, D, chi)
#         p1 = (s @ vh.conj() @ X2).T.reshape(chi, D, D, chi)
#         return p1, p2  # u, s, vh = truncate3(X1 @ X2.T, chi)  # s = np.where(s < s[0] * invprecision, 0, 1 / np.sqrt(s))  # if ifprint: print("\t", s)  # s = np.diag(s)  # p2 = (s @ u.conj().T @ X1).T.reshape(chi, D, D, chi)  # p1 = (s @ vh.conj() @ X2).T.reshape(chi, D, D, chi)  # return p1, p2
#
#     # oryginalne, zawiesza się dopeiro przy 6 kroku i błąd oscyluje wokół 6e-9
#     # PR1A, PR2A = create_isometries(RUA[0], RBA[0])
#     # PR1B, PR2B = create_isometries(RUB[0], RBB[0])
#     # PL1A, PL2A = create_isometries(LUA[0], LBA[0])
#     # PL1B, PL2B = create_isometries(LUB[0], LBB[0])
#
#     PR1A, PR2A = create_isometries(RUA[0], RBA[0])
#     PR1B, PR2B = create_isometries(RUB[0], RBB[0])
#     PL2A, PL1A = create_isometries(LUA[0], LBA[0])
#     PL2B, PL1B = create_isometries(LUB[0], LBB[0])
#
#     if ifprint: print("\tIsometries calculated")
#
#     if ifprint: print("Timing 4", time() - t0, "s")
#     t0 = time()
#
#     def CalcEdge(Proj1, Edge, X, Xc, Proj2):
#         tensors = [Proj1, Proj2, Edge, X, Xc]
#         connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
#         con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
#         return ncon(tensors, connects, con_order)
#
#     def CalcCorner1(Proj, Corner, Edge):
#         #
#         # dołącza C-E-
#         #        | |
#         return (Proj.reshape(chi * D * D, chi).T @ ncon([Corner, Edge], ([1, -1], [-4, 1, -2, -3])).reshape(chi * D * D, chi)).T
#
#     def CalcCorner2(Proj, Corner, Edge):
#         #
#         # dołącza -E-C
#         #         | |
#         return Proj.reshape(chi * D * D, chi).T @ ncon([Corner, Edge], ([-1, 1], [1, -4, -2, -3])).reshape(chi * D * D, chi)
#
#     # Uppercorner
#     # C_NW_A_new = (PR1A.reshape(chi * D * D, chi).T @ ncon([C_NW_B, E_N_A], ([1, -1], [-4, 1, -2, -3])).reshape(
#     #     chi * D * D, chi)).T
#     # C_NW_B_new = (PR1B.reshape(chi * D * D, chi).T @ ncon([C_NW_A, E_N_B], ([1, -1], [-4, 1, -2, -3])).reshape(
#     #     chi * D * D, chi)).T
#     C_NW_A_new = CalcCorner1(PR1A, C_NW_B, E_N_A)
#     C_NW_B_new = CalcCorner1(PR1B, C_NW_A, E_N_B)
#     C_SE_A_new = CalcCorner1(PL1A, C_SE_B, E_S_A)
#     C_SE_B_new = CalcCorner1(PL1B, C_SE_A, E_S_B)
#
#     # Lowercorner
#     # C_SW_A_new = PR2B.reshape(chi * D * D, chi).T @ ncon([C_SW_B, E_S_A], ([-1, 1], [1, -4, -2, -3])).reshape(
#     #     chi * D * D, chi)
#     # C_SW_B_new = PR2A.reshape(chi * D * D, chi).T @ ncon([C_SW_A, E_S_B], ([-1, 1], [1, -4, -2, -3])).reshape(
#     #     chi * D * D, chi)
#     C_SW_A_new = CalcCorner2(PR2B, C_SW_B, E_S_A)
#     C_SW_B_new = CalcCorner2(PR2A, C_SW_A, E_S_B)
#     C_NE_A_new = CalcCorner2(PL2B, C_NE_B, E_N_A)
#     C_NE_B_new = CalcCorner2(PL2A, C_NE_A, E_N_B)
#
#     # Edge
#     # tensors = [PR2A, PR1B, E_W_B, A, A.conj()]
#     # connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
#     # con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
#     # E_W_A_new = ncon(tensors, connects, con_order)
#     # tensors = [PR2B, PR1A, E_W_A, B, B.conj()]
#     # connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
#     # con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
#     # E_W_B_new = ncon(tensors, connects, con_order)
#
#     E_W_A_new = CalcEdge(PR2A, E_W_B, A, A.conj(), PR1B)
#     E_W_B_new = CalcEdge(PR2B, E_W_A, B, B.conj(), PR1A)
#     E_E_A_new = CalcEdge(PL2A, E_E_B, A, A.conj(), PL1B)
#     E_E_B_new = CalcEdge(PL2B, E_E_A, B, B.conj(), PL1A)
#
#     E_W_A = E_W_A_new / norm(E_W_A_new)
#     E_W_B = E_W_B_new / norm(E_W_B_new)
#     C_NW_A = C_NW_A_new / norm(C_NW_A_new)
#     C_NW_B = C_NW_B_new / norm(C_NW_B_new)
#     C_SW_A = C_SW_A_new / norm(C_SW_A_new)
#     C_SW_B = C_SW_B_new / norm(C_SW_B_new)
#     E_E_A = E_E_A_new / norm(E_E_A_new)
#     E_E_B = E_E_B_new / norm(E_E_B_new)
#     C_SE_A = C_SE_A_new / norm(C_SE_A_new)
#     C_SE_B = C_SE_B_new / norm(C_SE_B_new)
#     C_NE_A = C_NE_A_new / norm(C_NE_A_new)
#     C_NE_B = C_NE_B_new / norm(C_NE_B_new)
#
#     if ifprint: print("\tTensors updated")
#
#     if ifprint: print("Timing 5", time() - t0, "s")
#     t0 = time()
#
#     tensors = [C_NW_A, A, E_N_B, C_SW_A, E_W_B, C_SE_A, E_S_B, C_NE_A, E_E_B, A.conj()]
#     connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     con_order = [4, 3, 2, 7, 10, 11, 16, 6, 5, 1, 14, 13, 9, 8, 15, 12]
#     rhoA = ncon(tensors, connects, con_order)
#
#     tensors = [C_NW_B, B, E_N_A, C_SW_B, E_W_A, C_SE_B, E_S_A, C_NE_B, E_E_A, B.conj()]
#     connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     con_order = [1, 7, 5, 3, 2, 6, 13, 14, 4, 10, 11, 16, 9, 8, 12, 15]
#     rhoB = ncon(tensors, connects, con_order)
#
#     rhoA = rhoA / np.trace(rhoA)
#     rhoA = (rhoA + rhoA.conj().T) / 2
#     if ifprint: print("\t", rhoA)
#     rhoB = rhoB / np.trace(rhoB)
#     rhoB = (rhoB + rhoB.conj().T) / 2
#     if ifprint: print("\t", rhoB)
#
#     if ifprint: print("\tDensity matricies calculated")
#
#     if ifprint: print("Timing 6", time() - t0, "s")
#     t0 = time()
#
#     if ifprint: print("\t ---", time() - t0, "s")
#
#     return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B, 'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A, 'C_NW_B': C_NW_B, 'C_SW_A': C_SW_A, 'C_NE_A': C_NE_A, 'C_SE_B': C_SE_B, 'rhoA': rhoA, 'rhoB': rhoB}
#
# def __CTMRT_left_old(A, B, chi, env0={}, invprecision=1e-10, ifprint=False, ifrandom=False):
#     env = copy.deepcopy(env0)
#
#     t0 = time()
#     D = A.shape[0]
#     d = A.shape[-1]
#
#     if ifrandom and len(env) == 0:
#         env = {'E_E_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_E_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_W_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_W_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_S_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_S_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_N_A': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'E_N_B': np.random.randn(chi, chi, D, D) + 1j * np.random.randn(chi, chi, D, D), 'C_NW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi),
#                'C_NW_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SW_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_NE_A': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi), 'C_SE_B': np.random.randn(chi, chi) + 1j * np.random.randn(chi, chi)}
#
#     if len(env) == 0:
#         if chi <= D ** 2:
#             # temp = A.swapaxes(3,2).swapaxes(2,1).swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NW_A = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).reshape(D*D,D*D)
#             # temp = B.swapaxes(3,2).swapaxes(2,1).swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NW_B = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).reshape(D*D,D*D)
#             # temp = A.swapaxes(4,3).swapaxes(3,2).swapaxes(2,1).reshape(D*d,D*D*D)
#             # E_N_A = (temp.conj().T @ temp).reshape(D,D,D,D,D,D).swapaxes(2,3).swapaxes(3,4).swapaxes(4,5).swapaxes(3,4).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3).swapaxes(1,2).reshape(D**2,D**2,D,D)
#             # temp = B.swapaxes(4,3).swapaxes(3,2).swapaxes(2,1).reshape(D*d,D*D*D)
#             # E_N_B = (temp.conj().T @ temp).reshape(D,D,D,D,D,D).swapaxes(2,3).swapaxes(3,4).swapaxes(4,5).swapaxes(3,4).swapaxes(0,1).swapaxes(1,2).swapaxes(2,3).swapaxes(1,2).reshape(D**2,D**2,D,D)
#             #
#             # # work in progress, watch your step
#             # temp = A.swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NE_A = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).swapaxes(1,3).swapaxes(3,4).reshape(D*D,D*D)
#             # temp = B.swapaxes(4,3).swapaxes(3,2).reshape(D*D*d,D*D)
#             # C_NE_B = (temp.conj().T @ temp).reshape(D,D,D,D).swapaxes(1,2).swapaxes(1,3).swapaxes(3,4).reshape(D*D,D*D)
#             # temp = A.
#
#             C_NW_A = ncon([A, A.conj()], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(D * D, D * D)
#             C_NW_B = ncon([B, B.conj()], ([1, -1, -3, 2, 3], [1, -2, -4, 2, 3])).reshape(D * D, D * D)
#             C_NE_A = ncon([A, A.conj()], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(D * D, D * D)
#             C_NE_B = ncon([B, B.conj()], ([1, 2, -1, -3, 3], [1, 2, -2, -4, 3])).reshape(D * D, D * D)
#             C_SW_A = ncon([A, A.conj()], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(D * D, D * D)
#             C_SW_B = ncon([B, B.conj()], ([-1, -3, 1, 2, 3], [-2, -4, 1, 2, 3])).reshape(D * D, D * D)
#             C_SE_A = ncon([A, A.conj()], ([-1, 1, 2, -3, 3], [-2, 1, 2, -4, 3])).reshape(D * D, D * D)
#             C_SE_B = ncon([B, B.conj()], ([-1, 1, 2, -3, 3], [-2, 1, 2, -4, 3])).reshape(D * D, D * D)
#
#             E_E_A = ncon([A, A.conj()], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(D * D, D * D, D, D)
#             E_E_B = ncon([B, B.conj()], ([-3, 1, -1, -5, 2], [-4, 1, -2, -6, 2])).reshape(D * D, D * D, D, D)
#             E_N_A = ncon([A, A.conj()], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(D * D, D * D, D, D)
#             E_N_B = ncon([B, B.conj()], ([1, -1, -5, -3, 2], [1, -2, -6, -4, 2])).reshape(D * D, D * D, D, D)
#             E_W_A = ncon([A, A.conj()], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(D * D, D * D, D, D)
#             E_W_B = ncon([B, B.conj()], ([-1, -5, -3, 1, 2], [-2, -6, -4, 1, 2])).reshape(D * D, D * D, D, D)
#             E_S_A = ncon([A, A.conj()], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(D * D, D * D, D, D)
#             E_S_B = ncon([B, B.conj()], ([-5, -3, 1, -1, 2], [-6, -4, 1, -2, 2])).reshape(D * D, D * D, D, D)
#
#         if chi <= D ** 4 and chi > D ** 2:
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2], [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
#             con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
#             C_NW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2], [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
#             con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
#             C_SW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2], [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
#             con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
#             C_SE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj()]
#             connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2], [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
#             con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
#             C_NE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[15, 5, 6, 14, 1], [15, 10, 11, 14, 1], [16, -1, 8, 5, 2], [16, -3, 9, 10, 2], [6, 7, -5, 13, 4], [11, 12, -7, 13, 4], [8, -2, -6, 7, 3], [9, -4, -8, 12, 3]]
#             con_order = [15, 14, 1, 16, 2, 13, 4, 5, 10, 6, 11, 8, 7, 9, 12, 3]
#             C_NW_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[-1, 5, 6, 16, 1], [-3, 10, 11, 16, 1], [-2, -6, 8, 5, 2], [-4, -8, 9, 10, 2], [6, 7, 14, 15, 4], [11, 12, 14, 15, 4], [8, -5, 13, 7, 3], [9, -7, 13, 12, 3]]
#             con_order = [16, 1, 13, 3, 14, 15, 4, 7, 12, 6, 11, 10, 9, 5, 8, 2]
#             C_SW_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[-6, 5, 6, -2, 1], [-8, 10, 11, -4, 1], [-5, 13, 8, 5, 2], [-7, 13, 9, 10, 2], [6, 7, 16, -1, 4], [11, 12, 16, -3, 4], [8, 14, 15, 7, 3], [9, 14, 15, 12, 3]]
#             con_order = [16, 4, 14, 15, 3, 13, 2, 7, 12, 8, 9, 5, 6, 1, 10, 11]
#             C_SE_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj(), A, A.conj(), B, B.conj()]
#             connects = [[13, 5, 6, -5, 1], [13, 10, 11, -7, 1], [14, 15, 8, 5, 2], [14, 15, 9, 10, 2], [6, 7, -2, -6, 4], [11, 12, -4, -8, 4], [8, 16, -1, 7, 3], [9, 16, -3, 12, 3]]
#             con_order = [13, 1, 14, 15, 2, 16, 3, 8, 9, 5, 10, 6, 7, 11, 12, 4]
#             C_NE_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[-1, 3, -5, 5, 2], [-3, 4, -7, 5, 2], [-2, -9, -6, 3, 1], [-4, -10, -8, 4, 1]]
#             con_order = [5, 2, 4, 3, 1]
#             E_W_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[-1, 3, -5, 5, 1], [-3, 4, -7, 5, 1], [-2, -9, -6, 3, 2], [-4, -10, -8, 4, 2]]
#             con_order = [5, 1, 4, 3, 2]
#             E_W_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[-6, 3, -2, -9, 1], [-8, 4, -4, -10, 1], [-5, 5, -1, 3, 2], [-7, 5, -3, 4, 2]]
#             con_order = [5, 2, 3, 1, 4]
#             E_E_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[-6, 3, -2, -9, 2], [-8, 4, -4, -10, 2], [-5, 5, -1, 3, 1], [-7, 5, -3, 4, 1]]
#             con_order = [5, 1, 3, 2, 4]
#             E_E_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
#             con_order = [5, 1, 4, 3, 2]
#             E_N_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [B, B.conj(), A, A.conj()]
#             connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
#             con_order = [5, 2, 4, 3, 1]
#             E_S_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[-9, -6, 3, -2, 1], [-10, -8, 4, -4, 1], [3, -5, 5, -1, 2], [4, -7, 5, -3, 2]]
#             con_order = [5, 2, 3, 1, 4]
#             E_S_A = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#             tensors = [A, A.conj(), B, B.conj()]
#             connects = [[5, -1, 3, -5, 1], [5, -3, 4, -7, 1], [3, -2, -9, -6, 2], [4, -4, -10, -8, 2]]
#             con_order = [5, 1, 3, 4, 2]
#             E_N_B = ncon(tensors, connects, con_order).reshape(D * D * D * D, D * D * D * D, D, D)
#
#         E_E_A = E_E_A[:chi, :chi, :, :] / norm(E_E_A)
#         E_E_B = E_E_B[:chi, :chi, :, :] / norm(E_E_B)
#         E_W_A = E_W_A[:chi, :chi, :, :] / norm(E_W_A)
#         E_W_B = E_W_B[:chi, :chi, :, :] / norm(E_W_B)
#         E_S_A = E_S_A[:chi, :chi, :, :] / norm(E_S_A)
#         E_S_B = E_S_B[:chi, :chi, :, :] / norm(E_S_B)
#         E_N_A = E_N_A[:chi, :chi, :, :] / norm(E_N_A)
#         E_N_B = E_N_B[:chi, :chi, :, :] / norm(E_N_B)
#         C_NW_A = C_NW_A[:chi, :chi] / norm(C_NW_A)
#         C_NE_B = C_NE_B[:chi, :chi] / norm(C_NE_B)
#         C_SE_A = C_SE_A[:chi, :chi] / norm(C_SE_A)
#         C_SW_B = C_SW_B[:chi, :chi] / norm(C_SW_B)
#         C_NW_B = C_NW_B[:chi, :chi] / norm(C_NW_B)
#         C_NE_A = C_NE_A[:chi, :chi] / norm(C_NE_A)
#         C_SE_B = C_SE_B[:chi, :chi] / norm(C_SE_B)
#         C_SW_A = C_SW_A[:chi, :chi] / norm(C_SW_A)
#
#     else:
#         E_E_A = env['E_E_A']
#         E_E_B = env['E_E_B']
#         E_W_A = env['E_W_A']
#         E_W_B = env['E_W_B']
#         E_S_A = env['E_S_A']
#         E_S_B = env['E_S_B']
#         E_N_A = env['E_N_A']
#         E_N_B = env['E_N_B']
#         C_NW_A = env['C_NW_A']
#         C_NE_B = env['C_NE_B']
#         C_SE_A = env['C_SE_A']
#         C_SW_B = env['C_SW_B']
#         C_NW_B = env['C_NW_B']
#         C_NE_A = env['C_NE_A']
#         C_SE_B = env['C_SE_B']
#         C_SW_A = env['C_SW_A']
#
#     # errora, errorb = 1000000000, 1000000000
#     #
#     # tensors = [C_NW_A, A, E_N_B, C_SW_A, E_W_B, C_SE_A, E_S_B, C_NE_A, E_E_B, A.conj()]
#     # connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7],
#     #             [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     # con_order = [4, 3, 2, 7, 10, 11, 16, 6, 5, 1, 14, 13, 9, 8, 15, 12]
#     # rhoA = ncon(tensors, connects, con_order)
#     #
#     # tensors = [C_NW_B, B, E_N_A, C_SW_B, E_W_A, C_SE_B, E_S_A, C_NE_B, E_E_A, B.conj()]
#     # connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7],
#     #             [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     # con_order = [1, 7, 5, 3, 2, 6, 13, 14, 4, 10, 11, 16, 9, 8, 12, 15]
#     # rhoB = ncon(tensors, connects, con_order)
#     #
#     # rhoA = rhoA / np.trace(rhoA)
#     # rhoA = (rhoA + rhoA.conj().T) / 2
#     # print(rhoA)
#     # rhoB = rhoB / np.trace(rhoB)
#     # rhoB = (rhoB + rhoB.conj().T) / 2
#     # print(rhoB)
#
#     if ifprint: print("\t##################################################")
#
#     if False:
#         print("A:", A.shape)
#         print("B:", B.shape)
#
#         print("C_NW_A:", C_NW_A.shape)
#         print("C_NW_B:", C_NW_B.shape)
#         print("C_SW_A:", C_SW_A.shape)
#         print("C_SW_B:", C_SW_B.shape)
#         print("C_NE_A:", C_NW_A.shape)
#         print("C_NE_B:", C_NW_B.shape)
#         print("C_SE_A:", C_SW_A.shape)
#         print("C_SE_B:", C_SW_B.shape)
#
#         print("E_W_A:", E_W_A.shape)
#         print("E_W_B:", E_W_B.shape)
#         print("E_S_A:", E_S_A.shape)
#         print("E_S_B:", E_S_B.shape)
#         print("E_N_A:", E_N_A.shape)
#         print("E_N_B:", E_N_B.shape)
#         print("E_E_A:", E_E_A.shape)
#         print("E_E_B:", E_E_B.shape)
#
#     if ifprint: print("Timing 1", time() - t0, "s")
#     t0 = time()
#
#     tensors = [C_NW_A, A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, A.conj(), B, B.conj()]
#     connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1], [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
#     con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
#     UpperHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, B.conj(), A, A.conj()]
#     connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16], [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
#     con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
#     BottomHalfA = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     tensors = [C_NW_B, B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, B.conj(), A, A.conj()]
#     connects = [[3, 4], [10, 7, -5, 11, 5], [2, 3, 10, 13], [1, 2, 9, 14], [4, -4, 11, 12], [17, 1], [-1, 17, 16, 15], [13, 8, -6, 12, 5], [9, 16, -2, 7, 6], [14, 15, -3, 8, 6]]
#     con_order = [3, 4, 13, 12, 1, 17, 14, 15, 10, 11, 5, 9, 16, 6, 2, 8, 7]
#     UpperHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, A.conj(), B, B.conj()]
#     connects = [[1, 2], [-4, 1, 10, 11], [4, 5], [3, 4, 15, 14], [2, 3, 12, 13], [5, -1, 17, 16], [-5, 8, 12, 10, 7], [-6, 9, 13, 11, 7], [-2, 17, 15, 8, 6], [-3, 16, 14, 9, 6]]
#     con_order = [1, 5, 4, 17, 15, 2, 11, 13, 10, 12, 7, 16, 14, 6, 3, 9, 8]
#     BottomHalfB = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     if ifprint: print("\tHalves calculated")
#
#     if ifprint: print("Timing 2", time() - t0, "s")
#     t0 = time()
#
#     # tensors = [C_NW_A, A, E_N_B, C_SW_B, E_W_A, E_W_B, E_S_A, A.conj(), B, B.conj()]
#     # connects = [[1, 2], [10, -2, 8, 11, 6], [-1, 1, 10, 14], [4, 5], [3, 4, 12, 15], [2, 3, 11, 13],
#     #             [5, -4, 16, 17], [14, -3, 9, 13, 6], [8, -5, 16, 12, 7], [9, -6, 17, 15, 7]]
#     # con_order = [5, 2, 1, 11, 10, 4, 17, 15, 16, 12, 7, 13, 14, 6, 3, 8, 9]
#     # LeftHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#     # tensors = [E_N_A, C_SE_A, E_S_B, C_NE_B, E_E_A, E_E_B, B, B.conj(), A, A.conj()]
#     # connects = [[1, -4, 9, 10], [2, 3], [-1, 2, 12, 11], [17, 1], [4, 17, 16, 15], [3, 4, 14, 13],
#     #             [9, 16, 7, -5, 5], [10, 15, 8, -6, 5], [7, 14, 12, -2, 6], [8, 13, 11, -3, 6]]
#     # con_order = [1, 3, 17, 9, 16, 2, 14, 12, 13, 11, 6, 10, 15, 5, 4, 7, 8]
#     # RightHalf = ncon(tensors, connects, con_order).reshape(chi * D * D, chi * D * D)
#
#     # -R=
#     RUA = qr(UpperHalfA.T, mode='r')
#     RUB = qr(UpperHalfB.T, mode='r')
#     RBA = qr(BottomHalfA.T, mode='r')
#     RBB = qr(BottomHalfB.T, mode='r')
#
#     if ifprint: print("\tQR done")
#
#     if ifprint: print("Timing 3", time() - t0, "s")
#     t0 = time()
#
#     # =P-
#     def create_isometries(X1, X2):
#         # -X1= to ta górna macierz w CTMRG_BY_QR
#         u, s, vh = truncate3((X1 @ X2.T).T, chi)
#         s = np.where(s < s[0] * invprecision, 0, 1 / np.sqrt(s))
#         pr = False
#         for i in range(len(s)):
#             if s[i] < s[0] * invprecision:
#                 s[i] = 0
#                 if not pr:
#                     pr = True
#                     print("\t\t", i)
#             else:
#                 s[i] = 1 / np.sqrt(s[i])
#         if not pr: print("\t\t", i + 1, "\t", "!!!!!!!")
#         if ifprint: print("\t", s)
#         s = np.diag(s)
#         p2 = (s @ u.conj().T @ X1).T.reshape(chi, D, D, chi)
#         p1 = (s @ vh.conj() @ X2).T.reshape(chi, D, D, chi)
#         return p1, p2
#
#     P1A, P2A = create_isometries(RUA[0], RBA[0])
#     P1B, P2B = create_isometries(RUB[0], RBB[0])
#     # if ifprint: print((ncon([P1A,P2A],([-1,-2,-3,1],[-4,-5,-6,1]))).reshape(chi*D*D,chi*D*D))
#     # if ifprint: print((ncon([P1B,P2B],([-1,-2,-3,1],[-4,-5,-6,1]))).reshape(chi*D*D,chi*D*D))
#
#     if ifprint: print("\tIsometries calculated")
#
#     if ifprint: print("Timing 4", time() - t0, "s")
#     t0 = time()
#
#     # Uppercorner
#     C_NW_A_new = (P1A.reshape(chi * D * D, chi).T @ ncon([C_NW_B, E_N_A], ([1, -1], [-4, 1, -2, -3])).reshape(chi * D * D, chi)).T
#     C_NW_B_new = (P1B.reshape(chi * D * D, chi).T @ ncon([C_NW_A, E_N_B], ([1, -1], [-4, 1, -2, -3])).reshape(chi * D * D, chi)).T
#
#     # Lowercorner
#     C_SW_A_new = P2B.reshape(chi * D * D, chi).T @ ncon([C_SW_B, E_S_A], ([-1, 1], [1, -4, -2, -3])).reshape(chi * D * D, chi)
#     C_SW_B_new = P2A.reshape(chi * D * D, chi).T @ ncon([C_SW_A, E_S_B], ([-1, 1], [1, -4, -2, -3])).reshape(chi * D * D, chi)
#
#     # Edge
#     tensors = [P2A, P1B, E_W_B, A, A.conj()]
#     connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
#     con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
#     E_W_A_new = ncon(tensors, connects, con_order)
#     tensors = [P2B, P1A, E_W_A, B, B.conj()]
#     connects = [[1, 7, 6, -1], [2, 8, 5, -2], [1, 2, 3, 4], [7, -3, 8, 3, 9], [6, -4, 5, 4, 9]]
#     con_order = [2, 8, 3, 5, 4, 9, 1, 7, 6]
#     E_W_B_new = ncon(tensors, connects, con_order)
#
#     E_W_A = E_W_A_new / norm(E_W_A_new)
#     E_W_B = E_W_B_new / norm(E_W_B_new)
#     C_NW_A = C_NW_A_new / norm(C_NW_A_new)
#     C_SW_B = C_SW_B_new / norm(C_SW_B_new)
#     C_NW_B = C_NW_B_new / norm(C_NW_B_new)
#     C_SW_A = C_SW_A_new / norm(C_SW_A_new)
#
#     if ifprint: print("\tTensors updated")
#
#     if ifprint: print("Timing 5", time() - t0, "s")
#     t0 = time()
#
#     tensors = [C_NW_A, A, E_N_B, C_SW_A, E_W_B, C_SE_A, E_S_B, C_NE_A, E_E_B, A.conj()]
#     connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     con_order = [4, 3, 2, 7, 10, 11, 16, 6, 5, 1, 14, 13, 9, 8, 15, 12]
#     rhoA = ncon(tensors, connects, con_order)
#
#     tensors = [C_NW_B, B, E_N_A, C_SW_B, E_W_A, C_SE_B, E_S_A, C_NE_B, E_E_A, B.conj()]
#     connects = [[2, 3], [8, 15, 12, 9, -1], [1, 2, 8, 11], [4, 5], [3, 4, 9, 10], [6, 7], [5, 6, 12, 13], [16, 1], [7, 16, 15, 14], [11, 14, 13, 10, -2]]
#     con_order = [1, 7, 5, 3, 2, 6, 13, 14, 4, 10, 11, 16, 9, 8, 12, 15]
#     rhoB = ncon(tensors, connects, con_order)
#
#     rhoA = rhoA / np.trace(rhoA)
#     rhoA = (rhoA + rhoA.conj().T) / 2
#     if ifprint: print("\t", rhoA)
#     rhoB = rhoB / np.trace(rhoB)
#     rhoB = (rhoB + rhoB.conj().T) / 2
#     if ifprint: print("\t", rhoB)
#
#     if ifprint: print("\tDensity matricies calculated")
#
#     if ifprint: print("Timing 6", time() - t0, "s")
#     t0 = time()
#
#     if ifprint: print("\t ---", time() - t0, "s")
#
#     return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B, 'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A, 'C_NW_B': C_NW_B, 'C_SW_A': C_SW_A, 'C_NE_A': C_NE_A, 'C_SE_B': C_SE_B, 'rhoA': rhoA, 'rhoB': rhoB}

def __CTMRG_chi_change(env, chi):
    envp = {}
    for key in list(env):
        envp[key] = env[key]
        if key[:2] == 'C_':
            if env[key].shape[0] > chi:
                envp[key] = env[key][:chi, :chi]
            else:
                buff = np.zeros((chi, chi), dtype=np.complex128)
                buff[:env[key].shape[0], :env[key].shape[1]] = env[key]
                envp[key] = buff
        if key[:2] == 'E_':
            if env[key].shape[0] > chi:
                envp[key] = env[key][:chi, :chi, :, :]
            else:
                buff = np.zeros((chi, chi, env[key].shape[-2], env[key].shape[-1]), dtype=np.complex128)
                buff[:env[key].shape[0], :env[key].shape[1], :env[key].shape[-2], :env[key].shape[-1]] = env[key]
                envp[key] = buff
    return envp

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

    tensors = [C_NW_A, A, E_N_B, E_N_A, C_SW_B, E_W_A, E_W_B, C_SE_A, E_S_B, E_S_A, C_NE_B, E_E_A, E_E_B, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), OA1, OB2, OA2, OB1]
    connects = [[3, 4], [21, 12, 13, 22, 43], [2, 3, 21, 25], [1, 2, 20, 26], [6, 7], [5, 6, 23, 27], [4, 5, 22, 24], [9, 10], [8, 9, 31, 30], [7, 8, 28, 29], [36, 1], [11, 36, 35, 34], [10, 11, 33, 32], [25, 17, 18, 24, 44], [20, 35, 15, 12, 37], [26, 34, 16, 17, 38], [13, 14, 28, 23, 41], [18, 19, 29, 27, 42], [15, 33, 31, 14, 39], [16, 32, 30, 19, 40], [43, 44], [41, 42], [39, 40], [37, 38]]
    con_order = [42, 44, 17, 3, 1, 37, 7, 40, 9, 19, 35, 31, 28, 2, 6, 23, 10, 33, 30, 39, 32, 20, 36, 8, 29, 14, 41, 27, 5, 25, 26, 34, 38, 4, 11, 15, 18, 24, 16, 21, 12, 43, 13, 22]
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

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


def CTMRGstepLtest(A, B, chi, Ac=None, Bc=None, maxiter=1000, env0={}, invprecision=1e-10, precision=1e-20, ifprint=False, ifrandom=False, tests1=[], tests2=[]):
    if np.any(Ac == None): Ac = A.conj()
    if np.any(Bc == None): Bc = B.conj()
    vals1 = [0] * len(tests1)
    vals2 = [0] * len(tests2)

    names1, names2 = [], []
    for test in tests1:
        names1.append(test['name'])
    for test in tests2:
        names2.append(test['name'])

    env = __CTMRG_chi_change(env0, chi)
    if ifrandom:
        chi0 = 2
        env = {'E_E_A': np.random.randn(chi0, chi0, B.shape[1], Bc.shape[1]) + 1j * np.random.randn(chi0, chi0, B.shape[1], Bc.shape[1]), 'E_E_B': np.random.randn(chi0, chi0, A.shape[1], Ac.shape[1]) + 1j * np.random.randn(chi0, chi0, A.shape[1], Ac.shape[1]), 'E_W_A': np.random.randn(chi0, chi0, B.shape[3], Bc.shape[3]) + 1j * np.random.randn(chi0, chi0, B.shape[3], Bc.shape[3]), 'E_W_B': np.random.randn(chi0, chi0, A.shape[3], Ac.shape[3]) + 1j * np.random.randn(chi0, chi0, A.shape[3], Ac.shape[3]), 'E_S_A': np.random.randn(chi0, chi0, B.shape[2], Bc.shape[2]) + 1j * np.random.randn(chi0, chi0, B.shape[2], Bc.shape[2]), 'E_S_B': np.random.randn(chi0, chi0, A.shape[2], Ac.shape[2]) + 1j * np.random.randn(chi0, chi0, A.shape[2], Ac.shape[2]), 'E_N_A': np.random.randn(chi0, chi0, B.shape[0], Bc.shape[0]) + 1j * np.random.randn(chi0, chi0, B.shape[0], Bc.shape[0]), 'E_N_B': np.random.randn(chi0, chi0, A.shape[0], Ac.shape[0]) + 1j * np.random.randn(chi0, chi0, A.shape[0], Ac.shape[0]),
               'C_NW_A': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_SW_B': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_NE_B': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_SE_A': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_NW_B': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_SW_A': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_NE_A': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0), 'C_SE_B': np.random.randn(chi0, chi0) + 1j * np.random.randn(chi0, chi0)}
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

        prevvals1, vals1 = vals1, [0]*len(vals1)
        prevvals2, vals2 = vals2, [0]*len(vals2)
        if len(tests1) > 0:
            normal = __CTMRG_Rho_11(env,A,B,Ac,Bc,np.eye(A.shape[-1]),np.eye(B.shape[-1]))
            for n,test in enumerate(tests1):
                vals1[n] = __CTMRG_Rho_11(env,A,B,Ac,Bc,test['A'],test['B'])/normal
            for n,test in enumerate(tests1):
                if iter != 0:
                    print(test['name'],":")
                    for i in range(len(vals1[n])):
                        print("\t",f'{np.real_if_close(vals1[n][i]):.5e}',"\terr =",'-inf' if np.abs(vals1[n][i]-prevvals1[n][i]) <= 0 else f'{np.log10(np.abs(vals1[n][i]-prevvals1[n][i])):.5f}')

        if len(tests2) > 0:
            normal = __CTMRG_Rho_21(env,A,B,Ac,Bc,np.eye(A.shape[-1]),np.eye(B.shape[-1]))
            for n,test in enumerate(tests2):
                vals2[n] = __CTMRG_Rho_21(env,A,B,Ac,Bc,test['A'],test['B'])/normal
            for n,test in enumerate(tests2):
                if iter != 0:
                    print(test['name'],":")
                    for i in range(len(vals2[n])):
                        print("\t",f'{np.real_if_close(vals2[n][i]):.5e}',"\terr =",'-inf' if np.abs(vals2[n][i]-prevvals2[n][i]) <= 0 else f'{np.log10(np.abs(vals2[n][i]-prevvals2[n][i])):.5f}')


        print("Errorz =", f'{errora:.5e}', "\t", f'{errorb:.5e}', "\t\t >", precision)
        if (errora < precision and errorb < precision and np.all(np.abs(np.array(vals1)-np.array(prevvals1)) < precision) and np.all(np.abs(np.array(vals2)-np.array(prevvals2)) < precision)) or (preverrora < errora and preverrorb < errorb and errorb < 1e-8 and errora < 1e-5):
            break

    errors1, errors2 = np.array(vals1)-np.array(prevvals1), np.array(vals2)-np.array(prevvals2)
    env['names1'], env['names2'] = np.array(names1), np.array(names2)
    env['vals1'], env['vals2'] = np.array(vals1), np.array(vals2)
    env['errors1'], env['errors2'] = np.array(errors1), np.array(errors2)
    env['error'] = np.sqrt(norm(env["rhoA"] - prevrhoA) ** 2 / 2 + norm(env["rhoB"] - prevrhoB) ** 2 / 2)
    return env

def __CTMRT_left_test(A, B, chimax, Ac, Bc, env0={}, invprecision=1e-10, ifprint=False):
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

    if ifprint: print("Timing 1", f'{time() - t0:.5e}', "s")
    t0 = time()

    tensors = [C_NW_A, E_N_B, E_N_A, E_W_B, C_NE_B, E_E_A, A, Ac, B, Bc]
    connects = [[15, 16], [14, 15, 11, 12], [13, 14, 6, 3], [16, -4, 9, 10], [17, 13], [-1, 17, 4, 7], [11, 8, -5, 9, 2], [12, 5, -6, 10, 2], [6, 7, -2, 8, 1], [3, 4, -3, 5, 1]]
    con_order = [15, 13, 16, 12, 10, 11, 9, 2, 17, 3, 4, 6, 7, 1, 14, 5, 8]
    UpperHalfA = ncon(tensors, connects, con_order)

    tensors = [C_SW_B, E_W_A, C_SE_A, E_S_B, E_S_A, E_E_B, B, Bc, A, Ac]
    connects = [[13, 14], [-4, 13, 9, 10], [16, 17], [15, 16, 5, 6], [14, 15, 12, 11], [17, -1, 3, 4], [-5, 8, 11, 9, 1], [-6, 7, 12, 10, 1], [-2, 4, 6, 8, 2], [-3, 3, 5, 7, 2]]
    con_order = [13, 14, 16, 9, 11, 17, 6, 4, 10, 12, 1, 5, 3, 2, 15, 8, 7]
    BottomHalfA = ncon(tensors, connects, con_order)

    tensors = [C_NW_B, E_N_A, E_N_B, E_W_A, C_NE_A, E_E_B, B, Bc, A, Ac]
    connects = [[15, 16], [14, 15, 11, 12], [13, 14, 6, 3], [16, -4, 9, 10], [17, 13], [-1, 17, 4, 7], [11, 8, -5, 9, 2], [12, 5, -6, 10, 2], [6, 7, -2, 8, 1], [3, 4, -3, 5, 1]]
    con_order = [15, 16, 12, 10, 17, 13, 3, 4, 6, 7, 1, 11, 9, 2, 14, 5, 8]
    UpperHalfB = ncon(tensors, connects, con_order)

    tensors = [C_SW_A, E_W_B, C_SE_B, E_S_A, E_S_B, E_E_A, A, Ac, B, Bc]
    connects = [[13, 14], [-4, 13, 9, 10], [16, 17], [15, 16, 5, 6], [14, 15, 12, 11], [17, -1, 3, 4], [-5, 8, 11, 9, 1], [-6, 7, 12, 10, 1], [-2, 4, 6, 8, 2], [-3, 3, 5, 7, 2]]
    con_order = [13, 16, 14, 9, 11, 17, 6, 4, 10, 12, 1, 5, 3, 2, 15, 8, 7]
    BottomHalfB = ncon(tensors, connects, con_order)

    if ifprint: print("\tHalves calculated")

    if ifprint: print("Timing 2", f'{time() - t0:.5e}', "s")
    t0 = time()

    # -R=
    def HalfQR(H):
        ifqr = False
        Hp = H.reshape(H.shape[0]*H.shape[1]*H.shape[2],H.shape[3]*H.shape[4]*H.shape[5])
        R = (Hp if not ifqr else qr(Hp, mode='r')[0]) / norm(Hp)
        return R.reshape(H.shape[0]*H.shape[1]*H.shape[2],H.shape[3],H.shape[4],H.shape[5])

    RUA = HalfQR(UpperHalfA)
    RUB = HalfQR(UpperHalfB)
    RBA = HalfQR(BottomHalfA)
    RBB = HalfQR(BottomHalfB)


    # UpperHalfA = UpperHalfA.reshape(E_E_A.shape[1] * B.shape[2] ** 2, E_W_B.shape[0] * A.shape[2] ** 2)
    # UpperHalfB = UpperHalfB.reshape(E_E_B.shape[1] * A.shape[2] ** 2, E_W_A.shape[0] * B.shape[2] ** 2)
    # BottomHalfA = BottomHalfA.reshape(E_W_A.shape[0] * A.shape[0] ** 2, E_E_B.shape[1] * B.shape[0] ** 2)
    # BottomHalfB = BottomHalfB.reshape(E_W_B.shape[0] * B.shape[0] ** 2, E_E_A.shape[1] * A.shape[0] ** 2)
    # ifqr = False
    # RUA = (UpperHalfA if not ifqr else qr(UpperHalfA, mode='r')[0]) / norm(UpperHalfA)
    # RUB = (UpperHalfB if not ifqr else qr(UpperHalfB, mode='r')[0]) / norm(UpperHalfB)
    # RBA = (BottomHalfA if not ifqr else qr(BottomHalfA, mode='r')[0]) / norm(BottomHalfA)
    # RBB = (BottomHalfB if not ifqr else qr(BottomHalfB, mode='r')[0]) / norm(BottomHalfB)

    if ifprint: print("\tQR done")

    if ifprint: print("Timing 3", f'{time() - t0:.5e}', "s")
    t0 = time()

    # =P-
    def create_isometries(X1, X2):
        # -X1= to ta górna macierz w CTMRG_BY_QR (R bez tyldy)
        # P1 to trójkąt podstawą zwrócony w górę
        T1 = X1.reshape(X1.shape[0],X1.shape[1]*X1.shape[2]*X1.shape[3])
        T2 = X2.reshape(X2.shape[0],X2.shape[1]*X2.shape[2]*X2.shape[3])
        u, s, vh = svd(T1 @ T2.T)

        sd = [1]
        maxnonzeroindex = 1
        for i in range(1,len(s)):
            if s[i] > s[0] * invprecision and i < chimax:
                maxnonzeroindex = i
                sd.append(1 / np.sqrt(s[i] / s[0]))
            else: break

        # print("\t\t", maxnonzeroindex, " / ", chimax, "\t", str(s[chimax] / s[0]), "" if maxnonzeroindex < chimax else "\t!!!!!!!")
        try:
            print("\t\tchi =",maxnonzeroindex+1,"\t\t",f'{s[maxnonzeroindex+1]/s[0]:.5e}')
        except:
            print("\t\tchi =",maxnonzeroindex+1,"\t\t",f'{s[-1]/s[0]:.5e}',"\t P")
        if ifprint: print("\t", sd)

        sd = np.diag(sd)
        P1 = ncon([X2, sd @ (vh.conj())[:sd.shape[0], :]], ([1, -1, -2, -3], [-4, 1]))
        P2 = ncon([X1, sd @ (u.conj().T)[:sd.shape[0], :]], ([1, -1, -2, -3], [-4, 1]))
        return P1, P2

    P1A, P2A = create_isometries(RUB, RBB)
    P1B, P2B = create_isometries(RUA, RBA)

    if ifprint: print("\tIsometries calculated")

    if ifprint: print("Timing 4", f'{time() - t0:.5e}', "s")
    t0 = time()

    # Uppercorner
    C_NW_A_new = (P1B.reshape(P1B.shape[0] * P1B.shape[1] ** 2, P1B.shape[-1]).T @ ncon([C_NW_B, E_N_A], ([1, -1], [-4, 1, -2, -3])).reshape(C_NW_B.shape[1] * E_N_A.shape[2] ** 2, E_N_A.shape[0])).T
    C_NW_B_new = (P1A.reshape(P1A.shape[0] * P1A.shape[1] ** 2, P1A.shape[-1]).T @ ncon([C_NW_A, E_N_B], ([1, -1], [-4, 1, -2, -3])).reshape(C_NW_A.shape[1] * E_N_B.shape[2] ** 2, E_N_B.shape[0])).T

    # Lowercorner
    C_SW_A_new = P2A.reshape(P2A.shape[0] * P2A.shape[1] ** 2, P2A.shape[-1]).T @ ncon([C_SW_B, E_S_A], ([-1, 1], [1, -4, -2, -3])).reshape(C_SW_B.shape[0] * E_S_A.shape[2] ** 2, E_S_A.shape[1])
    C_SW_B_new = P2B.reshape(P2B.shape[0] * P2B.shape[1] ** 2, P2B.shape[-1]).T @ ncon([C_SW_A, E_S_B], ([-1, 1], [1, -4, -2, -3])).reshape(C_SW_A.shape[0] * E_S_B.shape[2] ** 2, E_S_B.shape[1])

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

    if ifprint: print("Timing 5", f'{time() - t0:.5e}', "s")
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

    if ifprint: print("Timing 6", f'{time() - t0:.5e}', "s")
    t0 = time()

    if ifprint: print("\t ---", f'{time() - t0:.5e}', "s")

    return {'E_E_A': E_E_A, 'E_E_B': E_E_B, 'E_W_A': E_W_A, 'E_W_B': E_W_B, 'E_S_A': E_S_A, 'E_S_B': E_S_B, 'E_N_A': E_N_A, 'E_N_B': E_N_B, 'C_NW_A': C_NW_A, 'C_SW_B': C_SW_B, 'C_NE_B': C_NE_B, 'C_SE_A': C_SE_A, 'C_NW_B': C_NW_B, 'C_SW_A': C_SW_A, 'C_NE_A': C_NE_A, 'C_SE_B': C_SE_B, 'rhoA': rhoA, 'rhoB': rhoB}

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

def __CTMRG_Rho_21(env,A,B,Ac,Bc,OPA,OPB):
    # AB pionowe (A na górze)
    # BA pionowe (B na górze)
    # AB poziome (A po lewej)
    # BA poziome (B po lewej)
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

    tensors = [C_NW_A, E_N_B, E_W_B, Ac, OPA, A, E_W_A, Bc, OPB, B, C_SW_B, E_S_A, C_NE_A, E_E_B, E_E_A, C_SE_B]
    connects = [[12, 13], [11, 12, 10, 9], [13, 14, 24, 23], [9, 21, 2, 23, 6], [6, 5], [10, 22, 1, 24, 5], [14, 15, 28, 27], [2, 25, 4, 27, 8], [8, 7], [1, 26, 3, 28, 7], [15, 16], [16, 17, 3, 4], [18, 11], [20, 18, 22, 21], [19, 20, 26, 25], [17, 19]]
    con_order = [5, 13, 11, 12, 15, 7, 19, 24, 10, 16, 28, 3, 27, 4, 8, 17, 26, 25, 20, 14, 18, 22, 1, 23, 9, 6, 2, 21]
    o1 = ncon(tensors, connects, con_order)

    tensors = [C_NE_B, E_E_A, E_E_B, C_SE_A, C_NW_B, E_N_A, E_W_A, Bc, OPB, B, E_W_B, Ac, OPA, A, C_SW_A, E_S_B]
    connects = [[1, 14], [3, 1, 22, 21], [2, 3, 26, 25], [20, 2], [15, 16], [14, 15, 13, 12], [16, 17, 24, 23], [12, 21, 5, 23, 9], [9, 8], [13, 22, 4, 24, 8], [17, 18, 28, 27], [5, 25, 7, 27, 11], [11, 10], [4, 26, 6, 28, 10], [18, 19], [19, 20, 6, 7]]
    con_order = [1, 14, 20, 8, 19, 16, 11, 2, 26, 6, 22, 13, 25, 7, 10, 18, 28, 27, 17, 3, 15, 4, 24, 21, 12, 9, 5, 23]
    o2 = ncon(tensors, connects, con_order)

    tensors = [C_NW_A, E_N_B, E_N_A, C_NE_B, E_W_B, Ac, OPA, A, Bc, OPB, B, E_E_A, C_SW_A, E_S_B, E_S_A, C_SE_B]
    connects = [[12, 13], [11, 12, 10, 9], [14, 11, 7, 8], [15, 14], [13, 16, 28, 27], [9, 25, 4, 27, 6], [6, 5], [10, 26, 3, 28, 5], [8, 23, 1, 25, 18], [18, 19], [7, 24, 2, 26, 19], [17, 15, 24, 23], [16, 20], [20, 21, 3, 4], [21, 22, 2, 1], [22, 17]]
    con_order = [12, 22, 14, 5, 19, 20, 15, 13, 10, 28, 7, 24, 8, 23, 18, 17, 2, 1, 21, 11, 16, 26, 3, 9, 27, 6, 25, 4]
    o3 = ncon(tensors, connects, con_order)

    tensors = [C_SW_B, E_S_A, E_S_B, C_SE_A, C_NW_B, E_N_A, E_N_B, C_NE_A, E_W_A, Bc, OPB, B, Ac, OPA, A, E_E_B]
    connects = [[19, 1], [1, 2, 6, 7], [2, 3, 5, 4], [3, 20], [15, 16], [14, 15, 13, 12], [17, 14, 10, 11], [18, 17], [16, 19, 28, 27], [12, 25, 7, 27, 9], [9, 8], [13, 26, 6, 28, 8], [11, 23, 4, 25, 21], [21, 22], [10, 24, 5, 26, 22], [20, 18, 24, 23]]
    con_order = [1, 16, 19, 8, 18, 21, 20, 3, 5, 24, 6, 28, 4, 23, 22, 17, 10, 11, 14, 2, 15, 13, 26, 7, 27, 9, 25, 12]
    o4 = ncon(tensors, connects, con_order)

    return np.array([o1, o2, o3, o4])

def __CTMRG_Rho_11(env,A,B,Ac,Bc,OPA,OPB):
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

    tensors = [C_NW_A, E_N_B, E_W_B, Ac, OPA, A, C_SE_A, C_NE_A, E_E_B, C_SW_A, E_S_B]
    connects = [[8, 9], [7, 8, 6, 5], [9, 10, 18, 17], [5, 15, 2, 17, 4], [4, 3], [6, 16, 1, 18, 3], [13, 14], [11, 7], [14, 11, 16, 15], [10, 12], [12, 13, 1, 2]]
    con_order = [9, 13, 12, 11, 3, 14, 8, 16, 1, 10, 18, 7, 6, 17, 5, 4, 2, 15]
    oA = ncon(tensors, connects, con_order)

    tensors = [C_NE_B, E_E_A, C_SW_B, E_S_A, C_NW_B, E_N_A, E_W_A, Bc, OPB, B, C_SE_B]
    connects = [[1, 11], [4, 1, 16, 15], [14, 2], [2, 3, 5, 6], [12, 13], [11, 12, 10, 9], [13, 14, 18, 17], [9, 15, 6, 17, 8], [8, 7], [10, 16, 5, 18, 7], [3, 4]]
    con_order = [1, 4, 12, 11, 7, 16, 10, 2, 14, 3, 13, 5, 18, 15, 9, 8, 6, 17]
    oB = ncon(tensors, connects, con_order)

    return np.array([oA, oB])


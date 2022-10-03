from ncon import ncon
import numpy as np
from numpy.linalg import qr, svd, norm, pinv

import Tools


def __copy(PEPS):
    PEPS = dict(PEPS)
    PEPS2 = {}
    for key in PEPS.keys():
        PEPS2[key] = PEPS[key]
    return PEPS2


def __rot(PEPS):
    def rototo(A):
        return A.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1) / norm(A)

    PEPS2 = __copy(PEPS)

    # if len(dict['A'].shape) == 5 or not (dict['A'].shape[0] == dict['A'].shape[1]) or not (
    #         dict['A'].shape[1] == dict['A'].shape[2]) or not (dict['A'].shape[2] == dict['A'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['A'].shape}")
    # if len(dict['B'].shape) == 5 or not (dict['B'].shape[0] == dict['B'].shape[1] == dict['B'].shape[2] == \
    #                                      dict['B'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['B'].shape}")
    PEPS2['A'] = rototo(PEPS2['A'])
    PEPS2['B'] = rototo(PEPS2['B'])
    return PEPS2


def __rotinv(PEPS):
    def rototo(A):
        return A.swapaxes(0, 1).swapaxes(0, 2).swapaxes(0, 3) / norm(A)

    PEPS2 = __copy(PEPS)

    # if len(dict['A'].shape) == 5 or not (dict['A'].shape[0] == dict['A'].shape[1]) or not (
    #         dict['A'].shape[1] == dict['A'].shape[2]) or not (dict['A'].shape[2] == dict['A'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['A'].shape}")
    # if len(dict['B'].shape) == 5 or not (dict['B'].shape[0] == dict['B'].shape[1] == dict['B'].shape[2] == \
    #                                      dict['B'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['B'].shape}")
    PEPS2['A'] = rototo(PEPS2['A'])
    PEPS2['B'] = rototo(PEPS2['B'])
    return PEPS2


def __step(PEPS, precision=1e-15, ifsvdu=False, maxiter=200, ifprint=True, precisionspeed=0, iffast=False):
    A = PEPS['A']
    B = PEPS['B']
    GA = PEPS['GA']
    GB = PEPS['GB']

    d = A.shape[-1]
    D = A.shape[0]
    r = GA.shape[-1]
    Dr, DDr, DDDr = D * r, D * D * r, D * D * D * r
    Dd, DDd, DDDd = D * d, D * D * d, D * D * D * d

    # RA, RB -<|-
    AG = ncon([A, GA], ([-1, -2, -4, -5, 1], [1, -6, -3])).reshape(D, Dr, D, D, d)
    Q, R = __qr(AG.swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4).reshape(DDDd, Dr), mode='reduced')
    QA = Q.reshape(D, D, D, d, Dr).swapaxes(3, 4).swapaxes(2, 3).swapaxes(1, 2)
    RA = R

    BG = ncon([B, GB], ([-1, -2, -3, -4, 1], [1, -6, -5])).reshape(D, D, D, Dr, d)
    Q, R = __qr(BG.swapaxes(3, 4).reshape(DDDd, Dr), mode='reduced')
    QB = Q.reshape(D, D, D, d, Dr).swapaxes(3, 4)
    RB = R

    # MA, MB -|>-
    MA, MB = Tools.truncate2(RA @ RB.T, D)
    MB = MB.T

    def calc_JA(MB):
        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB, QB.conj(), RA, RB, MB.conj()]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, 41, 24, 19, 7], [18, -1, 25, 20, 7], [12, 36, 37, 40, 8], [11, 35, 38, 42, 8], [41, 39], [40, 39], [42, -2]]
        con_order = [15, 16, 2, 34, 33, 32, 6, 42, 35, 31, 30, 5, 39, 10, 9, 1, 28, 29, 4, 40, 21, 23, 22, 3, 36, 8, 26, 27, 14, 13, 37, 38, 12, 11, 17, 24, 41, 19, 18, 25, 7, 20]
        return ncon(tensors, connects, con_order)

    def calc_gA(MB):
        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB, QB.conj(), MB.conj(), MB]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, -1, 24, 19, 7], [18, -3, 25, 20, 7], [12, 36, 37, 40, 8], [11, 35, 38, 39, 8], [39, -4], [40, -2]]
        con_order = [15, 16, 2, 10, 9, 1, 34, 33, 32, 6, 40, 31, 30, 5, 36, 28, 29, 4, 14, 13, 39, 35, 8, 26, 27, 12, 11, 37, 38, 17, 24, 21, 23, 22, 3, 20, 18, 25, 19, 7]
        return ncon(tensors, connects, con_order)

    def calc_JB(MA):
        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB, QB.conj(), RA, RB, MA.conj()]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, 41, 24, 19, 7], [18, 42, 25, 20, 7], [12, 36, 37, 40, 8], [11, 35, 38, -2, 8], [41, 39], [40, 39], [42, -1]]
        con_order = [15, 16, 2, 21, 23, 22, 3, 28, 29, 4, 10, 9, 1, 42, 31, 30, 5, 20, 34, 33, 32, 6, 26, 27, 39, 41, 19, 7, 14, 13, 17, 18, 25, 24, 12, 40, 37, 36, 11, 38, 8, 35]
        return ncon(tensors, connects, con_order)

    def calc_gB(MA):
        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB, QB.conj(), MA.conj(), MA]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, 40, 24, 19, 7], [18, 39, 25, 20, 7], [12, 36, 37, -2, 8], [11, 35, 38, -4, 8], [39, -3], [40, -1]]
        con_order = [15, 16, 2, 31, 30, 5, 28, 29, 4, 21, 23, 22, 3, 10, 9, 1, 26, 27, 39, 14, 13, 20, 40, 34, 33, 32, 6, 36, 19, 7, 25, 24, 17, 18, 12, 37, 11, 38, 35, 8]
        return ncon(tensors, connects, con_order)

    def calc_error(MA, MB):
        def err(MAMBT):
            if np.all(MAMBT == 0):
                W = RA @ RB.T
            else:
                W = MAMBT - RA @ RB.T
            tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB, QB.conj(), W, W.conj()]
            connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, 40, 24, 19, 7], [18, 41, 25, 20, 7], [12, 36, 37, 39, 8], [11, 35, 38, 42, 8], [40, 39], [41, 42]]
            con_order = [15, 16, 2, 28, 29, 4, 34, 33, 32, 6, 21, 23, 22, 3, 20, 35, 42, 41, 10, 9, 1, 31, 30, 5, 14, 13, 18, 11, 26, 27, 39, 25, 38, 12, 36, 8, 37, 17, 19, 7, 24, 40]
            return ncon(tensors, connects, con_order)
        MAMBT = MA@MB.T
        return err(MAMBT)/err(np.zeros_like(MAMBT))

    def calc_MA(gA, JA, rc=1e-8):
        g = gA.reshape(DDr, DDr).T
        g = (g + g.conj().T) / 2
        return (pinv(g, rcond=rc, hermitian=True) @ JA.reshape(DDr)).reshape(Dr, D)

    def calc_MB(gB, JB, rc=1e-8):
        g = gB.reshape(DDr, DDr).T
        g = (g + g.conj().T) / 2
        return (pinv(g, rcond=rc, hermitian=True) @ JB.reshape(DDr)).reshape(D, Dr).T

    SVDUerror = calc_error(MA, MB)
    if ifprint: print("\tSVDUerror:", np.abs(SVDUerror))
    NTUerror = SVDUerror
    if ifsvdu or SVDUerror < precision:
        return {'A': ncon([QA, MA], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': NTUerror}

    if iffast: rcs = [1e-8]
    else: rcs = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    MA_best, MB_best, error_best, error_prev = MA, MB, SVDUerror, SVDUerror
    for iter in range(maxiter):
        error_prev = error_best

        gA = calc_gA(MB_best)
        JA = calc_JA(MB_best)
        for rc in rcs:
            MA_temp = calc_MA(gA, JA, rc=rc)
            error_temp = calc_error(MA_temp, MB_best)
            if ifprint: print("\t\t\t", rc, " \t", np.abs(error_temp))
            if np.abs(error_temp) < np.abs(error_best):
                MA_best = MA_temp
                error_best = error_temp
        if ifprint: print("\t\t", iter, "A error =", np.abs(error_best))


        gB = calc_gB(MA_best)
        JB = calc_JB(MA_best)
        for rc in rcs:
            MB_temp = calc_MB(gB, JB, rc=rc)
            error_temp = calc_error(MA_best, MB_temp)
            if ifprint: print("\t\t\t", rc, " \t", np.abs(error_temp))
            if np.abs(error_temp) < np.abs(error_best):
                MB_best = MB_temp
                error_best = error_temp
        if ifprint: print("\t\t", iter, "B error =", np.abs(error_best))


        if (1+precisionspeed)*error_best >= error_prev:
            if ifprint: print("\tNTUerror:", np.abs(error_best), "\t\t\tCONVERGENCE")
            return {'A': ncon([QA, MA_best], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB_best], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': error_best}
        if error_best <= precision:
            if ifprint: print("\tNTUerror:", np.abs(error_best), "\t\t\tPRECISION")
            return {'A': ncon([QA, MA_best], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB_best], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': error_best}

    return {'A': ncon([QA, MA_best], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB_best], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': error_best}


def __NTUobs(A,B,Aop,Bop):
    tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), A, A.conj(), B, B.conj(), Aop, Bop]
    connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, 39, 24, 19, 7], [18, 41, 25, 20, 40], [12, 36, 37, 39, 8], [11, 35, 38, 41, 42], [40, 7], [42, 8]]
    con_order = [15, 16, 2, 28, 29, 4, 34, 33, 32, 6, 21, 23, 22, 3, 20, 35, 42, 41, 10, 9, 1, 31, 30, 5, 14, 13, 18, 11, 26, 27, 39, 25, 38, 12, 36, 8, 37, 17, 19, 7, 24, 40]
    return ncon(tensors, connects, con_order)

def __qr(A, mode='reduced'):
    d1, d2 = A.shape[0], A.shape[1]
    if A.shape[0] > A.shape[1]:
        return qr(A, mode=mode)
    Q, R = qr(A, mode=mode)
    Qbig, Rbig = np.zeros((d1,d2),dtype=np.complex128), np.zeros((d2,d2),dtype=np.complex128)
    Qbig[:d1, :d1] = Q
    Rbig[:d1, :d2] = R
    return Qbig, Rbig

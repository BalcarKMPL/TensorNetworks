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


def __step(PEPS, GATES, precision=1e-15, ifsvdu=False, maxiter=200, ifprint=True, precisionspeed=0, iffast=False):
    A = PEPS['A']  # (D0, D1, D2, D3, d)
    B = PEPS['B']  # (D2, D3, D0, D1, d)
    GA = GATES['GA']
    GB = GATES['GB']

    r = GA.shape[-1]
    d = A.shape[-1]
    D0 = A.shape[0]
    D1 = A.shape[1]
    D2 = A.shape[2]
    D3 = A.shape[3]
    D01 = D0 * D1
    D02 = D0 * D2
    D03 = D0 * D3
    D12 = D1 * D2
    D13 = D1 * D3
    D23 = D2 * D3
    D023 = D0 * D2 * D3
    D123 = D1 * D2 * D3
    D013 = D0 * D1 * D3
    D012 = D0 * D1 * D2
    D0123 = D0 * D1 * D2 * D3
    D0d = D0 * d
    D1d = D1 * d
    D2d = D2 * d
    D3d = D3 * d
    D01d = D0 * D1 * d
    D02d = D0 * D2 * d
    D03d = D0 * D3 * d
    D12d = D1 * D2 * d
    D13d = D1 * D3 * d
    D23d = D2 * D3 * d
    D023d = D0 * D2 * D3 * d
    D123d = D1 * D2 * D3 * d
    D013d = D0 * D1 * D3 * d
    D012d = D0 * D1 * D2 * d
    D0123d = D0 * D1 * D2 * D3 * d
    D0r = D0 * r
    D1r = D1 * r
    D2r = D2 * r
    D3r = D3 * r
    D01r = D0 * D1 * r
    D02r = D0 * D2 * r
    D03r = D0 * D3 * r
    D12r = D1 * D2 * r
    D13r = D1 * D3 * r
    D23r = D2 * D3 * r
    D023r = D0 * D2 * D3 * r
    D123r = D1 * D2 * D3 * r
    D013r = D0 * D1 * D3 * r
    D012r = D0 * D1 * D2 * r
    D0123r = D0 * D1 * D2 * D3 * r
    D1r = D1 * r

    An = ncon([A, GA], ([-1, -2, -4, -5, 1], [1, -6, -3])).reshape(D0, D1r, D2, D3, d).swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4).reshape(D023d, D1r)
    Bn = ncon([B, GB], ([-1, -2, -3, -4, 1], [1, -6, -5])).reshape(D2, D3, D0, D1r, d).swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0).reshape(D1r, D023d)

    UA, sA, VAh = svd(An, full_matrices=False)
    UB, sB, VBh = svd(Bn, full_matrices=False)

    U, s, Vh = svd(np.diag(sA) @ VAh @ UB @ np.diag(sB), full_matrices=False)
    # U = U[:, :D1]
    # Vh = Vh[:D1, :]
    # s = s[:D1]

    # RA, RB -<|-
    QA = UA.reshape(D0, D2, D3, d, D1r).swapaxes(3, 4).swapaxes(2, 3).swapaxes(1, 2)
    QB = VBh.reshape(D1r, D2, D3, D0, d).swapaxes(1, 0).swapaxes(2, 1).swapaxes(3, 2)
    RA = U @ np.diag(np.sqrt(s))
    RB = Vh.T @ np.diag(np.sqrt(s))

    # MA, MB -|>-
    MA, MB = Tools.truncate2(RA @ RB.T, D1)
    MB = MB.T

    def calc_g():
        BB = B.swapaxes(0, 1).swapaxes(1, 2)
        BBB = BB.reshape(D03, D12d)
        UB = (BBB @ BBB.conj().T).reshape(D3, D0, D3, D0).swapaxes(1, 2).reshape(D3 ** 2, D0 ** 2)
        AA = A.swapaxes(4, 3).swapaxes(3, 2)
        AAA = AA.reshape(D01d, D23).T
        UA = (AAA @ AAA.conj().T).reshape(D2, D3, D2, D3).swapaxes(1, 2).reshape(D2 ** 2, D3 ** 2)

        BB = B.reshape(D23, D01d)
        DB = (BB @ BB.conj().T).reshape(D2, D3, D2, D3).swapaxes(1, 2).reshape(D2 ** 2, D3 ** 2)
        AA = A.swapaxes(3, 2).swapaxes(2, 1)
        AAA = AA.reshape(D03, D12d)
        DA = (AAA @ AAA.conj().T).reshape(D0, D3, D0, D3).swapaxes(1, 2).reshape(D0 ** 2, D3 ** 2)

        # BB = B.swapaxes(1, 0)
        # BBB = BB.reshape(D3, D012d)
        # SB = (BBB @ BBB.conj().T).reshape(D3 ** 2)
        # AA = A.swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0)
        # AAA = AA.reshape(D3, D012d)
        # SA = (AAA @ AAA.conj().T).reshape(D3 ** 2)

        SA = np.trace(UA.reshape(D2, D2, D3 ** 2))
        SB = np.trace(UB.T.reshape(D0, D0, D3 ** 2))

        QAp = ncon([QA, SB.reshape(D3, D3)], ([-1, -2, -3, 1, -5], [1, -4]))
        QAq = (QAp.reshape(D012r, D3d) @ QA.reshape(D012r, D3d).conj().T)
        QAr = QAq.reshape(D0, D1r, D2, D0, D1r, D2)
        QAs = QAr.swapaxes(1, 3).swapaxes(2, 3).swapaxes(3, 4)
        QAw = QAs.reshape(D0 ** 2, D1r ** 2, D2 ** 2)
        QBp = ncon([QB, SA.reshape(D3, D3)], ([-1, 1, -3, -4, -5], [1, -2])).swapaxes(1, 2).swapaxes(2, 3)
        QBq = (QBp.reshape(D012r, D3d) @ QB.swapaxes(1, 2).swapaxes(2, 3).reshape(D012r, D3d).conj().T)
        QBr = QBq.reshape(D2, D0, D1r, D2, D0, D1r)
        QBs = QBr.swapaxes(1, 3).swapaxes(2, 3).swapaxes(3, 4)
        QBw = QBs.reshape(D2 ** 2, D0 ** 2, D1r ** 2)

        return ncon([QAw, QBw, UA, UB, DA, DB], ([3, -1, 6], [5, 4, -2], [5, 2], [2, 3], [4, 1], [6, 1])).reshape(D1r, D1r, D1r, D1r).swapaxes(1, 2)

    def calc_J():
        return ncon([g, RA, RB], ([4, 3, -1, -2], [4, 2], [3, 2])).reshape(D1r, D1r)

    g = calc_g()
    J = calc_J()

    def calc_JA(MB):
        return ncon([J, MB.conj()], ([-1, 1], [1, -2])).reshape(D1r * D1)
        # return ncon([g, MB.conj(), RA, RB], ([4, 3, -1, 1], [1, -2], [4, 2], [3, 2])).reshape(D1r * D1)

    def calc_gA(MB):
        return ncon([g, MB, MB.conj()], ([-1, 1, -3, 2], [1, -2], [2, -4]))

    def calc_JB(MA):
        return ncon([J, MA.conj()], ([1, -2], [1, -1])).reshape(D1r * D1)
        # return ncon([g, MA.conj(), RA, RB], ([4, 3, 1, -2], [1, -1], [4, 2], [3, 2])).reshape(D1r * D1)

    def calc_gB(MA):
        return ncon([g, MA, MA.conj()], ([1, -2, 2, -4], [1, -1], [2, -3]))

    def calc_error(MA, MB):
        _m = (MA @ MB.T).reshape(D1r ** 2)
        _r = (RA @ RB.T).reshape(D1r ** 2)
        _g = g.reshape(D1r ** 2, D1r ** 2)
        _J = J.reshape(D1r ** 2)
        # J = rg
        # return (mgm-Jm-mJ+Jr)
        return ((_m - _r) @ _g @ (_m - _r).conj()) / (_r @ _g @ _r.conj())

    def calc_MA(gA, JA, rc=1e-8):
        g = gA.reshape(D1r * D1, D1r * D1).T
        g = (g + g.conj().T) / 2
        return (pinv(g, rcond=rc, hermitian=True) @ JA.reshape(D1r * D1)).reshape(D1r, D1)

    def calc_MB(gB, JB, rc=1e-8):
        g = gB.reshape(D1r * D1, D1r * D1).T
        g = (g + g.conj().T) / 2
        return (pinv(g, rcond=rc, hermitian=True) @ JB.reshape(D1r * D1)).reshape(D1, D1r).T

    SVDUerror = calc_error(MA, MB)
    if ifprint: print("\tSVDUerror:", f'{np.abs(SVDUerror):.16e}')
    NTUerror = SVDUerror
    if ifsvdu or SVDUerror < precision:
        return {'A': ncon([QA, MA], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': NTUerror}

    if iffast:
        rcs = [1e-8]
    else:
        rcs = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    MA_best, MB_best, error_best, error_prev = MA, MB, SVDUerror, SVDUerror
    for iter in range(maxiter):
        error_prev = error_best

        gA = calc_gA(MB_best)
        JA = calc_JA(MB_best)
        for rc in rcs:
            MA_temp = calc_MA(gA, JA, rc=rc)
            error_temp = calc_error(MA_temp, MB_best)
            if ifprint: print("\t\t\t", rc, " \t", f'{np.abs(error_temp):.16e}')
            if np.abs(error_temp) < np.abs(error_best):
                MA_best = MA_temp
                error_best = error_temp
        if ifprint: print("\t\t", iter, "A error =", f'{np.abs(error_best):.16e}')

        gB = calc_gB(MA_best)
        JB = calc_JB(MA_best)
        for rc in rcs:
            MB_temp = calc_MB(gB, JB, rc=rc)
            error_temp = calc_error(MA_best, MB_temp)
            if ifprint: print("\t\t\t", rc, " \t", f'{np.abs(error_temp):.16e}')
            if np.abs(error_temp) < np.abs(error_best):
                MB_best = MB_temp
                error_best = error_temp
        if ifprint: print("\t\t", iter, "B error =", f'{np.abs(error_best):.16e}')

        if (1 + precisionspeed) * error_best >= error_prev:
            if ifprint: print("\tNTUerror:", f'{np.abs(error_best):.16e}', "\t\t\tCONVERGENCE")
            return {'A': ncon([QA, MA_best], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB_best], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': error_best}
        if error_best <= precision:
            if ifprint: print("\tNTUerror:", f'{np.abs(error_best):.16e}', "\t\t\tPRECISION")
            return {'A': ncon([QA, MA_best], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB_best], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': error_best}

    return {'A': ncon([QA, MA_best], ([-1, 1, -3, -4, -5], [1, -2])), 'B': ncon([QB, MB_best], ([-1, -2, -3, 1, -5], [1, -4])), 'GA': GA, 'GB': GB, 'SVDUerror': SVDUerror, 'NTUerror': error_best}


def __NTUobs1(A, B, Aop, Bop):
    tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), A, A.conj(), B, B.conj(), Aop, Bop]
    connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1], [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4], [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6], [17, 39, 24, 19, 7], [18, 41, 25, 20, 40], [12, 36, 37, 39, 8], [11, 35, 38, 41, 42], [40, 7], [42, 8]]
    con_order = [15, 16, 2, 28, 29, 4, 34, 33, 32, 6, 21, 23, 22, 3, 20, 35, 42, 41, 10, 9, 1, 31, 30, 5, 14, 13, 18, 11, 26, 27, 39, 25, 38, 12, 36, 8, 37, 17, 19, 7, 24, 40]
    return ncon(tensors, connects, con_order)


def __NTUobs2(A, B, Aop, Bop):
    return __NTUobs1(B, A, Bop, Aop)


def __NTUobs3(A, B, Aop, Bop):
    return __NTUobs1(B.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1), A.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1), Bop, Aop)


def __NTUobs4(A, B, Aop, Bop):
    return __NTUobs2(B.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1), A.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1), Bop, Aop)


def __qr(A, mode='reduced'):
    d1, d2 = A.shape[0], A.shape[1]
    if A.shape[0] > A.shape[1]:
        return qr(A, mode=mode)
    Q, R = qr(A, mode=mode)
    Qbig, Rbig = np.zeros((d1, d2), dtype=np.complex128), np.zeros((d2, d2), dtype=np.complex128)
    Qbig[:d1, :d1] = Q
    Rbig[:d1, :d2] = R
    return Qbig, Rbig

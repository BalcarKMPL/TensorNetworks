import numpy as np
from ncon import ncon
from scipy.linalg import svd
from Tools import truncate2
from scipy.linalg import norm
from scipy.linalg import qr
import copy


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


# dict={A,B,GA,GB} <- tensory iPEPS i bramki Trottera
def __step(PEPS0, maxiter=100, method="L", ifsvdu=False, ifprint=False, precision=1e-20, precisionspeed=0.01):
    PEPS = __copy(PEPS0)

    A = PEPS['A']
    B = PEPS['B']
    GA = PEPS['GA']
    GB = PEPS['GB']

    D = A.shape[0]
    d = A.shape[4]
    r = GA.shape[2]
    # print("D", D)
    # print("d", d)
    # print("r", r)

    # RA -<|- (Dr,Dr)
    An = ncon([A, GA], ([-1, -2, -4, -5, 1], [1, -6, -3])).reshape(D, D * r, D, D, d)
    Q, R = qr(An.swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4).reshape(D * D * D * d, D * r), mode='economic')
    QA = Q.reshape(D, D, D, d, D * r).swapaxes(4, 3).swapaxes(3, 2).swapaxes(2, 1)
    RA = R

    # RB -<|- (Dr,Dr)
    Bn = ncon([B, GB], ([-1, -2, -3, -4, 1], [1, -6, -5])).reshape(D, D, D, D * r, d)
    Q, R = qr(Bn.swapaxes(3, 4).reshape(D * D * D * d, D * r), mode='economic')
    QB = Q.reshape(D, D, D, d, D * r).swapaxes(4, 3)
    RB = R

    # MA, MB -|>- (Dr,D)
    MA, MB = truncate2(RA @ RB.T, D)
    MB = MB.T

    PEPS['A'] = ncon([QA, MA], ([-1, 1, -3, -4, -5], [1, -2]))
    PEPS['B'] = ncon([QB, MB], ([-1, -2, -3, 1, -5], [1, -4]))

    W = MA @ MB.T - RA @ RB.T
    tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB,
               QB.conj(), W, W.conj()]
    connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1],
                [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4],
                [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6],
                [17, 40, 24, 19, 7], [18, 41, 25, 20, 7], [12, 36, 37, 39, 8], [11, 35, 38, 42, 8], [40, 39],
                [41, 42]]
    con_order = [15, 16, 2, 28, 29, 4, 34, 33, 32, 6, 21, 23, 22, 3, 20, 35, 42, 41, 10, 9, 1, 31, 30, 5, 14,
                 13,
                 18, 11, 26, 27, 39, 25, 38, 12, 36, 8, 37, 17, 19, 7, 24, 40]
    svduerror = np.abs(ncon(tensors, connects, con_order))

    PEPS['NTUerror'] = svduerror
    PEPS['SVDUerror'] = svduerror
    print("\tSVDUE =",svduerror)

    if ifsvdu or svduerror < precision:
        return PEPS

    error = svduerror
    preverror = svduerror
    for iteration in range(maxiter):
        prevPEPS = __copy(PEPS)
        preverror = error

        def CalculateError(MA, MB):
            W = MA @ MB.T - RA @ RB.T
            tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB,
                       QB.conj(), W, W.conj()]
            connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1],
                        [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4],
                        [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6],
                        [17, 40, 24, 19, 7], [18, 41, 25, 20, 7], [12, 36, 37, 39, 8], [11, 35, 38, 42, 8], [40, 39],
                        [41, 42]]
            con_order = [15, 16, 2, 28, 29, 4, 34, 33, 32, 6, 21, 23, 22, 3, 20, 35, 42, 41, 10, 9, 1, 31, 30, 5, 14,
                         13,
                         18, 11, 26, 27, 39, 25, 38, 12, 36, 8, 37, 17, 19, 7, 24, 40]
            return np.abs(ncon(tensors, connects, con_order))
            # DDDd = D * D * D * d
            # DDd = D * D * d
            #
            # tensors = [B.swapaxes(0, 1).swapaxes(1, 2).reshape(D, D, DDd),
            #            B.conj().swapaxes(0, 1).swapaxes(1, 2).reshape(D, D, DDd),
            #            A.swapaxes(4, 3).swapaxes(3, 2).reshape(DDd, D, D),
            #            A.conj().swapaxes(4, 3).swapaxes(3, 2).reshape(DDd, D, D), B.swapaxes(0, 1).reshape(D, DDDd),
            #            B.conj().swapaxes(0, 1).reshape(D, DDDd), B.reshape(D, D, DDd), B.conj().reshape(D, D, DDd),
            #            A.swapaxes(4, 3).reshape(D, DDd, D), A.conj().swapaxes(4, 3).reshape(D, DDd, D),
            #            A.swapaxes(4, 3).reshape(DDDd, D), A.conj().swapaxes(4, 3).reshape(DDDd, D),
            #            QA, QA.conj(), QB, QB.conj(), W, W.conj()]
            # connects = [[14, 17, 2], [13, 18, 2], [3, 12, 14], [3, 11, 13], [19, 1], [20, 1], [24, 26, 4], [25, 27, 4],
            #             [37, 5, 26], [38, 5, 27], [6, 36], [6, 35], [17, 40, 24, 19, 7], [18, 41, 25, 20, 7],
            #             [12, 36, 37, 39, 8], [11, 35, 38, 42, 8], [40, 39], [41, 42]]
            # con_order = [41, 39, 6, 1, 2, 4, 3, 5, 14, 13, 26, 27, 35, 20, 42, 18, 11, 25, 38, 12, 36, 8, 37, 17, 19, 7,
            #              24, 40]
            # return np.abs(ncon(tensors, connects, con_order))

        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB,
                   QB.conj(), RA, RB, MB.conj()]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1],
                    [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4],
                    [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6],
                    [17, 41, 24, 19, 7], [18, -1, 25, 20, 7], [12, 36, 37, 40, 8], [11, 35, 38, 42, 8], [41, 39],
                    [40, 39], [42, -2]]
        con_order = [15, 16, 2, 34, 33, 32, 6, 42, 35, 31, 30, 5, 39, 10, 9, 1, 28, 29, 4, 40, 21, 23, 22, 3, 36, 8, 26,
                     27, 14, 13, 37, 38, 12, 11, 17, 24, 41, 19, 18, 25, 7, 20]
        JA = ncon(tensors, connects, con_order)

        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB,
                   QB.conj(), MB.conj(), MB]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1],
                    [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4],
                    [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6],
                    [17, -1, 24, 19, 7], [18, -3, 25, 20, 7], [12, 36, 37, 40, 8], [11, 35, 38, 39, 8], [39, -4],
                    [40, -2]]
        con_order = [15, 16, 2, 10, 9, 1, 34, 33, 32, 6, 40, 31, 30, 5, 36, 28, 29, 4, 14, 13, 39, 35, 8, 26, 27, 12,
                     11, 37, 38, 17, 24, 21, 23, 22, 3, 20, 18, 25, 19, 7]
        gA = ncon(tensors, connects, con_order)

        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB,
                   QB.conj(), MA.conj(), MA]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1],
                    [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4],
                    [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6],
                    [17, 40, 24, 19, 7], [18, 39, 25, 20, 7], [12, 36, 37, -2, 8], [11, 35, 38, -4, 8], [39, -3],
                    [40, -1]]
        con_order = [15, 16, 2, 31, 30, 5, 28, 29, 4, 21, 23, 22, 3, 10, 9, 1, 26, 27, 39, 14, 13, 20, 40, 34, 33, 32,
                     6, 36, 19, 7, 25, 24, 17, 18, 12, 37, 11, 38, 35, 8]
        gB = ncon(tensors, connects, con_order)

        tensors = [B, B.conj(), A, A.conj(), B, B.conj(), B, B.conj(), A, A.conj(), A, A.conj(), QA, QA.conj(), QB,
                   QB.conj(), RA, RB, MA.conj()]
        connects = [[15, 14, 17, 16, 2], [15, 13, 18, 16, 2], [10, 9, 12, 14, 1], [10, 9, 11, 13, 1],
                    [21, 19, 23, 22, 3], [21, 20, 23, 22, 3], [24, 26, 28, 29, 4], [25, 27, 28, 29, 4],
                    [37, 31, 30, 26, 5], [38, 31, 30, 27, 5], [34, 33, 32, 36, 6], [34, 33, 32, 35, 6],
                    [17, 41, 24, 19, 7], [18, 42, 25, 20, 7], [12, 36, 37, 40, 8], [11, 35, 38, -2, 8], [41, 39],
                    [40, 39], [42, -1]]
        con_order = [15, 16, 2, 21, 23, 22, 3, 28, 29, 4, 10, 9, 1, 42, 31, 30, 5, 20, 34, 33, 32, 6, 26, 27, 39, 41,
                     19, 7, 14, 13, 17, 18, 25, 24, 12, 40, 37, 36, 11, 38, 8, 35]
        JB = ncon(tensors, connects, con_order)

        # Metoda normalna (Dr,D)

        bestMA, bestMAerror = 0, 100000000000
        s = svd(gA.reshape(D * D * r, D * D * r).T, compute_uv=False)
        # print(s)
        for rc in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]:
            MA = __solve(gA.reshape(D * D * r, D * D * r).T, JA.reshape(D * D * r), rcond=rc)
            MA = MA.reshape(D * r, D)
            error = CalculateError(MA, MB)
            print("\t\t\t",rc,"\t",error)
            if error <= bestMAerror:
                bestMA = MA
                bestMAerror = error
            # else: break


        MA = bestMA
        if ifprint: print("\t\terror = ", bestMAerror)

        bestMB, bestMBerror = 0, 1000000000000
        s = svd(gB.reshape(D * D * r, D * D * r).T,compute_uv=False)
        # print(s)
        for rc in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]:
            MB = __solve(gB.reshape(D * D * r, D * D * r).T, JB.reshape(D * D * r), rcond=rc)
            MB = MB.reshape(D, D * r).T
            error = CalculateError(MA, MB)
            print("\t\t\t",rc,"\t",error)
            if error <= bestMBerror:
                bestMB = MB
                bestMBerror = error
            # else: break

        MB = bestMB
        if ifprint: print("\t\terror = ", bestMBerror)

        MA, MB = truncate2(MA @ MB.T, D)
        MB = MB.T

        error = np.sqrt(bestMAerror**2 + bestMBerror**2)
        PEPS['A'] = ncon([QA, MA], ([-1, 1, -3, -4, -5], [1, -2]))
        PEPS['B'] = ncon([QB, MB], ([-1, -2, -3, 1, -5], [1, -4]))
        PEPS['NTUerror'] = error

        if ifprint: print("\t", iteration, "\t", error)
        if abs(error) < precision:
            print("\t\tPREC\t\tNTUError =\t",error)
            return PEPS

        if abs(preverror) < abs(error):
            print("\t\tCONV\t\tNTUError =\t",preverror)
            return prevPEPS

    return PEPS


def NTUstep(PEPS0, maxiter=1000, method='L', ifsvdu=False, ifprint=False, precision=1e-20):
    PEPS = __copy(PEPS0)

    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __rot(PEPS)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __rot(PEPS)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __rot(PEPS)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __rotinv(PEPS)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __rotinv(PEPS)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS = __rotinv(PEPS)
    PEPS = __step(PEPS, maxiter=maxiter, method=method, ifsvdu=ifsvdu, ifprint=ifprint, precision=precision)
    PEPS['time_steps'] += 1
    return PEPS

# __step({'A': np.random.randn(5, 5, 5, 5, 2), 'B': np.random.randn(5, 5, 5, 5, 2), 'GA': np.random.randn(2, 2, 3),
#        'GB': np.random.randn(2, 2, 3)}, maxiter=5)
def __solve(A,b,rcond):
    #return np.linalg.lstsq(A,b,rcond=rcond)[0]
    # print("pinv")
    return np.linalg.pinv(A,rcond=rcond,hermitian=True) @ b
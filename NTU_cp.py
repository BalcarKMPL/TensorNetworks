import numpy as np
from ncon import ncon
from scipy.linalg import svd
from Tools import truncate2
from scipy.linalg import norm
from scipy.linalg import qr
import copy


def __rot(dict):
    def rototo(A):
        return A.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1) / norm(A)

    dict2={}
    for key in dict.keys():
        dict2[key] = dict[key]

    # if len(dict['A'].shape) == 5 or not (dict['A'].shape[0] == dict['A'].shape[1]) or not (
    #         dict['A'].shape[1] == dict['A'].shape[2]) or not (dict['A'].shape[2] == dict['A'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['A'].shape}")
    # if len(dict['B'].shape) == 5 or not (dict['B'].shape[0] == dict['B'].shape[1] == dict['B'].shape[2] == \
    #                                      dict['B'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['B'].shape}")
    dict2['A'] = rototo(dict2['A'])
    dict2['B'] = rototo(dict2['B'])
    return dict2


def __rotinv(dict):
    def rototo(A):
        return A.swapaxes(0, 1).swapaxes(0, 2).swapaxes(0, 3) / norm(A)

    dict2={}
    for key in dict.keys():
        dict2[key] = dict[key]

    # if len(dict['A'].shape) == 5 or not (dict['A'].shape[0] == dict['A'].shape[1]) or not (
    #         dict['A'].shape[1] == dict['A'].shape[2]) or not (dict['A'].shape[2] == dict['A'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['A'].shape}")
    # if len(dict['B'].shape) == 5 or not (dict['B'].shape[0] == dict['B'].shape[1] == dict['B'].shape[2] == \
    #                                      dict['B'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['B'].shape}")
    dict2['A'] = rototo(dict2['A'])
    dict2['B'] = rototo(dict2['B'])
    return dict2


# dict={A,B,GA,GB} <- tensory iPEPS i bramki Trottera
def __step(PEPS0, maxiter=1000, method="L", ifsvdu=False, ifprint=False, precision=1e-20, precisionspeed=0.01):
    # if not len(dict['A'].shape) == 5:
    #     raise Exception(f"Tensor ma rozmiar: {dict['A'].shape}")
    # if not dict['A'].shape[0] == dict['A'].shape[1] == dict['A'].shape[2] == dict['A'].shape[3]:
    #     raise Exception(f"Tensor ma rozmiar: {dict['A'].shape}")
    # if not len(dict['B'].shape) == 5:
    #     raise Exception(f"Tensor ma rozmiar: {dict['B'].shape}")
    # if not dict['B'].shape[0] == dict['B'].shape[1] == dict['B'].shape[2] == dict['B'].shape[3]:
    #     raise Exception(f"Tensor ma rozmiar: {dict['B'].shape}")
    # if not dict['A'].shape == dict['B'].shape:
    #     raise Exception(f"Rozmiary tensorów to {dict['A'].shape} i {dict['B'].shape}")
    # if not len(dict['GA'].shape) == 3:
    #     raise Exception(f"Bramka ma rozmiar {dict['GA'].shape}, {len(dict['GA'])}")
    # if not dict['GA'].shape[0] == dict['GA'].shape[1]:
    #     raise Exception(f"Bramka ma rozmiar {dict['GA'].shape}")
    # if not len(dict['GB'].shape) == 3:
    #     raise Exception(f"Bramka ma rozmiar {dict['GB'].shape}")
    # if not dict['GB'].shape[0] == dict['GB'].shape[1]:
    #     raise Exception(f"Bramka ma rozmiar {dict['GB'].shape}")
    # if not dict['GA'].shape == dict['GB'].shape:
    #     raise Exception(f"Rozmiary bramek to {dict['GA'].shape} i {dict['GB'].shape}")

    PEPS = copy.deepcopy(PEPS0)

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
    svduerror = ncon(tensors, connects, con_order)
    PEPS['NTUerror'] = svduerror
    PEPS['SVDUerror'] = svduerror
    print("\tSVDUE =", np.abs(svduerror))
    if ifsvdu or np.abs(svduerror) < precision:
        return PEPS

    preverror = svduerror
    for iteration in range(maxiter):

        def CalculateError():
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
            return ncon(tensors, connects, con_order)

        MA, MB = truncate2(MA @ MB.T, D)
        MB = MB.T

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

        # # Metoda normalna (Dr,D)
        # for rc in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]:
        # # for rc in (np.logspace(-30,-1,30,endpoint=True)):
        # # for rc in [precision]:
        #     MA = __solve(gA.reshape(D * D * r, D * D * r).T, JA.reshape(D * D * r), rcond=rc)
        #     MA = MA.reshape(D * r, D)
        #     error = CalculateError()
        #     print("\t\t\t",rc,"\t",error)
        #     if error < bestMAerror:
        #         bestMA = MA
        #         bestMAerror = error
        #     else: break
        # MA = bestMA

        U, s, Vh = svd(gA.reshape(D * D * r, D * D * r).T)
        bestMAerror = 10000000000
        print(s/s[0])
        for rc in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]:
            ind = np.argmax(np.diff(np.where(s/s[0]>rc,0,1)))
            if ind==0: ind = len(s)
            MA = (Vh.conj().T[:, :ind] @ np.diag(1 / s[:ind]) @ U.conj().T[:ind, :] @ JA.reshape(D * D * r)).reshape(D * r, D)
            error = CalculateError()
            print("\t\t\t",rc,"\t",ind,"\t",np.abs(error))
            if error < bestMAerror:
                bestMA = MA
                bestMAerror = error
            # else: break
        MA = bestMA
        if ifprint: print("\t\terror = ", np.abs(bestMAerror))

        MA, MB = truncate2(MA @ MB.T, D)
        MB = MB.T

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

        # ggg = gB.reshape(D * D * r, D * D * r)
        # print(ggg - ggg.conj().T)
        # print(norm(ggg - ggg.conj().T))

        # bestMB, bestMBerror = 0, 1000000000000
        # for rc in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]:
        # # for rc in (np.logspace(-8,-16,30,endpoint=True)):
        # # for rc in [precision]:
        #     MB = __solve(gB.reshape(D * D * r, D * D * r).T, JB.reshape(D * D * r), rcond=rc)
        #     MB = MB.reshape(D, D * r).T
        #     error = CalculateError()
        #     print("\t\t\t",rc,"\t",error)
        #     if error < bestMBerror:
        #         bestMB = MB
        #         bestMBerror = error
        #     else: break
        # MB = bestMB


        U, s, Vh = svd(gB.reshape(D * D * r, D * D * r).T)
        print(s/s[0])
        bestMBerror = 10000000000
        for rc in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18]:
            ind = np.argmax(np.diff(np.where(s/s[0]>rc,0,1)))
            if ind==0: ind = len(s)
            MB = (Vh.conj().T[:, :ind] @ np.diag(1 / s[:ind]) @ U.conj().T[:ind, :] @ JB.reshape(D * D * r)).reshape(D, D * r).T
            error = CalculateError()
            print("\t\t\t",rc,"\t",ind,"\t",np.abs(error))
            if error < bestMBerror:
                bestMB = MB
                bestMBerror = error
            # else: break
        MB = bestMB

        if ifprint: print("\t\terror = ", np.abs(bestMBerror))

        MA, MB = truncate2(MA @ MB.T, D)
        MB = MB.T

        error = np.sqrt(bestMAerror**2 + bestMBerror**2)

        if ifprint: print("\t", iteration, "\t", np.abs(error))
        if (np.abs(preverror) < (1 + precisionspeed)*np.abs(error) or np.abs(error) < precision):

            PEPS['A'] = ncon([QA, MA], ([-1, 1, -3, -4, -5], [1, -2]))
            PEPS['B'] = ncon([QB, MB], ([-1, -2, -3, 1, -5], [1, -4]))
            break
            pass

        preverror = error

    return PEPS


def NTUstep(PEPS0, maxiter=1000, method='L', ifsvdu=False, ifprint=False, precision=1e-20):
    PEPS = copy.deepcopy(PEPS0)

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


def __solve(A,b,rcond):
    #return np.linalg.lstsq(A,b,rcond=rcond)[0]
    # print("pinv")
    return np.linalg.pinv(A,rcond=rcond,hermitian=True) @ b

import numpy as np
from ncon import ncon
from scipy.linalg import svd
from Tools import truncate2
from scipy.linalg import norm

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
def __step(PEPS, GATES, ifprint=True, precision=0, ifsvdu=True, precisionspeed=0, maxiter=0, iffast=0):
    A = PEPS['A']
    B = PEPS['B']
    GA = GATES['GA']
    GB = GATES['GB']

    r = GA.shape[-1]
    d = A.shape[-1]
    D0 = A.shape[0]
    D1 = A.shape[1]
    D2 = A.shape[2]
    D3 = A.shape[3]
    D023 = D0*D2*D3
    D023d = D0*D2*D3*d
    D1r = D1*r

    An = ncon([A, GA], ([-1, -2, -4, -5, 1], [1, -6, -3])).reshape(D0, D1r, D2, D3, d).swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4).reshape(D023d, D1r)
    Bn = ncon([B, GB], ([-1, -2, -3, -4, 1], [1, -6, -5])).reshape(D2, D3, D0, D1r, d).swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0).reshape(D1r, D023d)

    UA, sA, VAh = svd(An,full_matrices=False)
    UB, sB, VBh = svd(Bn,full_matrices=False)

    U, s, Vh = svd(np.diag(sA) @ VAh @ UB @ np.diag(sB),full_matrices=False)
    U = U[:,:D1]
    Vh = Vh[:D1,:]
    s = s[:D1]

    QA = UA.reshape(D0,D2,D3,d,D1r).swapaxes(3, 4).swapaxes(2, 3).swapaxes(1, 2)
    QB = VBh.reshape(D1r,D2,D3,D0,d).swapaxes(1, 0).swapaxes(2, 1).swapaxes(3, 2)
    RA = U @ np.diag(np.sqrt(s))
    RB = Vh.T @ np.diag(np.sqrt(s))

    An = ncon([QA,RA],([-1,1,-3,-4,-5],[1,-2]))
    Bn = ncon([QB,RB],([-1,-2,-3,1,-5],[1,-4]))

    # An = (UA @ U @ np.diag(np.sqrt(s))).reshape(D,D,D,d,D).swapaxes(3, 4).swapaxes(2, 3).swapaxes(1, 2)
    # Bn = (np.diag(np.sqrt(s)) @ Vh @ VBh).reshape(D,D,D,D,d).swapaxes(1, 0).swapaxes(2, 1).swapaxes(3, 2)

    PEPS['A'] = An
    PEPS['B'] = Bn

    return PEPS


def SVDUstep(PEPS):
    for i in range(4):
        PEPS = __step(PEPS)
        PEPS = __rot(PEPS)
    PEPS['time_steps'] += 1
    return PEPS

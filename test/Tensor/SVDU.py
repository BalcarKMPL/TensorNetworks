import numpy as np
from ncon import ncon
from scipy.linalg import svd
from Tools import truncate2
from scipy.linalg import norm


def __rot(dict):
    def rototo(A):
        return A.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1) / norm(A)

    # if len(dict['A'].shape) == 5 or not (dict['A'].shape[0] == dict['A'].shape[1]) or not (
    #         dict['A'].shape[1] == dict['A'].shape[2]) or not (dict['A'].shape[2] == dict['A'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['A'].shape}")
    # if len(dict['B'].shape) == 5 or not (dict['B'].shape[0] == dict['B'].shape[1] == dict['B'].shape[2] == \
    #                                      dict['B'].shape[3]):
    #     raise Exception(f"Tensor ma rozmiary: {dict['B'].shape}")
    dict['A'] = rototo(dict['A'])
    dict['B'] = rototo(dict['B'])
    return dict


# dict={A,B,GA,GB} <- tensory iPEPS i bramki Trottera
def __step(PEPS):
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

    A = PEPS['A']
    B = PEPS['B']
    GA = PEPS['GA']
    GB = PEPS['GB']

    D = A.shape[0]
    d = A.shape[4]
    r = GA.shape[2]

    An = ncon([A, GA], ([-1, -2, -4, -5, 1], [1, -6, -3])).reshape(D, D * r, D, D, d)
    An = An.swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4).reshape(D * D * D * d, D * r)
    Bn = ncon([B, GB], ([-1, -2, -3, -4, 1], [1, -6, -5])).reshape(D, D, D, D * r, d)
    Bn = Bn.swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0).reshape(D * r, D * D * D * d)

    An, Bn = truncate2(An @ Bn, D)
    An = An.reshape(D, D, D, d, D).swapaxes(4, 3).swapaxes(3, 2).swapaxes(2, 1)
    Bn = Bn.reshape(D, D, D, D, d).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)

    PEPS['A'] = An
    PEPS['B'] = Bn

    return PEPS


def SVDUstep(PEPS):
    for i in range(4):
        PEPS = __step(PEPS)
        PEPS = __rot(PEPS)
    PEPS['time_steps'] += 1
    return PEPS

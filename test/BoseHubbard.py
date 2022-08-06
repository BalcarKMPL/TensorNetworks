import numpy as np
from ncon import ncon
from scipy.linalg import expm
import Tools


def H(d, J=0, U=1, mu=0):
    a = np.diag(np.sqrt(np.arange(0, d)), k=1)
    ah = a.T
    n = ah @ a
    nn = n @ n
    I = np.eye(d)

    HJ = -J * (np.einsum('ij,kl->jlik', a, ah) + np.einsum('ij,kl->jlik', ah, a)) / 2
    HU = U / 2 * (np.einsum('ij,kl->jlik', n, I) + np.einsum('ij,kl->jlik', I, n)) / 2 - mu * (
            np.einsum('ij,kl->jlik', nn - n, I) + np.einsum('ij,kl->jlik', I, nn - n)) / 2

    return HJ + HU


def TrotterGate(d, r, dt, J=0, U=1, mu=0):
    a = np.diag(np.sqrt(np.arange(0, d - 1)), k=1)
    ah = a.T
    n = ah @ a
    nn = n @ n
    I = np.eye(d)

    HJ = -J * (np.einsum('ij,kl->jlik', a, ah) + np.einsum('ij,kl->jlik', ah, a)) / 2
    HU = U / 2 * (nn - n) - mu * n

    GA, GB = Tools.truncate2(
        expm(-dt * HJ.reshape(d * d, d * d)).reshape(d, d, d, d).swapaxes(1, 2).reshape(d * d, d * d), r)
    GU = np.exp(-dt / 2 * HU)

    GA = GA.reshape(d, d, r)
    GB = GB.reshape(d, d, r)
    GA = ncon([GU, GA, GU], ([-2, 2], [2, 1, -3], [1, -1]))
    GB = ncon([GU, GB, GU], ([-2, 2], [2, 1, -3], [1, -1]))

    return {'GA': GA, 'GB': GB}

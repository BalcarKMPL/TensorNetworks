import numpy as np
from ncon import ncon


def H(gx=0, gz=0, J=1):
    I = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    Hzz = -np.einsum('ij,kl->ikjl', Z, Z)
    Hx = -np.einsum('ij,kl->ikjl', I, X) / 2 - np.einsum('ij,kl->ikjl', X, I) / 2
    Hz = -np.einsum('ij,kl->ikjl', I, Z) / 2 - np.einsum('ij,kl->ikjl', Z, I) / 2
    return J * Hzz + gx * Hx + gz * Hz


def TrotterGate(dt, J=1, gx=0, gz=0):
    g = np.sqrt(np.abs(gx * gx + gz * gz))
    I = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    # GX = expm(-dt / 2 * gx / 2 * X)

    Jdb = J * dt * 1j
    Gz = (np.sqrt(np.sinh(Jdb))) * np.einsum('ij,k->ijk', Z, np.array([0, 1])) + (np.sqrt(np.cosh(Jdb))) * np.einsum('ij,k->ijk', I, np.array([1, 0]))
    Gx = np.cosh(Jdb*gx/2) * I + np.sinh(Jdb*gx/2) * X

    G = ncon([Gx, Gz, Gx], ([-2, 2], [1, 2, -3], [1, -1]))
    return {'GA': G, 'GB': G}


def TrotterGateOld(dbeta, J=1, gx=0, gz=0):
    g = np.sqrt(np.abs(gx * gx + gz * gz))
    I = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    # GX = expm(-dt / 2 * gx / 2 * X)
    if g == 0:
        GZ = np.sqrt(1j * np.sin(J * dbeta)) * np.einsum('ij,k->ijk', Z, np.array([0, 1])) + \
             np.sqrt(np.cos(J * dbeta)) * np.einsum('ij,k->ijk', I, np.array([1, 0]))
        return {'GA': GZ, 'GB': GZ}
    else:
        GX = I * np.cos(dbeta * g / 4) + 1j * (gx * X + gz * Z) / g * np.sin(dbeta * g / 4)
        GZ = np.sqrt(1j * np.sin(J * dbeta)) * np.einsum('ij,k->ijk', Z, np.array([0, 1])) + \
             np.sqrt(np.cos(J * dbeta)) * np.einsum('ij,k->ijk', I, np.array([1, 0]))
        G = ncon([GX, GZ, GX], ([2, -2], [1, 2, -3], [1, -1]))
        return {'GA': G, 'GB': G}


import numpy as np
from ncon import ncon
from scipy.linalg import svd
from scipy.sparse.linalg import svds


def truncate3(A, k=-1):
    u, s, vh = svd(A, full_matrices=False)
    if k <= 0 or k >= s.shape[0]:
        return u, s, vh
    else:
        s = s[:k]
        u = (u.T[:k]).T
        vh = (vh[:k])
        return u, s, vh

def truncate3_sparse(A, k=-1):
    if k==-1 or k>=np.min(A.shape): return svd(A, full_matrices=False)
    return svds(A, k, which='LM')


def truncate2(A, k=-1):
    # A = X1 @ X2.T
    u, s, vh = truncate3(A, k)
    return u @ np.diag(np.sqrt(s)), np.diag(np.sqrt(s)) @ vh


def truncate1(A, k=-1):
    u, s, vh = truncate3(A, k)
    return u @ np.diag(s) @ vh


def estimated_time(i, n, time):
    if i == 0:
        return "ETA: inf"
    if i == n:
        return "ETA: 0"
    eta = (n - i) / i * time
    eta_h = int(eta / 3600)
    eta_m = int(eta / 60) - eta_h * 60
    eta_s = eta - eta_h * 3600 - eta_m * 60
    return "ETA: {} h {} m {:.3f} s".format(eta_h, eta_m, eta_s)

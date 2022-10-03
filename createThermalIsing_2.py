import numpy as np
import Ising
import sys, os
from ncon import ncon
import CTMRG_better_3 as CTM

if __name__ == '__main__':
    d=2
    D=2
    chi = 5
    maxiter = 50

    b_c = 1/2.269185
    betas = 1j * np.linspace(0,5,100,endpoint=True)
    dirs = '['

    for ver in np.arange(1,2):
        dir = './TI_'+str(chi)+'_'+str(ver)
        dirs = dirs + ('' if ver==1 else ',') + '\'' + dir + '\''
        if not os.path.isdir(dir): os.mkdir(dir)

        for n,beta in enumerate(betas):
            print("beta = ", beta)
            print(dir)
            G = (np.sqrt(np.tanh(beta)))*np.einsum('ij,k->ijk', np.diag([1,-1]), np.array([0, 1])) + np.einsum('ij,k->ijk', np.diag([1,1]), np.array([1, 0]))
            A = ncon([G,G,G,G,np.array([1,1])],([4,1,-1],[1,2,-2],[2,3,-3],[3,-5,-4],[4])).reshape(D,D,D,D,d)
            env = CTM.CTMRGstepL(A,A,chi,maxiter=20,ifprint=False,ver=ver,precision=1e-10,invprecision=1e-14)
            np.savez(dir + "/RHOA_{:05d}.npz".format(n), RhoThermalA=env['rhoA'], RhoThermalB=env['rhoB'], rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], beta=beta)
            print("beta =", beta)
            print("MagA =",(env['rhoA'][0,0]-env['rhoA'][1,1]).real)
            print("MagB =",(env['rhoB'][0,0]-env['rhoB'][1,1]).real)

    dirs += ']'
    print(dirs)

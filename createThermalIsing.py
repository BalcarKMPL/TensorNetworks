import numpy as np
import Ising
import sys, os
from ncon import ncon
import CTMRG_better_3 as CTM

if __name__ == '__main__':
    d=4
    D=2
    chi = 20
    maxiter = 50

    b_c = 1/2.269185
    betas = np.linspace(0.43,0.45,50,endpoint=True)
    dirs = '['

    for ver in np.arange(1,2):
        dir = './TIII_'+str(chi)+'_'+str(ver)
        dirs = dirs + ('' if ver==1 else ',') + '\'' + dir + '\''
        if not os.path.isdir(dir): os.mkdir(dir)

        for n,beta in enumerate(betas):
            # beta=100000
            print("beta = ", beta)
            print(dir)
            G = (np.sqrt(np.tanh(beta/2)))*np.einsum('ij,k->ijk', np.diag([1,-1]), np.array([0, 1])) + np.einsum('ij,k->ijk', np.diag([1,1]), np.array([1, 0]))
            print(G)
            A = ncon([G,G,G,G],([-5,1,-1],[1,2,-2],[2,3,-3],[3,-6,-4])).reshape(2,2,2,2,4)
            print(A[1,1,1,1,:].reshape(2,2))
            # exit()
            env = CTM.CTMRGstepL(A,A,chi,A,A,maxiter=200,ifprint=False,ifrandom=True,ver=ver,precision=1e-14,invprecision=1e-14)
            print(env['rhoA'])
            print(env['rhoB'])
            RhoThermalA = ncon([env['rhoA'].reshape(2,2,2,2)],([1,-1,1,-2]))
            RhoThermalB = ncon([env['rhoB'].reshape(2,2,2,2)],([1,-1,1,-2]))
            print(ncon([env['rhoA'].reshape(2,2,2,2)],([1,-1,1,-2])))
            print(ncon([env['rhoB'].reshape(2,2,2,2)],([1,-1,1,-2])))
            print(ncon([env['rhoA'].reshape(2,2,2,2)],([-1,1,-2,1])))
            print(ncon([env['rhoB'].reshape(2,2,2,2)],([-1,1,-2,1])))
            np.savez(dir + "/RHOA_{:05d}.npz".format(n), RhoThermalA=RhoThermalA, RhoThermalB=RhoThermalB, rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], beta=beta)
            print("beta =", beta)
            print("MagA =",(RhoThermalA[0,0]-RhoThermalA[1,1]).real)
            print("MagB =",(RhoThermalB[0,0]-RhoThermalB[1,1]).real)

    dirs += ']'
    print(dirs)

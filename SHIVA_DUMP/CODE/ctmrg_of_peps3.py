print("Importing OS")
import os

print("Importing numpy")
import numpy as np

print("Importing CTMRG")
import CTMRG_better

if __name__ == '__main__':
    env = {}
    chi = 25

    indexes = np.arange(0, 5, 1)
    print(indexes)

    paths = os.scandir('./BS_19.6_0.05_10')
    for pepspath in paths:
        if len(pepspath.name) < 10:
            print("Discarding", pepspath.path)
            continue
        if pepspath.name[0:4] != 'PEPS':
            print("Discarding", pepspath.path)
            continue
        index = int(pepspath.name[5:10])
        if not index in indexes:
            print("Discarding", pepspath.path)
            continue

        path = ""
        if not os.path.exists("./BS_19.6_0.05_10/RHOA_" + pepspath.name[5:10] + ".npz"):
            print("CTMRG-ing PEPS nr:", int(pepspath.name[5:10]))
            PEPS = np.load(pepspath.path)
            precision = 1e-13

            env = CTMRG_better.CTMRGstepLR(PEPS['A'], PEPS['B'], chi, maxiter=20, env=env, invprecision=1e-10,
                                           precision=precision, ifprint=True, ifrandom=False)

            np.savez(r"./BS_19.6_0.05_10/RHOA_" + pepspath.name[5:10], rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
                     E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'],
                     E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
                     C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
                     C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])

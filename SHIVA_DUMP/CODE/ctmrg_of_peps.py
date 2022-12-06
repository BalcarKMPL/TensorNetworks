print("Importing libs")
import os
import numpy as np
import CTMRG_better
import NTU
import Corelations_working as Corelations

if __name__ == '__main__':
    env = {}
    dirs = ['./brrr_10_5_1.0_17.5_0.005_1000']
    for dir in dirs:
        # peps60 --- chi=23
        indexes = np.arange(0, 30, 1)

        chi = int(dir.split('_')[1])*0+26
        print(indexes)
        paths = os.scandir(dir)
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
            precision = 1e-20

            if True or not os.path.exists(dir + "/RHOA_" + pepspath.name[5:10] + ".npz"):
                print("CTMRG-ing PEPS nr:", int(pepspath.name[5:10]))
                PEPS = np.load(pepspath.path)
                try:
                    env0 = dict(np.load(dir + "/RHOA_" + pepspath.name[5:10]+".npz"))
                    print("error = ", env0['error'].max())
                    if env0['error'].max() < precision:
                        continue
                except:
                    env0={}
                for i in range(100):
                    print("CTMRG-ing PEPS nr:", int(pepspath.name[5:10]))
                    # env0 = env
                    env = CTMRG_better.CTMRGstepL(PEPS['A'], PEPS['B'], chi, env0=env0, maxiter=10, invprecision=1e-10, precision=precision, ifprint=False, ifrandom=False)
                    # print(env)
                    if env['error'] < precision:
                        break
                    if 'error' in env0:
                        if env['error'] > env0['error']:
                            break
                    env0 = env

                    np.savez(dir + "/RHOA_" + pepspath.name[5:10], rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
                             E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'],
                             E_S_B=env['E_S_B'],
                             E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
                             C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
                             C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])
                print("Corelating PEPS nr:", int(pepspath.name[5:10]))

                a = np.diag(np.sqrt(np.arange(1, 3)), k=1)
                ah = a.T
                n = ah @ a
                nn = n @ n

                envrot = CTMRG_better.__rot1env(env)
                PEPSrot = NTU.__rotinv(PEPS)

                i = int(pepspath.name[5:10])
                corrWE = Corelations.Corelation(env, PEPS, ah, a, 1)
                corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 1)
                np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])
                corrWE = Corelations.Corelation(env, PEPS, a, ah, 1)
                corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 1)
                np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])
                corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 1)
                corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 1)
                np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])

print("Importing libs")
import os
import numpy as np
import CTMRG_better_3 as CTM
import NTU
import sys
import Corelations_working as Corelations

if __name__ == '__main__':
    changed = True

    def chii(i):
        if i <= 5:  return 9
        if i <= 10: return 14
        if i <= 15: return 17
        if i <= 20: return 24
        if i <= 40: return 24
        return 24

    while changed:
        changed = False
        env = {}
        dirs = ['./BHNEWNTU3_6_6_1.0_4.9_0.005_99_3','./BHNEWNTU3_6_6_1.0_4.9_0.005_99_4','./BHNEWNTU3_6_6_1.0_4.9_0.005_99_5','./BHNEWNTU3_6_6_1.0_4.9_0.005_99_6','./BHNEWNTU3_6_6_1.0_4.9_0.005_99_7','./BHNEWNTU3_6_6_1.0_4.9_0.005_99_8']

        for i in range(0,100,5):
            for dir in dirs:
                ver = int(dir.split('_')[-1])
                try: paths = os.scandir(dir)
                except: continue
                if ver == 3 and i < 43: continue
                if ver != 8 and i < 31: continue
                if ver == 8 and i < 29: continue
                print(dir)
                chi = chii(i)
                print("CTMRG-ing PEPS nr:", i)
                PEPS = dict(np.load(dir + "/PEPS_{:05d}.npz".format(i)))

                env = CTM.CTMRGstepL(PEPS['A'], PEPS['B'], chi, env0={}, maxiter=20, invprecision=1e-10,
                                                   precision=1e-15, ifprint=False, ifrandom=False, ver=ver)


                np.savez(dir + "/RHOA_{:05d}.npz".format(i), rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
                         E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'],
                         E_S_B=env['E_S_B'],
                         E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
                         C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
                         C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])
                print("Corelating PEPS nr:", i)

                a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
                ah = a.T
                n = ah @ a
                nn = n @ n

                envrot = CTM.__rot1env(env)
                PEPSrot = NTU.__rotinv(PEPS)

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

                # X, Z = np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])
                #
                # np.savez(dir + ('/OBS_{:05d}.npz').format(i), XA=np.trace(X @ env['rhoA']), XB=np.trace(X @ env['rhoB']),
                #          ZA=np.trace(Z @ env['rhoA']), ZB=np.trace(Z @ env['rhoB']))
                # corrWE = Corelations.Corelation(env, PEPS, Z, Z, 1)
                # corrNS = Corelations.Corelation(envrot, PEPSrot, Z, Z, 1)
                # np.savez(dir + ('/CORR_ZZ_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                # np.savez(dir + ('/CORR_ZZ_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])
                # corrWE = Corelations.Corelation(env, PEPS, X, X, 1)
                # corrNS = Corelations.Corelation(envrot, PEPSrot, X, X, 1)
                # np.savez(dir + ('/CORR_XX_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                # np.savez(dir + ('/CORR_XX_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])

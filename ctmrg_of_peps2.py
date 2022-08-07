print("Importing libs")
import os
import numpy as np
import CTMRG_better
import NTU
import sys
import Corelations_working as Corelations

if __name__ == '__main__':
    # i0 = int(float(sys.argv[1]))
    changed = True
    while changed:
        changed = False
        env = {}
        dirs = ['./ISING/ISING_NEWNTU_sud_5_5_1.0_0.3044_0.01_200']

        for dir in dirs:
            step = int(int(dir.split('_')[8]) / 25)
            indexes = np.arange(25,75,1)

            chi = int(dir.split('_')[3])*0+20
            print(indexes)
            print(chi)
            try: paths = os.scandir(dir)
            except: continue
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
                precision = 1e-13
                if True or not os.path.exists(dir + "/RHOA_" + pepspath.name[5:10] + ".npz"):
                    print("CTMRG-ing PEPS nr:", int(pepspath.name[5:10]))
                    PEPS = np.load(pepspath.path)
                    try:
                        env0 = dict(np.load(dir + "/RHOA_" + pepspath.name[5:10]+".npz"))
                        if env0['error'].max()<precision: continue
                    except: env0={}
                    for i in range(15):
                        print("CTMRG-ing PEPS nr:", int(pepspath.name[5:10]))
                        # env0 = env
                        changed = True
                        env = CTMRG_better.CTMRGstepLR(PEPS['A'], PEPS['B'], chi, env0=env0, maxiter=5, invprecision=1e-10,
                                                       precision=precision, ifprint=False, ifrandom=False)
                        # print(env)
                        if 'error' in env0:
                            if env['error'] > env0['error'] or env['error'] < precision:
                                break
                        env0 = env

                    np.savez(dir + "/RHOA_" + pepspath.name[5:10], rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'],
                             E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'],
                             E_S_B=env['E_S_B'],
                             E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'],
                             C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'],
                             C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'])
                    print("Corelating PEPS nr:", int(pepspath.name[5:10]))

                    a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
                    ah = a.T
                    n = ah @ a
                    nn = n @ n

                    envrot = CTMRG_better.__rot1env(env)
                    PEPSrot = NTU.__rotinv(PEPS)

                    i = int(pepspath.name[5:10])
                    # corrWE = Corelations.Corelation(env, PEPS, ah, a, 2)
                    # corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, 2)
                    # np.savez(dir + ('/CORR_AHA_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                    # np.savez(dir + ('/CORR_AHA_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])
                    # corrWE = Corelations.Corelation(env, PEPS, a, ah, 2)
                    # corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, 2)
                    # np.savez(dir + ('/CORR_AAH_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                    # np.savez(dir + ('/CORR_AAH_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])
                    # corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, 2)
                    # corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, 2)
                    # np.savez(dir + ('/CORR_NN_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                    # np.savez(dir + ('/CORR_NN_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])

                    X, Z = np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])

                    np.savez(dir + ('/OBS_{:05d}.npz').format(i), XA=np.trace(X @ env['rhoA']), XB=np.trace(X @ env['rhoB']),
                             ZA=np.trace(Z @ env['rhoA']), ZB=np.trace(Z @ env['rhoB']))
                    corrWE = Corelations.Corelation(env, PEPS, Z, Z, 1)
                    corrNS = Corelations.Corelation(envrot, PEPSrot, Z, Z, 1)
                    np.savez(dir + ('/CORR_ZZ_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                    np.savez(dir + ('/CORR_ZZ_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])
                    corrWE = Corelations.Corelation(env, PEPS, X, X, 1)
                    corrNS = Corelations.Corelation(envrot, PEPSrot, X, X, 1)
                    np.savez(dir + ('/CORR_XX_NS_{:05d}.npz'.format(i)), corA=corrNS['corA'], corB=corrNS['corB'])
                    np.savez(dir + ('/CORR_XX_WE_{:05d}.npz'.format(i)), corA=corrWE['corA'], corB=corrWE['corB'])

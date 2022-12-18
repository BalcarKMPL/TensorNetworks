import numpy as np
import sys
import CTMRG_better_4 as CTM
import NTU_NEW_4 as NTU
import Corelations_working as Corelations

print("plz work")
def file(D, k):
    return './BH-cubicramp_3_' + str(D) + '_' + str(float(k)) + '_0.1'

chimult = 2
INVprecision = 1e-10
CTMRGprecision = 1e-12

ks = np.arange(0, 110, 5)

print("i beg you fella")
for chimult in [2, 3, 4, 5]:
    for D in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
        for k in ks:
            i_around = int(3 / 2 * 2 ** (k / 10))
            inds = np.arange(i_around - 2, i_around + 3)
            for i in inds:
                print("############################################################")
                print(f"Completing D = {D}, k = {k}, i = {i}, chi / D = {chimult}")
                a = np.diag(np.sqrt(np.arange(1, 3)), k=1)
                ah = a.T
                dirr = file(D, k)
                try: env = np.load(dirr + f'/RHOA_{chimult:01d}_{i:05d}.npz')
                except:
                    print("Env not found\n")
                    continue
                PEPS = np.load(dirr + f'/PEPS_{i:05d}.npz')

                envrot = CTM.__rot1env(env)
                PEPSrot = NTU.__rotinv(PEPS)
                dt = PEPS['dt']
                range = 15

                try:
                    cor = dict(np.load(dirr + f'/CORR_{chimult:01d}_AHA_WE_{i:05d}.npz'))
                    if (cor['corA'].shape[0] < 2 * range and cor['corB'].shape[0] < 2 * range): raise Exception("")
                except:
                    corrWE = Corelations.Corelation(env, PEPS, ah, a, range)
                    np.savez(dirr + (f'/CORR_{chimult:01d}_AHA_WE_{i:05d}.npz'), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)

                try:
                    cor = dict(np.load(dirr + f'/CORR_{chimult:01d}_AHA_NS_{i:05d}.npz'))
                    if (cor['corA'].shape[0] < 2 * range and cor['corB'].shape[0] < 2 * range): raise Exception("")
                except:
                    corrNS = Corelations.Corelation(envrot, PEPSrot, ah, a, range)
                    np.savez(dirr + (f'/CORR_{chimult:01d}_AHA_NS_{i:05d}.npz'), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)

                try:
                    cor = dict(np.load(dirr + f'/CORR_{chimult:01d}_AAH_WE_{i:05d}.npz'))
                    if (cor['corA'].shape[0] < 2 * range and cor['corB'].shape[0] < 2 * range): raise Exception("")
                except:
                    corrWE = Corelations.Corelation(env, PEPS, a, ah, range)
                    np.savez(dirr + (f'/CORR_{chimult:01d}_AAH_WE_{i:05d}.npz'), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)

                try:
                    cor = dict(np.load(dirr + f'/CORR_{chimult:01d}_AAH_NS_{i:05d}.npz'))
                    if (cor['corA'].shape[0] < 2 * range and cor['corB'].shape[0] < 2 * range): raise Exception("")
                except:
                    corrNS = Corelations.Corelation(envrot, PEPSrot, a, ah, range)
                    np.savez(dirr + (f'/CORR_{chimult:01d}_AAH_NS_{i:05d}.npz'), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)

                try:
                    cor = dict(np.load(dirr + f'/CORR_{chimult:01d}_NN_WE_{i:05d}.npz'))
                    if (cor['corA'].shape[0] < 2 * range and cor['corB'].shape[0] < 2 * range): raise Exception("")
                except:
                    corrWE = Corelations.Corelation(env, PEPS, ah @ a, ah @ a, range)
                    np.savez(dirr + (f'/CORR_{chimult:01d}_NN_WE_{i:05d}.npz'), corA=corrWE['corA'], corB=corrWE['corB'], iter=i, dt=dt)

                try:
                    cor = dict(np.load(dirr + f'/CORR_{chimult:01d}_NN_NS_{i:05d}.npz'))
                    if (cor['corA'].shape[0] < 2 * range and cor['corB'].shape[0] < 2 * range): raise Exception("")
                except:
                    corrNS = Corelations.Corelation(envrot, PEPSrot, ah @ a, ah @ a, range)
                    np.savez(dirr + (f'/CORR_{chimult:01d}_NN_NS_{i:05d}.npz'), corA=corrNS['corA'], corB=corrNS['corB'], iter=i, dt=dt)

                print("Done\n\n")
print("wreszcie mogę odpocząć")
print("do zobaczenia")
import numpy as np
import sys
import CTMRG_better_4 as CTM

print("plz work")
def file(D, k):
    return './BH-cubicramp_3_' + str(D) + '_' + str(float(k)) + '_0.1'

chimult = 2
INVprecision = 1e-10
CTMRGprecision = 1e-12

ks = np.arange(0, 110, 10)

print("i beg you fella")
for chimult in [2, 3, 4]:
    for D in [4, 6, 8, 10, 12, 14]:
        for k in ks:
            i_around = 3 / 2 * 2 ** (k / 10)
            if i_around - np.floor(i_around) > 0.3:
                inds_around = [int(np.floor(i_around)), int(np.ceil(i_around))]
            else:
                inds_around = [int(np.round(i_around))-1,int(np.round(i_around)),int(np.round(i_around))+1]
            print(inds_around)
            for i in np.arange(0,int(np.round(i_around))+2):
                print("############################################################")
                print(f"Completing D = {D}, k = {k}, i = {i}, chi / D = {chimult}")
                dirr = file(D, k)
                try:
                    env = np.load(dirr + f'/RHOA_{chimult:01d}_{i:05d}.npz')
                    print("Found env for said params")
                    continue
                except:
                    try:
                        PEPS = dict(np.load(dirr + f'/PEPS_{i:05d}.npz'))
                    except:
                        print("Not found PEPS for said params")
                        continue

                    env0 = {}
                    chimult0 = 0
                    # ładuje jako env0 ctm o największym dostępnym chi
                    for chitemp in np.arange(2, chimult):
                        try:
                            env0 = np.load(dirr + f'/RHOA_{chitemp:01d}_{i:05d}.npz')
                            chimult0 = chitemp
                        except: pass
                    print(f"Using env with chi / D = {chimult0}")

                    a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
                    ah = a.T
                    env = CTM.CTMRGstepLtest(PEPS['A'],PEPS['B'],chimult*PEPS['A'].shape[0],PEPS['A'].conj(),PEPS['B'].conj(),maxiter=100,env0=env0,invprecision=INVprecision,precision=CTMRGprecision,ifprint=False,ifrandom=False,tests1=[{'name':'N','A':ah @ a,'B':ah @ a},{'name':'NN','A':ah @ a@ah @ a,'B':ah @ a@ah @ a}], tests2=[{'name':'AhA','A':ah,'B':a},{'name':'AAh','A':a,'B':ah},{'name':'NN','A':ah @ a,'B':ah @ a}])
                    np.savez(dirr + f'/RHOA_{chimult:01d}_{i:05d}.npz', rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=i, dt=PEPS['dt'], A=PEPS['A'], B=PEPS['B'], J=PEPS['J_exact'], U=PEPS['U'], names1=env['names1'], names2=env['names2'], vals1=env['vals1'], vals2=env['vals2'], errors1=env['errors1'], errors2=env['errors2'], INVprecision=INVprecision, CTMRGprecision=CTMRGprecision)
            print("\n\n")
print("trud mój zakończon, mogę odpocząć")
print("dobranoc")
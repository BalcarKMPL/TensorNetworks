import numpy as np
import sys
import CTMRG_better_4 as CTM

print("plz work")
def file(D, k):
    return './BH-cubicramp_3_' + str(D) + '_' + str(float(k)) + '_0.1'

print("before anything really")
chimult = 2
INVprecision = 1e-10
CTMRGprecision = 1e-12

if int(sys.argv[1]) == 0:
    ks = np.arange(0, 60, 10)
else:
    ks = np.arange(60, 110, 10)

print("i beg you fella")
for D in [4, 6, 8, 10, 12, 14]:
    for k in ks:
        i_around = 3 / 2 * 2 ** (k / 10)
        if i_around - np.floor(i_around) > 0.3:
            inds_around = [int(np.floor(i_around)), int(np.ceil(i_around))]
        else:
            inds_around = [int(np.round(i_around))]
        print(inds_around)
        for i in inds_around:
            dirr = file(D, k)
            try: env = np.load(dirr + f'/RHOA_{chimult:01d}_{i:05d}.npz')
            except:
                try: PEPS = dict(np.load(dirr + f'/PEPS_{i:05d}.npz'))
                except: continue

                print("#################################################")
                print(f"  completing D = {D}, k = {k}, i = {i}")
                print("#################################################")
                a = np.diag(np.sqrt(np.arange(1, PEPS['A'].shape[-1])), k=1)
                ah = a.T
                env = CTM.CTMRGstepLtest(PEPS['A'],PEPS['B'],chimult*PEPS['A'].shape[0],PEPS['A'].conj(),PEPS['B'].conj(),maxiter=100,env0={},invprecision=INVprecision,precision=CTMRGprecision,ifprint=False,ifrandom=False,tests1=[{'name':'N','A':ah @ a,'B':ah @ a},{'name':'NN','A':ah @ a@ah @ a,'B':ah @ a@ah @ a}], tests2=[{'name':'AhA','A':ah,'B':a},{'name':'AAh','A':a,'B':ah},{'name':'NN','A':ah @ a,'B':ah @ a}])
                np.savez(dirr + f'/RHOA_{chimult:01d}_{i:05d}.npz', rhoA=env['rhoA'], rhoB=env['rhoB'], E_E_A=env['E_E_A'], E_E_B=env['E_E_B'], E_W_A=env['E_W_A'], E_W_B=env['E_W_B'], E_S_A=env['E_S_A'], E_S_B=env['E_S_B'], E_N_A=env['E_N_A'], E_N_B=env['E_N_B'], C_NW_A=env['C_NW_A'], C_SW_B=env['C_SW_B'], C_NE_B=env['C_NE_B'], C_SE_A=env['C_SE_A'], C_NW_B=env['C_NW_B'], C_SW_A=env['C_SW_A'], C_NE_A=env['C_NE_A'], C_SE_B=env['C_SE_B'], error=env['error'], iter=i, dt=PEPS['dt'], A=PEPS['A'], B=PEPS['B'], J=PEPS['J_exact'], U=PEPS['U'], names1=env['names1'], names2=env['names2'], vals1=env['vals1'], vals2=env['vals2'], errors1=env['errors1'], errors2=env['errors2'], INVprecision=INVprecision, CTMRGprecision=CTMRGprecision)
                print("\n\n\n")
print("trud mój zakończon, mogę odpocząć")
print("dobranoc")
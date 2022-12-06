import numpy as np
from matplotlib import pyplot as plt
from ncon import ncon

def GetData(d,D,k,chimult):
    data = []
    for suffix in ["-trzy","-dwa",""]:
        try:
            path = f"./all_bh_ramps/BH-cubicramp_{d}_{D}_{k:.1f}_0.1"+suffix
            SPECS = dict(np.load(path+"/SPECS.npz"))
            break
        except:
            continue
    try:
        SPECS = dict(np.load(path+"/SPECS.npz"))
        finaliter = int(3/2*SPECS['tQ']/SPECS['dt']+0.0001) + 1
        # print(D,k,"\t\t",int(3/2*SPECS['tQ']/SPECS['dt']+0.0001))
        env = dict(np.load(path+f"/RHOA_{chimult}_{finaliter:05d}.npz"))
    except: raise Exception("aaa")
    for i in range(0,finaliter+1):
        try:
            env = dict(np.load(path+f"/RHOA_{chimult}_{i:05d}.npz"))
        except:
            break
        U,J=env['U'],env['J']
        m = [[env['iter'],env['iter']*env['dt'],0.1*2**(k/10)]]
        # name mean syst_error std_error
        for iter in range(len(env['names1'])):
            m.append([np.real_if_close(np.mean(env['vals1'][iter])), np.real_if_close(np.mean(env['errors1'][iter])), np.real_if_close(np.std(env['vals1'][iter]))])
        for iter in range(len(env['names2'])):
            m.append([np.real_if_close(np.mean(env['vals2'][iter])), np.real_if_close(np.mean(env['errors2'][iter])), np.real_if_close(np.std(env['vals2'][iter]))])
        m = np.array(m)
        M = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0],[0,-U/2,U/2,-2*J,-2*J,0],[0,0,0,0,0,1]])
        m = ncon([M,m],([-1,1],[1,-2]))
        data.append(m)
    data = np.array(data)
    return data

d=3
# for D in [10]:
#     for k in [60]:
for D in [4,6,8,10,12,14]:
    for k in [10,20,30,40,50,60,70,80,90,100]:
        print(D,"  ",k)
        chimult=2
        try: drr = GetData(d,D,k,2)
        except: continue
        if drr.shape[0] == 0: continue
        drrreshaped = drr.reshape(drr.shape[0],drr.shape[1]*drr.shape[2])
        np.savetxt(f"./BH-cubicramp_{d}_{D}_{k:.1f}_0.1.csv",drrreshaped,delimiter=',')
        np.savetxt(f"./BH-cubicramp_{d}_{D}_{k:.1f}_0.1-real.csv",drrreshaped.real,delimiter=',')
        # f = open(f"./BH-cubicramp_{d}_{D}_{k:.1f}_0.1.txt","w")
        # filecontent = ""
        # for i0 in range(0,drr.shape[0]):
        #     print(i0,"\t\t",drr[i0,:,:])
        #     t = i0*0.1 - 3/2*0.1*2**(k/10)
        #     for i1 in range(0,drr.shape[1]):
        #         for i2 in range(0,drr.shape[2]):
        #             filecontent += str(drr[i0,i1,i2])+"\t"
        #         filecontent += "\n"
        # f.close()
        # for ind in range(0,5):
        #     plt.plot(ts,ind/10000*0+data[:,ind,1].astype(complex)-data[0,ind,1].astype(complex),label=data[0,ind,0]+" - "+data[0,ind,0]+"(t=0)")
        #     plt.legend()
print("DONE")
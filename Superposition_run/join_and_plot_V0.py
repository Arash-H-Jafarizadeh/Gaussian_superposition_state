import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import glob
import sys


job_number = int(sys.argv[1])


# folder_path = 'raw_data/'
# folder_path = '/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition/raw_data/'
folder_path = 'Superposition/raw_data/'
all_files =  sorted( glob.glob('*.npy', root_dir=folder_path) )

L = 14
maxdim = sp.special.binom(L, L//2)


if True: ########################################################################### Ploting SCALING of Energy Convergence - 13022025
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    ed_energis0 = [-4.670424529225023, -5.908793067783501, -7.148823327207109, -8.389848287475354, -9.631516108034338 ]
    ed_energis1 = [-4.064323068264334, -5.088418396429466, -6.112851067725644, -7.137489761802797, -8.16226349870654 ]
    
    hf_data0, hf_data1 = [], []
    inf_data0, inf_data1 = [], []
    for nome in all_files:
        if nome[:3] == "TRN":
            data = np.load(folder_path + nome)
            
            hf_data0.append(data[0])
            hf_data1.append(data[1])
        
        if nome[:4] == "INFD":
            data = np.load(folder_path + nome)

            inf_data0.append(data[0])
            inf_data1.append(data[1])
                    
    
    scaling_X = np.array([8, 10, 12, 14, 16])
    num_dims = 15
    x_array = []
    for L in scaling_X:
        maxdim = sp.special.binom(L, L//2)
        truncation = np.linspace(1, maxdim, num_dims, dtype=np.int64)
        # x_array.append( truncation / maxdim )
        x_array.append(truncation)
    
    hf_data0 = np.reshape(hf_data0, (5, num_dims))
    hf_data0[1,-1] = 5.32412e-16
    hf_data1 = np.reshape(hf_data1, (5, num_dims))
    hf_data1[0,-1] = 5.73901e-16


    # scaling_data = []
    # scaling_labels = []
    # for thrsh in [1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-12]:
    #     scaling_labels.append(str(thrsh))
    #     loop = []
    #     for p in range(5):
    #         loop_data = np.array(hf_data0[p])
    #         loop_var = len(loop_data[loop_data > thrsh])    
    #         loop.append( x_array[p][loop_var-1] )        
    #     scaling_data.append(loop)    
        
    #     ##############fiting data
    #     # fitness = np.polyfit(np.log(scaling_X), np.log(np.array(loop)), 1)
    #     # print("fitness: ",fitness)

    # print("scaling_data")        
    # print(scaling_data)        
    # print("")        

    fig, ax = plt.subplots( figsize=(10, 6))

    ax.set_title(r"NEW error scaling with size for V=0.2 (linear-linear) and no fiiting")#: $\log y = a_0 x + a_1$")
    ax.set_ylabel(r"# states")    
    # ax.set_yscale('log')
    ax.grid(which='major', axis='both', linestyle=':')
    # ax.set_xlabel(r"Ratio of states in superposition basis")    
    # ax.set_xlabel(r"$\frac{\# state}{10^4 - 3\times 10^3L + 10^2L^2 + 10L^3 - 5\times 10^{-1}L^4 -  7 \times10^{-2} L^5 + 3\times10^{-3}L^6|}$")    
    ax.set_xlabel(r"L")    
    # ax.set_xlim(0,1.e+2)
    # ax.set_xscale('log')
    ax.set_xticks(scaling_X, ['8','10','12','14','16'])
     
    # cores = [f'C{indx}' for indx in range()]
    marks = ['o','s','^','v','D','p','h','8']
    # lebels = ['L=8','L=10','L=12','L=14','L=16','L=18']
    

    # scaling_data = []
    # scaling_labels = []
    for p, thrsh in enumerate([1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-12]):
        # scaling_labels.append(str(thrsh))
        loop = []
        for d in range(5):
            new_hf = np.abs(ed_energis0[d]) * np.array(hf_data0[d]) / scaling_X[d]
            loop_data = new_hf #np.array(hf_data0[d])
            loop_var = len(loop_data[loop_data > thrsh])    
            loop.append( x_array[d][loop_var-1] )
        #################################################### for V=1.8
        ax.plot(scaling_X, loop, label=str(thrsh), color = f'C{p}', ls=':', linewidth=0.8, marker=marks[p] )
        #################################################### for V=1.8
        # ax.plot(, , label=, color = f'C{p}', ls=':', linewidth=0.9, marker=marks[p])  #, fillstyle='none')        
        #################################################### fiting log-log
        # fitness = np.polyfit(np.log(scaling_X), np.log(np.array(loop)), 1)
        # lin_data = [np.exp(fitness[1])*(x**fitness[0]) for x in scaling_X]
        # ax.plot(scaling_X, lin_data, label=rf'$e^{{{fitness[1]:.2f}}} L^{{{fitness[0]:.2f}}}$', color = f'C{p}', ls='--', linewidth=0.7, marker='None')
        #################################################### fiting log-linear
        # fitness = np.polyfit(scaling_X, np.log(np.array(loop)), 1)
        # lin_data = [np.exp(fitness[1]) * np.exp( x*fitness[0]) for x in scaling_X]
        # ax.plot(scaling_X, lin_data, label=rf'''$e^{{{fitness[1]:.2f}}}e^{{{fitness[0]:.2f}L}}$''', color = f'C{p}', ls='--', linewidth=0.7, marker='None')
        # print("fitness: ",fitness)


    # for p in range(7):
    #     #################################################### for V=0.2
    #     ax.plot(scaling_X, scaling_data[p], label=scaling_labels[p], color = f'C{p}', ls='', linewidth=0.8, marker=marks[p] )
    #     # ins_ax.plot(x_array[p], inf_data0[p], label=lebels[p], ls=':',linewidth=0.6, marker=marks[p], markersize=4) #color='C0',
    #     #################################################### for V=1.8
    #     # ax.plot(x_array[p], hf_data1[p], label=lebels[p], color = f'C{p}', ls=':', linewidth=0.9, marker=marks[p])  #, fillstyle='none')
    #     # ins_ax.plot(x_array[p], inf_data1[p], label=lebels[p], ls=':',linewidth=0.6, marker=marks[p], markersize=4) #color='C0',
        
    # lin_data = [np.exp(-6.65)*(x**5.12) for x in scaling_X]
    # ax.plot(scaling_X, lin_data, label='linear', color = 'C8', ls='-', linewidth=0.8, marker='None')

    ax.legend(loc='best', title="Energy error")
    ############## Re-arranging labels ordering
    # handles,labels = ax.get_legend_handles_labels()
    # order = [0,2,4,6,8,10,12,1,3,5,7,9,11,13]
    # handles = [handles[ii] for ii in order]
    # labels = [labels[jj] for jj in order]
    # ax.legend(handles, labels, loc='best', title="Energy error           fitted line", ncol=2)
    
    plt.savefig(f"Superposition/output/New_ErrorScaling_V02_(lin-lin)_J{job_number}.pdf", bbox_inches = 'tight')


if False: ########################################################################### Ploting Energy Convergence Scaling with system size - 10022025
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    ed_energis0 = [-4.670424529225023, -5.908793067783501, -7.148823327207109, -8.389848287475354, -9.631516108034338 ]
    ed_energis1 = [-4.064323068264334, -5.088418396429466, -6.112851067725644, -7.137489761802797, -8.16226349870654 ]
    
    hf_data0, hf_data1 = [], []
    inf_data0, inf_data1 = [], []
    for nome in all_files:
        if nome[:3] == "TRN":
            data = np.load(folder_path + nome)
            # print("   - ",nome)
            # print("") # print("   - ",data) # print("")
            
            hf_data0.append(data[0])
            hf_data1.append(data[1])
        
        if nome[:4] == "INFD":
            data = np.load(folder_path + nome)
            # print("   - (data name) ",nome, " - (data shgape) ", np.shape(data))
            inf_data0.append(data[0])
            inf_data1.append(data[1])
    
                    
    sizes = [8, 10, 12, 14, 16]
    num_dims = 15
    x_array = []
    for L in sizes:
        maxdim = sp.special.binom(L, L//2)
        truncation = np.linspace(1, maxdim, num_dims, dtype=np.int64)
        # x_array.append( truncation / maxdim )
        x_array.append(truncation)
                 
    
    hf_data0 = np.reshape(hf_data0, (5, num_dims))
    hf_data0[1,-1] = 5.32412e-16
    hf_data1 = np.reshape(hf_data1, (5, num_dims))
    hf_data1[0,-1] = 5.73901e-16
    
    
    # fig = plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots( figsize=(13, 6))

    ax.set_title(r"Superposition energy scaled by sector size for V=0.2")
    # ax.set_ylabel(r"$\frac{|E - E_{ed}|}{L}$")    #(r"$|\frac{E - E_{ed}}{E_{ed}}| $")    
    ax.set_yscale('log')
    ax.grid(which='major', axis='both', linestyle=':')
    # ax.set_xlabel(r"Ratio of states in superposition basis")    
    ax.set_xlabel(r"$\frac{\# state}{N_L}$")    #(r"$\frac{\# state}{10^4 - 3\times 10^3L + 10^2L^2 + 10L^3 - 5\times 10^{-1}L^4 -  7 \times10^{-2} L^5 + 3\times10^{-3}L^6|}$")    
    # ax.set_xlim(0,2.e+2)
    # ax.set_xscale('log')
    # ax.set_xticks(pnt_list)
    
    # *************************************** inset plot for infidality ******************************************
    # ins_ax = ax.inset_axes([0.04,0.06,0.5,0.45])#width="20%", height="20%", loc="lower right", borderpad=1)
    # ins_ax.set_yscale('log')
    # # ins_ax.set_xlim(0,5.e+2)
    # ins_ax.grid(which='major', axis='both', linestyle=':')
    # ins_ax.tick_params(direction='in', labelsize=7)
    # ins_ax.set_title(r"$1-\mathcal{F}$")
    
    # # cores = [f'C{indx}' for indx in range()]
    marks = ['o','s','^','v','D','p','h','8']
    lebels = ['L=8','L=10','L=12','L=14','L=16','L=18']
    for p in range(5):
        #################################################### for V=0.2
        # new_hf = np.abs(ed_energis0[p]) * np.array(hf_data0[p]) / sizes[p]
        # ax.plot(x_array[p], new_hf, label=lebels[p], color = f'C{p}', ls='--', linewidth=0.8, marker=marks[p] )
        ax.plot(x_array[p], hf_data0[p], label=lebels[p], color = f'C{p}', ls='--', linewidth=0.8, marker=marks[p] )
        ins_ax.plot(x_array[p], inf_data0[p], label=lebels[p], ls=':',linewidth=0.6, marker=marks[p], markersize=4) #color='C0',
        #################################################### for V=1.8
        # ax.plot(x_array[p], hf_data1[p], label=lebels[p], color = f'C{p}', ls=':', linewidth=0.9, marker=marks[p])  #, fillstyle='none')
        # ins_ax.plot(x_array[p], inf_data1[p], label=lebels[p], ls=':',linewidth=0.6, marker=marks[p], markersize=4) #color='C0',
        
    ax.legend(loc='best')
    
    plt.savefig(f"Superposition/output/NewSizeScalingEnrgyConverg_V02_(log)_J{job_number}.pdf", bbox_inches = 'tight')


if False: ########################################################################### Ploting Scaling of Amplitudes with system size - 10022025
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    amp_data0, amp_data1 = [], []
    for nome in all_files:
        if nome[:4] == "AMPS":
            data = np.load(folder_path + nome)
            print("   - ",nome)
            # print("") # print("   - ",data) # print("")
            
            amp_data0.append(data[0])
            amp_data1.append(data[1])
                    
                    
    fig, ax = plt.subplots( figsize=(11, 7))

    
    # cores = [f'C{indx}' for indx in range()]
    marks = ['o','s','^','v','D','p','h','8']
    lebels = ['L=8','L=10','L=12','L=14','L=16','L=18']
    # labals = ['L=8 V=1.8','L=10 V=1.8','L=12 V=1.8','L=14 V=1.8','L=16 V=1.8','L=18 V=1.8']
    for p in range(5):
        ################# for V=0.2
        LEN = len(amp_data0[p])
        # x_array = [ii/LEN for ii in range(1,LEN+1)]
        x_array = [ii for ii in range(1,len(amp_data0[p])+1)]
        # print(LEN) # print(x_array)
        ax.plot(x_array, amp_data0[p], label=lebels[p], color = f'C{p}', ls=':', linewidth=0.6, marker=marks[p], markersize = 3 )
        
        ################# for V=1.8
        # LEN = len(amp_data1[p])
        # # x_array = [ii/LEN for ii in range(1,LEN+1)]
        # x_array = [ii for ii in range(1,LEN+1)]
        # ax.plot(x_array, amp_data1[p], label=lebels[p], color = f'C{p}', ls=':', linewidth=0.6, marker=marks[p], markersize = 3 )
   
    # ax.set_xticks(pnt_list)
    # # pl.xscale('log')

    ax.legend(loc='best')
    
    ax.set_title(f"Amplitudes for V=1.8 and different system sizes (sorted)(not scaled)")
    ax.set_ylabel(r"$|a_n| $", rotation='horizontal', fontsize='large')    
    ax.set_yscale('log')   #ax.set_xscale('log')
    ax.set_ylim(10**-12,5)
    ax.grid(which='major', axis='y', linestyle=':')
    ax.set_xlabel(r"$\frac{n}{N}$ as in $| \frac{n}{N} \rangle $")    
    ax.set_xlim(-1,5e+2)

    plt.savefig(f"Superposition/output/NotScalingSuperAmps_V18_(sorted)_J{job_number}.pdf", dpi=300, bbox_inches = 'tight')


if False: ####################################################################################################################
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    hf_data0, hf_data1 = [], []
    ff_data0, ff_data1 = [], []
    inf_data = []
    for nome in all_files:
        if nome[:3] == "TRN":
            data = np.load(folder_path + nome)
            # print("   - ",nome)
            # print("")
            # print("   - ",data)
            # print("")
            
            hf_data0.append(data[0])
            hf_data1.append(data[1])
            ff_data0.append(data[2])
            ff_data1.append(data[3])    
            # all_data.append(data)
        
        if nome[:3] == "IFD":
            data = np.load(folder_path + nome)
            inf_data.append(data)
            # print("   - ",nome)
            # print("")
            # print("   - ",data)
            # print("")
                    
    # cores = tuple(f'C{indx}' for indx in range() )
    num_dims = 14
    maxdim = sp.special.binom(L, L//2)
    pnt_list = np.linspace(1,maxdim, num_dims, dtype=np.int64)

    # fig = plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots( figsize=(13, 6))

    ax.plot(pnt_list, hf_data0, label=f'HF V=0.2', ls='--',linewidth=0.8, marker='s')
    ax.plot(pnt_list, hf_data1, label=f'HF V=1.8', ls='--',linewidth=0.8, marker='D')
    ax.plot(pnt_list, ff_data0, label=f'FF V=0.2', ls='--',linewidth=0.8, marker='s', fillstyle='none')
    ax.plot(pnt_list, ff_data1, label=f'FF V=1.8', ls='--',linewidth=0.8, marker='D', fillstyle='none')

    ax.set_title(f"Superposition energy convergence FF")
    ax.set_xlabel(r"number of states in superposition basis")    
    ax.set_xticks(pnt_list)
    # pl.xscale('log')

    ax.set_ylabel(r"$|\frac{E - E_{ed}}{E_{ed}}| $")    
    ax.set_yscale('log')
    ax.grid(which='major', axis='both', linestyle=':')

    ax.legend(loc='best')
    
    
    ins_ax = ax.inset_axes([0.04,0.06,0.5,0.45])#width="20%", height="20%", loc="lower right", borderpad=1)
    ins_ax.set_yscale('log')
    ins_ax.grid(which='major', axis='both', linestyle=':')
    ins_ax.tick_params(direction='in', labelsize=8)
    ins_ax.set_title(r"$1-\mathcal{F}$")
    ins_ax.plot(pnt_list, inf_data[0], label=f'HF V={phys0[0]:.1f}', ls=':',linewidth=0.6, marker='s') #color='C0',
    ins_ax.plot(pnt_list, inf_data[1], label=f'HF V={phys1[0]:.1f}', ls=':',linewidth=0.6, marker='D')
    ins_ax.plot(pnt_list, inf_data[2], label=f'FF V={phys0[0]:.1f}', ls=':',linewidth=0.6, marker='s', fillstyle='none')
    ins_ax.plot(pnt_list, inf_data[3], label=f'FF V={phys1[0]:.1f}', ls=':',linewidth=0.6, marker='D', fillstyle='none')
    
    plt.savefig(f"Superposition/output/SuperEnrgyConverg_L{L}_(HF,FF)_J{job_number}.pdf", bbox_inches = 'tight')

if False:
    
    t_i = tt.time()
    
    Vs =  [0.2, 0.6, 1.0, 1.4, 1.8] 

    
    fig = plt.figure(figsize=(12, 5))
    # blues = pl.cm.get_cmap('Blues_r',num_plts+3)
    # blues = pl.cm.get_cmap('autumn',num_plts)
    marks = ['o','*','^','v','p','s','D','h','8']
    clr = 0
    # all_data = []
    for nome in all_files:
        if nome[:5] == "FMPSS":
            data = np.load(folder_path + nome)
            # print("   - ",nome)
            # print("")
            # print("   - ",data)
            # print("")
            # plt.plot(data, label=f'V={Vs[clr]:.2f}', color=f'C{clr}', marker=marks[clr], linestyle='', alpha=1-(clr *0.75)/len(Vs))
            plt.plot( data[1:], label=f'V={data[0]:.2f}', color=f'C{clr}', marker=marks[clr], linestyle='', alpha=1-(clr *0.75)/len(Vs))
            # plt.plot( np.flip(data[1:]), label=f'V={data[0]:.2f}', color=f'C{clr}', marker=marks[clr], linestyle='', alpha=1-(clr *0.75)/(L+1))

            clr += 1

    plt.title(f"Amplitudes for different V's (energy sorted)")
    plt.xlabel(r"$n$ as in $|n\rangle $", fontsize='large')    
    plt.ylabel(r"$|a_n| $", rotation='horizontal', fontsize='large')    
    plt.yscale('log')
    plt.ylim(10**-12,10)
    plt.legend(loc='best',ncols=2)
    plt.grid(which='major', axis='y', linestyle=':')
    plt.savefig(f"Superposition/output/SuperAmps_L{L}_(FF)(energy sorted)_J{job_number}.pdf", dpi=300, bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Plotting Run Time: ", tt.time() - t_i,"(s)")
    print("")

if False:
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    x = np.arange(1,maxdim+1,1)
    y = np.arange(1,maxdim+1,1)
    
    t_i = tt.time()
    
    # hf_data0, hf_data1 = [], []
    # ff_data0, ff_data1 = [], []
    # hf_data, ff_data = [], []
    
    all_data = []
    for nome in all_files:
        if nome[:3] == "MEP":
            data = np.load(folder_path + nome)
            print("   - ",nome)
            print("")
            # print("   - ",data)
            # print("")
            
            # hf_data1.append(data[1])
            # ff_data0.append(data[2])
            # ff_data1.append(data[3])    
            all_data.append(data)
        
    # v_fac0 =  np.round(np.max([abs(np.max(all_data[0])), abs(np.min(all_data[0]))]), decimals = 8)
    # v_fac1 =  np.round(np.max([abs(np.max(all_data[1])), abs(np.min(all_data[1]))]), decimals = 8)
    v_fac2 =  np.round(np.max([abs(np.max(all_data[2])), abs(np.min(all_data[2]))]), decimals = 8)
    v_fac3 =  np.round(np.max([abs(np.max(all_data[3])), abs(np.min(all_data[3]))]), decimals = 8)
    # v_fac0 =  np.round(np.max([abs(np.max(supmat0)), abs(np.min(supmat0))]), decimals = 5)
    
    # fac = np.max([v_fac0,v_fac1])
    # hf_fac = np.max([v_fac0,v_fac1])
    ff_fac = np.max([v_fac2,v_fac3])

    cbar_tiks = [-1.e+1, -1.e-0, -1.e-2, -1.e-4, -1.e-6, -1.e-8, -1.e-10, 0, 1.e-10, 1.e-8, 1.e-6, 1.e-4, 1.e-2, 1.e-0, 1.e+1]

    # t_i = tt.time()

    fig, axs = plt.subplots(1,2, figsize=(8, 4), sharey=True, sharex=True )
    

    # hf_norm=colors.SymLogNorm(linthresh=1.e-10, linscale=1e-1, vmin=-hf_fac, vmax=hf_fac)
    ff_norm=colors.SymLogNorm(linthresh=1.e-8, linscale=1e-1, vmin=-ff_fac, vmax=ff_fac)
    
    # im00 = axs[0,0].pcolormesh(x, y, np.flipud(all_data[0]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[0,0].set_title(f'V={phys0[0]}', pad=0)
    # axs[0,0].set_ylabel('HF method', fontsize=14)
    # im01 = axs[0,1].pcolormesh(x, y, np.flipud(all_data[1]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[0,1].set_title(f'V={phys1[0]}', pad=0) 
    # im00 = axs[0].pcolormesh(x, y, np.flipud(all_data[0]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[0].set_title(f'V={phys0[0]}', pad=0)
    # axs[0].set_ylabel('HF method', fontsize=14)
    # im01 = axs[1].pcolormesh(x, y, np.flipud(all_data[1]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[1].set_title(f'V={phys1[0]}', pad=0) 
    
    # im10 = axs[1,0].pcolormesh(x, y, np.flipud(all_data[2]), norm=ff_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[1,0].set_ylabel('FF method', fontsize=14)
    # im11 = axs[1,1].pcolormesh(x, y, np.flipud(all_data[3]), norm=ff_norm, cmap='bwr', shading='nearest', rasterized=True)
    im10 = axs[0].pcolormesh(x, y, np.flipud(all_data[2]), norm=ff_norm, cmap='bwr', shading='nearest', rasterized=True)
    axs[0].set_ylabel('FF method', fontsize=14)
    axs[0].set_title(f'V={phys0[0]}', pad=0)
    im11 = axs[1].pcolormesh(x, y, np.flipud(all_data[3]), norm=ff_norm, cmap='bwr', shading='nearest', rasterized=True)
    axs[1].set_title(f'V={phys1[0]}', pad=0) 
     
    fig.tight_layout()

    # bbox_ax_top = axs[1].get_position()
    bbox_ax_bottom = axs[1].get_position()

    # cbar_im01_ax = fig.add_axes([1.01, bbox_ax_top.y0, 0.02, bbox_ax_top.y1-bbox_ax_top.y0])
    # # cbar_im01 = plt.colorbar(im01, cax=cbar_im01_ax)
    # cbar_im01 = plt.colorbar(im01, cax=cbar_im01_ax, fraction=0.0215, pad=0.04, ticks=cbar_tiks)#orientation='vertical')

    cbar_im11_ax = fig.add_axes([1.01, bbox_ax_bottom.y0, 0.02, bbox_ax_bottom.y1-bbox_ax_bottom.y0])
    # cbar_im11 = plt.colorbar(im11, cax=cbar_im11_ax)
    cbar_im11 = plt.colorbar(im11, cax=cbar_im11_ax, fraction=0.0215, pad=0.04, ticks=cbar_tiks)#orientation='vertical')
    
    
    # axs[0,0].set_xticklabels([])
    # axs[0,0].set_yticklabels([])
    # axs[1,0].set_xticklabels([])
    # axs[1,0].set_yticklabels([])
    # axs[1,1].set_xticklabels([])
    # axs[1,1].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    # fig.colorbar(im00, ax=axs, fraction=0.0215, pad=0.04, ticks=[-10,-1.0,-0.1,-0.01,-1.e-3,-1.e-6,0,1.e-6,1.e-5,1.e-3,0.01,0.1,1.0,10])#orientation='vertical')
    fig.suptitle(f'Plot of matrix elements', fontsize=14, y=1.02)

    plt.savefig(f"Superposition/output/SuperMatElement_L{L}_(FF)_J{job_number}.pdf", dpi=300, bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Matrix Element Ploting Time: ", tt.time() - t_i,"(s)")
    print("")
    

if False: ################################################################# PLOTING SHADOW DATA WITH and/or WITHOUT MAT ELEMNETS
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    x = np.arange(1,maxdim+1,1)
    y = np.arange(1,maxdim+1,1)
    
    t_i = tt.time()
    
    # hf_data0, hf_data1 = [], []
    # ff_data0, ff_data1 = [], []
    # hf_data, ff_data = [], []
    sdw_data = []
    all_data = []
    for nome in all_files:
        # if nome[:3] == "MEP":
        if nome[:7] == "MEP__01":
            data = np.load(folder_path + nome)
            print("   - ",nome)
            print("")

            all_data.append(data)
        # if nome[:3] == "SHW":
        if nome[:7] == "SHW__01":
            data = np.load(folder_path + nome)
            print("   - ",nome)
            print("")

            sdw_data.append(data)
        
    v_fac0 =  np.round(np.max([abs(np.max(all_data)), abs(np.min(all_data))]), decimals = 8)
    v_fac1 =  np.round(np.max([abs(np.max(all_data)), abs(np.min(all_data))]), decimals = 8)
    # v_fac0 =  np.round(np.max([abs(np.max(all_data[0])), abs(np.min(all_data[0]))]), decimals = 8)
    # v_fac1 =  np.round(np.max([abs(np.max(all_data[1])), abs(np.min(all_data[1]))]), decimals = 8)
    # v_fac2 =  np.round(np.max([abs(np.max(sdw_data[1])), abs(np.min(all_data[2]))]), decimals = 8)
    # v_fac3 =  np.round(np.max([abs(np.max(all_data[3])), abs(np.min(all_data[3]))]), decimals = 8)
    # v_fac0 =  np.round(np.max([abs(np.max(supmat0)), abs(np.min(supmat0))]), decimals = 5)
    
    # fac = np.max([v_fac0,v_fac1])
    hf_fac = np.max([v_fac0,v_fac1])
    # ff_fac = np.max([v_fac2,v_fac3])

    cbar_tiks = [-1.e+1, -1.e-0, -1.e-2, -1.e-4, -1.e-6, -1.e-8, -1.e-10, 0, 1.e-10, 1.e-8, 1.e-6, 1.e-4, 1.e-2, 1.e-0, 1.e+1]

    # t_i = tt.time()

    fig, axs = plt.subplots(1,2, figsize=(8, 4), sharey=True, sharex=True )
    

    hf_norm=colors.SymLogNorm(linthresh=1.e-10, linscale=1e-1, vmin=-hf_fac, vmax=hf_fac)
    # ff_norm=colors.SymLogNorm(linthresh=1.e-8, linscale=1e-1, vmin=-ff_fac, vmax=ff_fac)
    
    # im00 = axs[0,0].pcolormesh(x, y, np.flipud(all_data[0]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[0,0].set_title(f'V={phys0[0]}', pad=0)
    # axs[0,0].set_ylabel('HF method', fontsize=14)
    # im01 = axs[0,1].pcolormesh(x, y, np.flipud(all_data[1]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[0,1].set_title(f'V={phys1[0]}', pad=0) 
    
    im00 = axs[0].pcolormesh(x, y, np.flipud(sdw_data[0]), norm=hf_norm, cmap='binary', shading='nearest', rasterized=True)
    axs[0].set_title(f'V={phys1[0]}', pad=0)
    axs[1].set_ylabel('Allowed matrix elements', fontsize=14)
    im01 = axs[1].pcolormesh(x, y, np.flipud(all_data[0]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    axs[1].set_title(f'V={phys1[0]}', pad=0) 
    axs[0].set_ylabel('HF matrix elements', fontsize=14)
    
    # im10 = axs[1,0].pcolormesh(x, y, np.flipud(all_data[2]), norm=ff_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[1,0].set_ylabel('FF method', fontsize=14)
    # im11 = axs[1,1].pcolormesh(x, y, np.flipud(all_data[3]), norm=ff_norm, cmap='bwr', shading='nearest', rasterized=True)
    # im10 = axs[0].pcolormesh(x, y, np.flipud(all_data[1]), norm=hf_norm, cmap='bwr', shading='nearest', rasterized=True)
    # axs[0].set_ylabel('FF method', fontsize=14)
    # axs[0].set_title(f'V={phys0[0]}', pad=0)
    
     
    fig.tight_layout()

    bbox_ax_top = axs[1].get_position()
    # bbox_ax_bottom = axs[1].get_position()

    cbar_im01_ax = fig.add_axes([1.01, bbox_ax_top.y0, 0.02, bbox_ax_top.y1-bbox_ax_top.y0])
    # # cbar_im01 = plt.colorbar(im01, cax=cbar_im01_ax)
    cbar_im01 = plt.colorbar(im01, cax=cbar_im01_ax, fraction=0.0215, pad=0.04, ticks=cbar_tiks)#orientation='vertical')

    # cbar_im11_ax = fig.add_axes([1.01, bbox_ax_bottom.y0, 0.02, bbox_ax_bottom.y1-bbox_ax_bottom.y0])
    # # cbar_im11 = plt.colorbar(im11, cax=cbar_im11_ax)
    # cbar_im11 = plt.colorbar(im11, cax=cbar_im11_ax, fraction=0.0215, pad=0.04, ticks=cbar_tiks)#orientation='vertical')
    
    
    # axs[0,0].set_xticklabels([])
    # axs[0,0].set_yticklabels([])
    # axs[1,0].set_xticklabels([])
    # axs[1,0].set_yticklabels([])
    # axs[1,1].set_xticklabels([])
    # axs[1,1].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    # fig.colorbar(im00, ax=axs, fraction=0.0215, pad=0.04, ticks=[-10,-1.0,-0.1,-0.01,-1.e-3,-1.e-6,0,1.e-6,1.e-5,1.e-3,0.01,0.1,1.0,10])#orientation='vertical')
    fig.suptitle(f'Plot of matrix elements', fontsize=14, y=1.02)

    plt.savefig(f"Superposition/output/Shadow_MatEle_L{L}_(HF)_J{job_number}.pdf", dpi=300, bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Matrix Element Ploting Time: ", tt.time() - t_i,"(s)")
    print("")
    
if False: ################################################################# PLOTING SHADOW DATA WITH and/or WITHOUT MAT ELEMNETS
    
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0
    
    x = np.arange(1,maxdim+1,1)
    y = np.arange(1,maxdim+1,1)
    
    t_i = tt.time()
    

    sdw_data = []
    all_data = []
    for nome in all_files:
        # if nome[:3] == "MEP":
        if nome[:7] == "MEP__01":
            data = np.load(folder_path + nome)
            print("   - ",nome)
            print("")

            all_data.append(data)
        # if nome[:3] == "SHW":
        if nome[:7] == "SHW__01":
            data = np.load(folder_path + nome)
            print("   - ",nome)
            print("")

            sdw_data.append(data)
        
    new_data = np.where(sdw_data == 1., all_data, 0.0)
    new_data2 = np.where(new_data > 0.0, 1 , 0.0)

    fig, axs = plt.subplots(1,1, figsize=(8, 8), sharey=True, sharex=True )
    

    # hf_norm=colors.SymLogNorm(linthresh=1.e-10, linscale=1e-1, vmin=-hf_fac, vmax=hf_fac)
    # ff_norm=colors.SymLogNorm(linthresh=1.e-8, linscale=1e-1, vmin=-ff_fac, vmax=ff_fac)
    
    im00 = axs.pcolormesh(x, y, np.flipud(new_data[0]), cmap='binary', shading='nearest', rasterized=True)
    axs.set_title(f'V={phys1[0]}', pad=0)
    axs.set_ylabel('Allowed matrix elements', fontsize=14)
    # im01 = axs[1].pcolormesh(x, y, np.flipud(all_data[0]), cmap='bwr', shading='nearest', rasterized=True)
    # axs[1].set_title(f'V={phys1[0]}', pad=0) 
    # axs[0].set_ylabel('HF matrix elements', fontsize=14)
    
     
    fig.tight_layout()

    # bbox_ax_top = axs[1].get_position()
    # bbox_ax_bottom = axs[1].get_position()

    # cbar_im01_ax = fig.add_axes([1.01, bbox_ax_top.y0, 0.02, bbox_ax_top.y1-bbox_ax_top.y0])
    # # cbar_im01 = plt.colorbar(im01, cax=cbar_im01_ax)
    # cbar_im01 = plt.colorbar(im01, cax=cbar_im01_ax, fraction=0.0215, pad=0.04, ticks=cbar_tiks)#orientation='vertical')

    # cbar_im11_ax = fig.add_axes([1.01, bbox_ax_bottom.y0, 0.02, bbox_ax_bottom.y1-bbox_ax_bottom.y0])
    # # cbar_im11 = plt.colorbar(im11, cax=cbar_im11_ax)
    # cbar_im11 = plt.colorbar(im11, cax=cbar_im11_ax, fraction=0.0215, pad=0.04, ticks=cbar_tiks)#orientation='vertical')
    
    
    # axs[0,0].set_xticklabels([])
    # axs[0,0].set_yticklabels([])
    # axs[1,0].set_xticklabels([])
    # axs[1,0].set_yticklabels([])
    # axs[1,1].set_xticklabels([])
    # axs[1,1].set_yticklabels([])
    axs.set_xticklabels([])
    axs.set_yticklabels([])

    # fig.colorbar(im00, ax=axs, fraction=0.0215, pad=0.04, ticks=[-10,-1.0,-0.1,-0.01,-1.e-3,-1.e-6,0,1.e-6,1.e-5,1.e-3,0.01,0.1,1.0,10])#orientation='vertical')
    fig.suptitle(f'Plot of matrix elements', fontsize=14, y=1.02)

    plt.savefig(f"Superposition/output/Shadow_MatEle_test_L{L}_(HF)_J{job_number}.pdf", dpi=300, bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Matrix Element Ploting Time: ", tt.time() - t_i,"(s)")
    print("")
    

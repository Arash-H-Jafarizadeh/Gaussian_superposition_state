########################################################################################################################################################################################
################################################################################# Join & Plot 19022025 #################################################################################
########################################################################################################################################################################################
"""
V:1.0 created on 19022025
V:1.0 modified on 28032025 (just ploting)
loading each data to:
    - plot the matrix elemets in black and whit (first if)
    - plot of truncated energy convergence and infidality (second if)
    - polt of error scaling with L's and V's
    - plot of amplitudes 
    ...
"""

import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import glob
import sys

if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0


# folder_path = 'raw_data/'
# folder_path = '/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/raw_data/'



if False: ################################################################################## plotting matrix elements in BW w#############################################
    folder_path = 'Superposition_run/raw_data/'
    all_files =  sorted( glob.glob('FulMat'+'*.npy', root_dir=folder_path) )
    
    nome = all_files[array_number]
    
    Vs = float(nome[8:11])
    Ls = int(nome[12:14])
        
    print(f"- Loading Full Matrix for L={Ls} and V={Vs} - data file name:",nome)
    print("")
    
    t_i = tt.time()
    data = np.abs(np.load(folder_path + nome))
    print(f"- Full Matrix Loding Time: ", tt.time() - t_i,"(s)")
    print("")
    
    # print("data max and min ", np.max(data)," , ",np.min(data[np.nonzero(data)]))
    # print("")
    vmax = np.max(data)
    vmin = 1.e-10
    # pnorm=colors.SymLogNorm(linthresh=1.e-10, linscale=1e-0, vmin=vmix, vmax=vmix)
    pnorm=colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    
    maxdim = sp.special.binom(Ls, Ls//2)    
    
    x = np.arange(1,maxdim+1,1)
    y = np.arange(1,maxdim+1,1)
    
    t_i = tt.time()    
    
    fig, ax = plt.subplots( figsize=(10, 8), sharey=True, sharex=True )
    fig.tight_layout()
    
    img = ax.pcolormesh(x, y, np.flipud(data), norm=pnorm, cmap='binary', shading='nearest', rasterized=True)
    # img = ax.pcolormesh(x, y, np.flipud(data), cmap='bwr', shading='nearest', rasterized=True)
    ax.set_title(f'Abs of matrix elements for L={Ls} and V={Vs} (log)')
    # ax.set_ylabel('matrix elements', fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # cbar_tiks = [-1.e+1, -1.e-0, -1.e-2, -1.e-4, -1.e-6, -1.e-8, -1.e-10, 0, 1.e-10, 1.e-8, 1.e-6, 1.e-4, 1.e-2, 1.e-0, 1.e+1]
    fig.colorbar(img, ax=ax)#, ticks=cbar_tiks, extend='max')

    # plt.savefig(f"Superposition_run/output/MatrixElement_L{Ls:02}_V{Vs}_(log).png", dpi=400, bbox_inches = 'tight')

    print("- Matrix Element Ploting Time: ", tt.time() - t_i,"(s)")
    print("")
    
    
if False: ######################################################################### Ploting energy-convergence/density-convergence for fixed V and different L - 10022025
    
    """
    This bid plots the energy convergence (also density convergence) for fixed V and different L
    """
    folder_path = 'Superposition_run/raw_data/'
    all_files =  sorted( glob.glob('TRNC'+'*.npy', root_dir=folder_path) )[:55]
    print(all_files)
    print("")
    all_files = np.reshape(all_files, (5,11))
    print(all_files)
    print("")
    
    # marks = ['o','s','^','v','D','p','.','h','8']
    # marks = ['None','None','None','None','None','None','None','.','None']
    
    fig, ax = plt.subplots(5,1, figsize=(8, 21), sharex=True, 
            gridspec_kw=dict( hspace=0.0,
                ),
            subplot_kw=dict( 
                xscale = 'linear', yscale ='log',               
                ylabel = r'$|\frac{ E-E_{_*} }{ L }|$', #r'$|\frac{ E-E_{_*} }{ E_{_*} }|$', 
                xlabel = r'$\frac{ \# \text{states} }{ N_L }$',
                ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                yticks=[10**(-s) for s in range(2,17,2)],
                # title = ["Hello world","1","2","3","4"],         
                )
            )
    
    
    fig_num = 0
    for file_set in all_files:
        print("- data for figure name:", file_set )
        
        for pnt, nome in enumerate(file_set):
            print("- - file name:", nome )
            Vs = float(nome[5:8])
            Ls = int(nome[9:11])
            maxdim = sp.special.binom(Ls, Ls//2)    
        
            data = np.load(folder_path + nome)
            gsE = np.load(folder_path + f"EDGS_{Vs}_{Ls:02}.npy") 

            y_data = np.abs( data[0] - gsE) / np.abs(Ls) # np.abs(gsE)
            y_data[-1] = 1.e-16 if y_data[-1] == 0.0 else y_data[-1]
            x_data = data[1] / maxdim
            
            # ax[fig_num].plot(x_data, y_data, label=f"L={Ls}", color = f'C{pnt}', ls='--', linewidth=0.7, marker=marks[pnt], markersize=4 )
            ax[fig_num].plot(x_data, y_data, label=f"L={Ls}", color = f'C{pnt}', ls='--', linewidth=0.5, marker='.', markersize=5)
            ax[fig_num].set_title(f"V={Vs}", y = 0.91)
            # ax[fig_num].legend(loc='best')
            ax[fig_num].grid(which='major', axis='both', linestyle=':')
            ax[fig_num].tick_params(direction='in', labelsize=7)
            
        fig_num += 1
        
            
    ax[0].legend(loc='best', ncol=2)
    ax[4].legend(loc='best', ncol=2)
            
    
    
    fig.suptitle(r"Energy density error ($|\frac{E-E_{*}}{L}|$) for different L's and V's ($E_{_*}=E_{dmrg}$ or $E_{ed}$)", fontsize='large', y=0.9001)
    # fig.suptitle(r"Energy error ($|\frac{E-E_{*}}{E_{*}}|$) for different L's and V's", fontsize='large', y=0.9001)

    plt.savefig(f"Superposition_run/output/EnergyDensity_vs_L_fix_V.pdf", bbox_inches = 'tight')


if False: ######################################################################### Ploting Energy convergenece for fixed L and different V - 23022025
    
    """
    This bid plots the                for fixed V and different L
    """
    folder_path = 'Superposition_run/raw_data/'
    all_files =  sorted( glob.glob('TRNC'+'*.npy', root_dir=folder_path) )[:55]
    all_files = np.reshape(all_files, (5,11)).T
    # print(all_files)
    
    marks = ['o','s','^','v','D','p','h','8']
    
    fig, ax = plt.subplots(6, 1, figsize=(8, 21), sharex=True, 
            gridspec_kw=dict( hspace=0.05,
                ),
            subplot_kw=dict( 
                xscale = 'linear', yscale ='log',               
                ylabel = r'$|\frac{ E-E_{_*} }{ L }|$', #r'$|\frac{ E-E_{*} }{ E_{*} }|$', 
                xlabel = r'$\frac{ \# \text{states} }{ N_L }$',
                # ylim = (1.e-13, 1e-2), #xlim = (1.e-12,100),   
                yticks=[10**(-s) for s in range(2,17,2)],
                )
            )
    
    
    # p1,p2,p3,p4,p5 = 0,0,0,0,0
    fig_num = 0
    for file_set in all_files[2:-3]:
        print("- data for figure name:", file_set )
        
        for pnt, nome in enumerate(file_set):
            print("- - file name:", nome )
            Vs = float(nome[5:8])
            Ls = int(nome[9:11])
            maxdim = sp.special.binom(Ls, Ls//2)    
        
            data = np.load(folder_path + nome)
            gsE = np.load(folder_path + f"EDGS_{Vs}_{Ls:02}.npy") # gsE = np.load(folder_path + "EDGS"+nome[4:])

            y_data = np.abs( data[0] - gsE) / Ls # np.abs(gsE)  
            y_data[-1] = 1.e-16 if y_data[-1] == 0.0 else y_data[-1]
            x_data = data[1] / maxdim
            
            # ax[fig_num].plot(x_data, y_data, label=f"V={Vs}", color = f'C{pnt}', ls='--', linewidth=0.8, marker=marks[pnt], markersize=4 )
            ax[fig_num].plot(x_data, y_data, label=f"V={Vs}", color = f'C{pnt}', ls='--', linewidth=0.5, marker='.', markersize=5 )
            ax[fig_num].set_title(f"L={Ls}", y = 0.91)
            ax[fig_num].set_ylim(1.e-12, 1.e-2)
            ax[fig_num].grid(which='major', axis='both', linestyle=':')
            ax[fig_num].tick_params(direction='in', labelsize=7)
            
        fig_num += 1
        
        
    ax[0].legend(loc='best')
    # ax[3].legend(loc='best')
    ax[-1].legend(loc='best')
    
    fig.suptitle(r"Energy error for different V and fixed L ($E_{_*}=E_{dmrg}$ or $E_{ed}$)", fontsize='large', y=0.9001)
    # fig.suptitle(r"Energy convergence ($|\frac{E-E_{ed}}{E_{ed}}|$) for different V's and L's", fontsize='large', y=0.91)
    
    plt.savefig(f"Superposition_run/output/EnergyError_vs_V_fix_L.pdf", bbox_inches = 'tight')



if False: ######################################################################### Ploting Infidality for fixed V and different L - 23022025
    folder_path = 'Superposition_run/raw_data/'
    all_files =  sorted( glob.glob('INFD'+'*.npy', root_dir=folder_path) )
    all_files = np.reshape(all_files, (5,5))
    
    marks = ['o','s','^','v','D','p','h','8']
    
    fig, ax = plt.subplots(5,1, figsize=(8, 22), sharex=True, 
            gridspec_kw=dict( hspace=0.0,
                ),
            subplot_kw=dict( 
                xscale = 'linear', yscale ='log',               
                ylabel = r'$1-\mathcal{F}$', #r'$|\frac{ E-E_{ed} }{ L }|$', #r'$|\frac{ E-E_{ed} }{ E_{ed} }|$', 
                xlabel = r'$\frac{ \# \text{states} }{ N_L }$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                yticks=[10**(-s) for s in range(2,17,2)],         
                )
            )
    
    
    fig_num = 0
    for file_set in all_files:
        # print("- data for figure name:", file_set )
        for pnt, nome in enumerate(file_set):
            print("- - file name:", nome )
            Vs = float(nome[5:8])
            Ls = int(nome[9:11])
            maxdim = sp.special.binom(Ls, Ls//2)    
        
            data = np.load(folder_path + nome)
            other_data = np.load(folder_path + f"TRNC_{Vs}_{Ls:02}.npy")

            y_data = data
            x_data = other_data[1] / maxdim
            
            ax[fig_num].plot(x_data, y_data, label=f"L={Ls}", color = f'C{pnt}', ls='--', linewidth=0.8, marker=marks[pnt], markersize=4 )
            ax[fig_num].set_title(f"V={Vs}", y = 0.91)
            ax[fig_num].grid(which='major', axis='both', linestyle=':')
            ax[fig_num].tick_params(direction='in', labelsize=7)
            
        fig_num += 1
        
            
    ax[0].legend(loc='best')
    ax[-1].legend(loc='best')
            
    fig.suptitle(r"Infidality for different L's and V's", fontsize='large', y=0.888)

    plt.savefig(f"Superposition_run/output/Infidality_vs_L_fix_V.pdf", bbox_inches = 'tight')


if False: ######################################################################### Ploting Amplitudes for fixed L and different V - 26022025
    kink_vline = False
    error_vline = True
    
    folder_path = 'Superposition_run/raw_data/Amplitudes'
    # all_files =  sorted( glob.glob('AMPS'+'*.npy', root_dir=folder_path) )[:55]
    # all_files = np.reshape(all_files, (5,11)).T
    
    all_files =  np.array([
                            ['AMPS_0.1_12.npy','AMPS_0.2_12.npy','AMPS_0.3_12.npy','AMPS_0.4_12.npy','FulMat__0.5_12.npy'],
                            ['AMPS_0.1_14.npy','AMPS_0.2_14.npy','AMPS_0.3_14.npy','AMPS_0.4_14.npy','FulMat__0.5_14.npy'],
                            ['AMPS_0.1_16.npy','AMPS_0.2_16.npy','AMPS_0.3_16.npy','AMPS_0.4_16.npy','FulMat__0.5_16.npy'],
                            # ['AMPS_0.1_18.npy','AMPS_0.2_18.npy','AMPS_0.3_18.npy','AMPS_0.4_18.npy','FulMat__0.5_18.npy'],
                            #['AMPS_0.1_26.npy','AMPS_0.2_26.npy','AMPS_0.3_26.npy','AMPS_0.4_26.npy','FulMat__0.5_26.npy'],
                            #['AMPS_0.1_28.npy','AMPS_0.2_28.npy','AMPS_0.3_28.npy','FulMat__0.4_28.npy','FulMat__0.5_28.npy']
                           ])
    
    marks = ['o','s','^','v','D','p','h','8']
    #errors for V lines!
    errors = [1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10]
    
    fig, ax = plt.subplots(3,1, figsize=(8, 14), sharex=False, 
            gridspec_kw=dict( hspace=0.15,
                ),
            subplot_kw=dict( 
                xscale = 'linear', yscale ='log',               
                ylabel = r'$|a_n|$',  
                xlabel = r'$|n \rangle$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                # yticks=[10**(-s) for s in range(2,17,2)],         
                ),
            )
    
    
    fig_num = 0
    for file_set in all_files[:, 0:1]:
        print("- data for figure name:", file_set )
        for pnt, nome in enumerate(file_set):
            print("- - file name:", nome )
            Vs = float(nome[5:8])
            Ls = int(nome[9:11])
            maxdim = sp.special.binom(Ls, Ls//2)    
        
            data = np.load(folder_path + nome)
            
            # ax[fig_num].plot(data, label=f"V={Vs}", color = f'C{pnt}', ls='', marker=marks[pnt], markersize=3, )#alpha=(1.0 - 0.7*pnt))
            ax[fig_num].plot(np.sort(data)[::-1], label=f"V={Vs}", color = f'C{pnt}', ls='', marker=marks[pnt], markersize=3, )#alpha=(1.0 - 0.7*pnt))
            ax[fig_num].set_title(f"L={Ls}", y = 0.899)
            ax[fig_num].grid(which='major', axis='y', linestyle=':')
            # ax[fig_num].tick_params(direction='in', labelsize=7)
            
            trunc_data, gsE = np.load(folder_path + f"TRNC_{Vs}_{Ls:02}.npy"), np.load(folder_path + f"EDGS_{Vs}_{Ls:02}.npy")
            
            if kink_vline:### to get the vertical line for kink position
                
                loop_data = np.log(np.abs( trunc_data[0] - gsE) / np.abs(gsE))
                
                # kink_pos = np.argmax(np.diff(loop_data, n=2)[:-2]) + 1            
                kink_pos = np.max( [np.argmax( np.diff(loop_data, n = 1)[:-1]) +0 , np.argmax( np.diff(loop_data, n = 2)[:-1]) +3] )

                mark_point = int(trunc_data[1][kink_pos])
                ax[fig_num].vlines(mark_point, 0, 1, transform=ax[fig_num].get_xaxis_transform(), colors='b',ls='--', linewidth=0.6,)
                ax[fig_num].text(mark_point, 0.81, "kink", transform=ax[fig_num].get_xaxis_transform(), fontsize=7, rotation=90, color='b')
                # print("- - - - - - - - -", mark_point)
                
            if error_vline: ### to get the vertical line for error values
                
                # trunc_data, gsE = np.load(folder_path + f"TRNC_{Vs}_{Ls:02}.npy"), np.load(folder_path + f"EDGS_{Vs}_{Ls:02}.npy") 
                loop_data = np.abs( trunc_data[0] - gsE) / Ls #np.abs(gsE)
                
                for thrsh in errors:
                    the_pos = len(loop_data[loop_data >= thrsh])
                    mark_point = int(trunc_data[1][the_pos])
                    ax[fig_num].vlines(mark_point, 0, 1, transform=ax[fig_num].get_xaxis_transform(), colors='r',ls='--', linewidth=0.6,)
                    ax[fig_num].text(mark_point, 0.81, str(thrsh), transform=ax[fig_num].get_xaxis_transform(), fontsize=7, rotation=90, color='r', alpha=0.8)
                # print("- - - - - - - - -", mark_point)
            
            
        fig_num += 1
        
            
    ax[0].legend(loc='best')
    ax[-1].legend(loc='best')
            
    fig.suptitle(r"Amplitudes for different L's and V's ($error\:=|\frac{E-E_{ed}}{L}|$)", fontsize='large', y=0.901)

    # plt.savefig(f"Superposition_run/output/_test_Amplitudes_vs_V_fix_L_J{job_number}.pdf", bbox_inches = 'tight')



if False: ######################################################################### Ploting energy-error-scalling for fixed V and different L - 23022025
    
    fitting = True
    scaling = "log-linear" #  "log-log" # 

    folder_path = 'Superposition_run/raw_data/'
    all_files =  sorted( glob.glob('TRNC'+'*.npy', root_dir=folder_path) )[:55]
    all_files = np.reshape(all_files, (5,11))
    
    
    fig, ax = plt.subplots(3,1, figsize=(10, 16), sharex=True, 
            gridspec_kw=dict( hspace=0.0,
                ),
            subplot_kw=dict( 
                yscale = scaling[:3], xscale =scaling[4:],               
                ylabel = r'$\#$states', xlabel = r'$L$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                xticks= [x for x in range(8,30,2)], xticklabels=[str(x) for x in range(8,30,2)],
                # xticks= [8, 10, 12, 14, 16, 18, 20, 22], xticklabels=['8','10','12','14','16','18','20','22'],
                # yticks=[10**(-s) for s in range(2,17,2)],    
                ),
            )
    
    marks = ['o','d','^','v','p','h','>','<','8','s','*','X']
    sizes = [x for x in range(8,30,2)] #[8,10,12,14,16,18,20,22,24,26]
    # errors = [1.e-3, 1.e-4, 5.e-4, 1.e-5, 5.e-5, 1.e-6, 5.e-6, 1.e-7, 5.e-7, 1.e-8, 5.e-8] #, 1.e-9, 1.e-10, 1.e-13]
    errors = [1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-11]
    
    fig_num = 0
    for file_set in all_files[::2]:
        print("- data for figure name:", file_set )
        print("-  - - - - --------------------------------")
        for pnt, thrsh in enumerate(errors):
            print("- - error value:", thrsh )
            y_data = []
            Vs = ""
            for nome in file_set[:]:
                print("- - - file name:", nome )
                Vs = float(nome[5:8])
                Ls = int(nome[9:11])
            
                data = np.load(folder_path + nome)
                gsE = np.load(folder_path + f"EDGS_{Vs}_{Ls:02}.npy")

                # loop_data = np.abs( data[0] - gsE) / np.abs(gsE)
                loop_data = np.abs( data[0] - gsE) / Ls
                
                the_pos = len(loop_data[loop_data >= thrsh])
                # if the_pos == len(loop_data):
                #     the_pos += -1
                # print(f"- - - error={thrsh} position:", the_pos )
                # y_data.append( data[1][the_pos] )
                
                if the_pos != len(loop_data):
                    y_data.append( data[1][the_pos] )
                x_data = sizes[:len(y_data)]
                
            # ax[fig_num].plot(sizes, y_data, label=f"{thrsh}", color = f'C{pnt}', ls='', marker='o', markersize=4 )
            ax[fig_num].plot(x_data, y_data, label=f"{thrsh}", color = f'C{pnt}', ls='', marker=marks[pnt], markersize=4 )
            ax[fig_num].set_title(f"V={Vs}", y = 0.91)
            ax[fig_num].grid(which='major', axis='both', linestyle=':')
            ax[fig_num].tick_params(direction='in', labelsize=8)
            
            if scaling == "log-log" and fitting:
                fitness = np.polyfit(np.log(x_data), np.log(np.array(y_data)), 1)
                lin_data = [np.exp(fitness[1])*(x**fitness[0]) for x in sizes]
                ax[fig_num].plot(sizes, lin_data, label=rf'$L^{{{fitness[0]:.2f}}}$', color = f'C{pnt}', ls='--', linewidth=0.7, marker='None')
            
            if scaling == "log-linear" and fitting:
                fitness = np.polyfit(x_data, np.log(np.array(y_data)), 1)
                lin_data = [np.exp(fitness[1]) * np.exp(x*fitness[0]) for x in sizes]
                ax[fig_num].plot(sizes, lin_data, label=rf'$e^{{{fitness[0]:.2f}L}}$', color = f'C{pnt}', ls='--', linewidth=0.7, marker='None')
            # ax[fig_num].legend(loc='best')
        fig_num += 1
    
        
    if fitting:    
        handles,labels = ax[2].get_legend_handles_labels()
        order = [2*i for i in range(len(errors))]+[2*i+1 for i in range(len(errors))] #[0,2,4,6,8,10,12,1,3,5,7,9,11,13]
        handles = [handles[ii] for ii in order]
        labels = [labels[jj] for jj in order]
                
        ax[0].legend(handles, labels, loc='best', ncol=2, title="Error               fit")
        ax[-1].legend(handles, labels, loc='best', ncol=2, title="Error               fit")
    # else:
    #     ax[0].legend(loc='best', ncol=1, title="Error ")
    #     ax[-1].legend(loc='best', ncol=1, title="Error ")
            
    fig.suptitle(f" Error scaling for different V's ({scaling})", fontsize='large', y=0.9001)

    plt.savefig(f"Superposition_run/output/ErrorScaling_vs_L_fix_V_({scaling})_Some.pdf", bbox_inches = 'tight')


if False: ######################################################################### Ploting the kink-scalling for fixed V and different L - 23022025
    
    fitting = True
    scaling = "log-linear" # "log-log" # "log-linear" #
    
    folder_path = 'Superposition_run/raw_data/'
    all_files =  sorted( glob.glob('TRNC'+'*.npy', root_dir=folder_path) )[:45]
    all_files = np.reshape(all_files, (5,9))
    
    fig, ax = plt.subplots(1,1, figsize=(8, 5), sharex=True, 
            gridspec_kw=dict( hspace=0.0,
                ),
            subplot_kw=dict( 
                yscale = scaling[:3], xscale =scaling[4:],               
                ylabel = r'$\#$states', 
                xlabel = r'$L$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                xticks=[s for s in range(8,26,2)],    xticklabels=[str(s) for s in range(8,26,2)], #['8','10','12','14','16','18','20','22','24'],
                # yticks=[10**(-s) for s in range(2,17,2)],       
                ),
            )
    
    marks = ['o','s','^','v','p','h','8','D']
    sizes = [s for s in range(8,24,2)] #[8,10,12,14,16,18,20]
    
    fig_num = 0
    for file_set in all_files[::2]:
        print("- data for figure name:")
        print(file_set )
        y_data = []
        for nome in file_set[:-1]:
            # print("- - - file name:", nome )
            Vs = float(nome[5:8])
            Ls = int(nome[9:11])
        
            data = np.load(folder_path + nome)
            gsE = np.load(folder_path + f"EDGS_{Vs}_{Ls:02}.npy")

            # loop_data = np.log( np.abs(data[0] - gsE) / Ls)[2:-2] #np.abs(gsE))[2:]
            loop_data = np.log( np.abs(data[0, 2:-1] - gsE) / Ls) #np.abs(gsE))
            
            # diff_data = np.diff(loop_data, n = 2)
            # kink_pos = np.argmax(diff_data[:-1]) + 1
            # y_data.append( data[1][kink_pos] )
            # print(np.max(diff_data[:-1])," - ",np.argmax(diff_data[:-1]))
            # print("******************: ", kink_pos," ->", data[1][kink_pos])
            
            # diff_data1 = np.diff(loop_data, n = 1)
            # diff_data2 = np.diff(loop_data, n = 2)
            # diff_data3 = np.diff(loop_data, n = 3)
            # diff_data4 = np.diff(loop_data, n = 4)
            # diff_data5 = np.diff(loop_data, n = 5)
            # kink_pos_print = [np.argmax(diff_data1[:-1]) ,np.argmax(diff_data2[:-1]) ,np.argmax(diff_data3[:-1]) ,np.argmax(diff_data4[:-1]) ,np.argmax(diff_data5[:-1]) ]
            # print("******************: ", kink_pos_print)
            
            kink_pos = np.max( [np.argmax( np.diff(loop_data, n = 1)[:-1]) +0 , np.argmax( np.diff(loop_data, n = 2)[:-1]) +3] )
            y_data.append( data[1][kink_pos] )
            print(f"******** kink position for L={Ls} and V={Vs}: ", kink_pos)
                

        ax.plot(sizes, y_data, label=f"Vs={Vs}", color = f'C{fig_num}', ls='', marker=marks[fig_num], markersize=5 )
        # ax.set_title(f"V={Vs}", y = 0.91)
        ax.grid(which='major', axis='both', linestyle=':')
        # ax.tick_params(direction='in', labelsize=8)
        
        if scaling == "log-log" and fitting:
            fitness = np.polyfit(np.log(sizes), np.log(np.array(y_data)), 1)
            lin_data = [np.exp(fitness[1])*(x**fitness[0]) for x in sizes]
            ax.plot(sizes, lin_data, label=rf'$L^{{{fitness[0]:.2f}}}$', color = f'C{fig_num}', ls='--', linewidth=0.7, marker='None')
        
        if scaling == "log-linear" and fitting:
            fitness = np.polyfit(sizes, np.log(np.array(y_data)), 1)
            lin_data = [np.exp(fitness[1]) * np.exp(x*fitness[0]) for x in sizes]
            ax.plot(sizes, lin_data, label=rf'$e^{{{fitness[0]:.2f}L}}$', color = f'C{fig_num}', ls='--', linewidth=0.7, marker='None')
        
        fig_num += 1
    
        
    handles,labels = ax.get_legend_handles_labels()
    order = [s for s in range(len(handles)) if s % 2 == 0]+[s for s in range(len(handles)+1) if s % 2 != 0] #[0,2,4,6,8,1,3,5,7,9]
    print('order')
    print(order)
    handles = [handles[ii] for ii in order]
    labels = [labels[jj] for jj in order]
    ax.legend(handles, labels, loc='best', ncol=2, title="Error                fit")
    
    # ax.legend(loc='best', ncol=2, )
            
    fig.suptitle(f"Scaling of the kink for different V's ({scaling})", fontsize='large', y=0.9501)

    plt.savefig(f"Superposition_run/output/KinkScaling_vs_L_fix_V_({scaling})_2025_03_14.pdf", dpi=400, bbox_inches = 'tight')




if False: ######################################################################### TESTETSTETSTETSTETST- 23022025
    
    # fig, ax = plt.subplots(1,1, figsize=(8, 6), sharex=True, 
    #         gridspec_kw=dict( hspace=0.0,
    #             ),
    #         subplot_kw=dict( 
    #             xscale = 'linear', yscale ='log',               
    #             ylabel = r'$\frac{E-E_{ed}}{E_{ed}}$', 
    #             xlabel = r'$\frac{\#\; states}{N_{L}}$',
    #             # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
    #             # xticks=[8, 10, 12, 14, 16],    xticklabels=['8','10','12','14','16'],
    #             # yticks=[10**(-s) for s in range(2,17,2)],
    #             # title = ["Hello world","1","2","3","4"],         
    #             ),
    #         )
    
    # marks = ['o','s','^','v','p','h','8','D']
    # # sizes = [8,10,12,14,16]
    
    # all_files =  np.array([['FulMat__0.1_16.npy','FulMat__0.1_18.npy','FulMat__0.1_20.npy','FulMat__0.1_22.npy'],
    #               ['FulMat__0.2_16.npy','FulMat__0.2_18.npy','FulMat__0.2_20.npy','FulMat__0.2_22.npy'],
    #               ['FulMat__0.3_16.npy','FulMat__0.3_18.npy','FulMat__0.3_20.npy','FulMat__0.3_22.npy'],
    #               ['FulMat__0.4_16.npy','FulMat__0.4_18.npy','FulMat__0.4_20.npy','FulMat__0.4_22.npy']
    #               ])
    
    # num_datas = 30
    
    # # for nome in all_files[0,2]:
    # nome = 'FulMat__0.1_18.npy'    
    # Vs = float(nome[8:11])
    # Ls = int(nome[12:14])
    # maxdim = sp.special.binom(Ls, Ls//2) if Ls <= 16 else 16000   
    # MAXDIM = sp.special.binom(Ls, Ls//2)  
    # print("--: ",maxdim," <-> ", MAXDIM)  
    # print("")
    # # threshod = 1.e-12
    
    # t_i = tt.time()
    # ham = np.load(folder_path + nome)
    # print(f"- Full Matrix Loding Time: ", tt.time() - t_i,f"(s) for L={Ls} and V={Vs}")
    # print("")
        
    # # GS_E = np.load(folder_path + 'EDGS_0.1_18.npy')
    # # print(f"- ED ground state energy: {GS_E}")
    # # print("")

    # T_I = tt.time()
    # trncs = np.linspace(1, maxdim, num_datas, dtype=np.int64)
    # # truncated_energy = []
    # truncated_energy = np.zeros( trncs.shape )
    # for l_idx, l in enumerate(trncs):
    #     t_i = tt.time()
    #     Es, vec = np.linalg.eigh(ham[:l,:l])
    #     # print(f"- - - Eigenvalue Time: ", tt.time() - t_i,"(s) for ", l,"",l/maxdim)
    #     # truncated_energy.append(Es[0])
    #     truncated_energy[l_idx] = Es[0]
    
    # print(f"- All Eigenvalues Time: {tt.time() - T_I} (s) for L={Ls} and V={Vs}")
    # print("")
    
    # # data = np.load(folder_path + nome)
    # # gsE = np.load(folder_path + f"EDGS_{Vs}_{Ls}.npy")
    # # loop_data = np.log(np.abs( data[0] - gsE) / np.abs(gsE))
    # # plot_data = np.abs( np.array(truncated_energy) - 0.5*Ls )/(0.5*Ls) 
    # GS_E = truncated_energy[-3]
    # plot_data = (truncated_energy - GS_E ) / GS_E 

    # ax.plot(trncs/MAXDIM, np.abs(plot_data), label=f"L={Ls}", ls='--', marker=marks[0], markersize=6)
    # # plt.yscale('log')
    # # ax.plot(trncs/MAXDIM, truncated_energy, label=f"L={Ls}", ls='--', marker=marks[0], markersize=6)
    # # plt.yscale('symlog')
    # # plt.grid(which='major', axis='both', linestyle=':')
    
            
    # # ax.plot(data[1], diff_data, label=f"diff", color = f'C{pnt}', ls='--', marker=marks[pnt+1], markersize=4 )
    # # ax.set_title(f"V={Vs}", y = 0.91)
    # ax.grid(which='major', axis='both', linestyle=':')
    
        
    # fig.savefig(f"Superposition_run/output/_test_BIG_SIZE_test_J{job_number}.pdf", bbox_inches = 'tight')


    for nome in ['EDGS_0.1_18.npy','EDGS_0.2_18.npy','EDGS_0.3_18.npy','EDGS_0.4_18.npy']:
        Vs = float(nome[5:8])
        Ls = int(nome[9:10])
        GS_E = np.load(folder_path + nome)
        print(f"- ED ground state energy for L={Ls} and V={Vs}: {GS_E}")



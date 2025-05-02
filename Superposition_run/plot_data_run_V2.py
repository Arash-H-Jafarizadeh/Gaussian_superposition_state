########################################################################################################################################################################################
############################################################################### Plot NEXUS data 20250410 ###############################################################################
########################################################################################################################################################################################
"""
V:2.0 created on 20240410
this script is used to plot the data from the NEXUS algorithm
loading each data to:
    - plot of nexus truncated energy convergence
    - polt of error scaling with L's and V's
    - plot of amplitudes 
    ...
"""

import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import matplotlib.cm as cmaps # type: ignore

import glob
import sys

if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0

sys.path.append('/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/source_code/')
import hartree_fock_function as hf

    
if True: ######################################################################### Ploting energy-density-convergence for fixed V and different L - 20250410
    """
    This bid plots the energy density convergence of nexus algorithm for fixed V and different L's
    """
    
    # Vs = 0.2
    
    folder_path = 'Superposition_run/raw_data/test/'
    all_files =  sorted( glob.glob(f'NXET'+'*.npy', root_dir=folder_path) )[:10]
    print(all_files)
    all_files = np.reshape(all_files,(2,5))
    print(all_files)
    print("")
    # all_files = np.reshape(all_files, (5,11))
    # print(all_files)
    # print("")
    
    # marks = ['o','s','^','v','D','p','.','h','8']
    # marks = ['None','None','None','None','None','None','None','.','None']
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
            gridspec_kw=dict( hspace=0.0,
                ),
            subplot_kw=dict( 
                xscale = 'linear', yscale ='log',               
                ylabel = r'$|\frac{ E-E_{_{ed}} }{ L }|$', #r'$|\frac{ E-E_{_*} }{ E_{_*} }|$', 
                xlabel = r'$\frac{ \# \text{states} }{ N_L }$',
                ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                yticks=[10**(-s) for s in range(2,17,1)],         
                )
            )
    
    
    fig_num = 0
    for file_set in all_files:
        print("- data for figure name:", file_set )
        
        for pnt, nome in enumerate(file_set):#(all_files):#
            print("- - file name:", nome )

            Vs = float(nome[5:8])
            Ls = int(nome[9:11])

            maxdim = sp.special.binom(Ls, Ls//2)    
        
            new_data = np.load(folder_path + nome)
            old_data = np.load(folder_path + f'TRNC_{Vs}_{Ls:02}.npy')
            
            gsE = np.load(f"Superposition_run/raw_data/Ground_State_Energy/EDGS_{Vs}_{Ls:02}.npy") 

            new_y = np.abs( new_data[0] - gsE) / np.abs(Ls) # np.abs(gsE)
            old_y = np.abs( old_data[0] - gsE) / np.abs(Ls) # np.abs(gsE)

            x_data = new_data[1] / maxdim
            x_data2 = old_data[1] / maxdim
            
            ax[fig_num].plot(x_data, new_y, label=f"{Ls}  ", color = f'C{pnt}', ls='-.', linewidth=0.5, marker='*', markersize=4, zorder=1) #f"new={Ls}"
            ax[fig_num].plot(x_data2, old_y, label=" ", color = f'C{pnt}', ls=':', linewidth=0.5, marker='.', markersize=5.5, zorder=-10) # f"old={Ls}"
            ax[fig_num].set_title(f"V={Vs}", y = 0.91)
            ax[fig_num].grid(which='major', axis='both', linestyle=':', linewidth=0.5, alpha=0.7)
            ax[fig_num].tick_params(direction='in', labelsize=7)
            
        fig_num += 1
        
            
    # ax[fig_num].legend(loc='best', ncol=2)
    # ax[4].legend(loc='best', ncol=2)
            
    handles,labels = ax[0].get_legend_handles_labels()
    order = [2*i for i in range(5)]+[2*i+1 for i in range(5)] 
    handles = [handles[ii] for ii in order]
    labels = [labels[jj] for jj in order]
    
    legen_title = " "*4+"L"+" "*3+"new"+" "*5+"old"+" "*3
    ax[0].legend(handles, labels, loc='best', ncol=2, columnspacing=1.0, title=legen_title, fontsize='small', markerfirst=False)
    # ax[1].legend(handles, labels, loc='best', ncol=2, columnspacing=1.0, title=legen_title, fontsize='small', markerfirst=False)

    
    fig.suptitle(r"Comparing new \& old nergy density error ($|\frac{E-E_{ed}}{L}|$) for different L's and V's", fontsize='medium', y=0.91001)

    fig.savefig(f"Superposition_run/plots/EnergyDensity__new_vs_old_fix_V_J{job_number}.pdf", dpi=400, bbox_inches = 'tight')
    # fig.savefig(f"Superposition_run/plots/EnergyDensity__new_vs_old__fix_V_2.png",  dpi=800, bbox_inches='tight')



if False: ######################################################################### Ploting FADED colorfull Amplitudes comparing nexus and blind - 20250400
    """
    This bid plots the colorful amplitudes (faded or full) of nexus algorithm for fixed V and L
    """
    
    # kink_vline = False# error_vline = True

    sorted = False
    cbar_in = False
    plot_type = "faded" #"full" # "full" #  
    
    Vs, Ls = 0.1, 14
    nth = 9
    
    # ########## creating custom color map for colorbar:
    Ncolors = Ls//2 + 1
    colmap = plt.get_cmap('tab10') #'jet'
    new_colors = colmap(np.linspace(0, 1, 10)[:Ncolors])
    custom_cmap = colors.ListedColormap(new_colors) #change name here
    

    data_all = np.load(f'Superposition_run/raw_data/test/BSAM_{Vs}_{Ls:02}.npy', allow_pickle=True)
    
    amps_all = np.abs(data_all[0])
    base_all = data_all[1]

    X_all = np.arange(np.size(amps_all)).astype(int)
    dist_all = np.array(hf.basis_distance(base_all, Ls)).astype(np.int64)
    # dist_all = np.load(f'Superposition_run/raw_data/Amplitudes/DSTN_{Vs}_{Ls:02}.npy')
    
    PN = "N"
    pendix = "not sorted"
    
    if sorted:

        order = np.argsort(amps_all)[::-1]
        
        amps_all = amps_all[order]
        base_all = base_all[order]
        dist_all = dist_all[order]
        
        PN = "S"
        pendix = "sorted"
        
    
    print("len dist_all ", len(dist_all))

    bond_data = np.load(f'Superposition_run/raw_data/test/NXET_{Vs}_{Ls:02}.npy', allow_pickle=True)[1]#[nth]
        
    # amps_some = np.load(f'Superposition_run/raw_data/test/AMNX_{Vs}_{Ls:02}.npy', allow_pickle=True)[nth]
    # X_some = np.arange(np.size(amps_some)).astype(int)
    base_some = np.load(f'Superposition_run/raw_data/test/NXBL_{Vs}_{Ls:02}.npy', allow_pickle=True)[nth]
    print("len base_some ", len(base_some))

    # indx_set = np.array([i for i, x in enumerate(base_all) if x in set(base_some)])
    indx_set = np.nonzero(np.isin(base_all, base_some))[0]
    
    
    amps_some = amps_all[indx_set]
    X_some = X_all[indx_set]
    dist_some = dist_all[indx_set]
    
    # ########## creating smaller custom color map for faded plot:
    # num = len(np.unique(dist_some))#+1
    # some_colors = colmap(np.linspace(0, 1, 10)[:num+(num+1)%2])    
    num = np.unique(dist_some)
    some_colors = colmap(np.linspace(0, 1, 10)[np.ix_(num)])

    some_cmap = colors.ListedColormap(some_colors) #change name here

    
    maxdim = sp.special.binom(Ls, Ls//2)    

    
    marks = ['o','s','^','v','D','p','h','8']
    
    fig, ax = plt.subplots(1,1, figsize=(8, 5), sharex=False, 
            gridspec_kw=dict( hspace=0.15,
                ),
            subplot_kw=dict( 
                xscale = 'linear', yscale ='log',               
                ylabel = r'$|a_n|$',  
                xlabel = r'$|n \rangle$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                # yticks=[10**(-s) for s in range(0,27,2)],         
                ),
            )

    
    if plot_type == "faded":
        im0 = ax.scatter(X_all, amps_all, c=dist_all, s=10, cmap=custom_cmap, edgecolors='none', alpha=0.051, zorder=-10)
        ax.scatter(X_some, amps_some, c=dist_some, s=10, cmap=some_cmap, linewidth=0.05, edgecolors='0',  zorder=10) # lw=0.5, edgecolors='y', 
        
        plot_title = f"Selected amplitudes ({pendix}) for bond |M|={int(bond_data[nth])} ({nth}th) - L={Ls}, V={Vs}"
        plot_name = f"Superposition_run/plots/{PN}_Faded_Colored_Amplitudes_V{Vs}_L{Ls:02}_{nth}"
    
    if plot_type == "full":
        im0 = ax.scatter(X_all, amps_all, c=dist_all, s=10, cmap=custom_cmap, zorder=10)

        plot_title = f"All amplitudes ({pendix}) for L={Ls} and V={Vs}"
        plot_name = f"Superposition_run/plots/{PN}_Full_Colored_Amplitudes_V{Vs}_L{Ls:02}"


    ax.grid(which='major', axis='both', linestyle=':', alpha=0.75)
    ax.tick_params(direction='in', labelsize=8)

    if cbar_in:
        cbar_ax = ax.inset_axes([0.97,0.3, 0.01, 0.65])
        cbar = fig.colorbar(im0, cax=cbar_ax, location='left')#, extend='both', )
    else:
        cbar_ax = ax.inset_axes([1.03,0.195, 0.01, 0.65])
        cbar = fig.colorbar(im0, cax=cbar_ax, location='right')#, extend='both', )
    
    cbar.solids.set(alpha=1)
    n_clusters = Ls//2+1
    tick_locs = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
    tick_labs = [str(x) for x in range(Ls//2+1)]
    cbar.set_ticks(ticks = tick_locs, labels=tick_labs)

    
    fig.suptitle(plot_title, fontsize= 'medium', y=0.95)
    
    # fig.savefig(plot_name+".png",  dpi=800, bbox_inches = 'tight')
    # fig.savefig(plot_name+".pdf",  dpi=600, bbox_inches = 'tight')




if True: ######################################################################### Ploting error-scalling for NEXUS method for fixed V and different L - 20250415
    
    fitting = True
    scaling = "log-linear" # "log-log" #
     
     
    def nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    folder_path = 'Superposition_run/raw_data/test/'
    all_files =  sorted( glob.glob('NXET'+'*.npy', root_dir=folder_path) )[:10]
    all_files = np.reshape(all_files, (2,5))
    print(all_files)
    
    fig, ax = plt.subplots(2,1, figsize=(10, 11), sharex=True, 
            gridspec_kw=dict( hspace=0.0,
                ),
            subplot_kw=dict( 
                yscale = scaling[:3], xscale =scaling[4:],               
                ylabel = r'$\#$states', xlabel = r'$L$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                xticks= [x for x in range(8,18,2)], xticklabels=[str(x) for x in range(8,18,2)],
                # xticks= [8, 10, 12, 14, 16, 18, 20, 22], xticklabels=['8','10','12','14','16','18','20','22'],
                yticks=[10**(-s) for s in range(2,17,2)],    
                ),
            )
    
    marks = ['o','d','^','v','p','h','>','<','8','s','*','X']
    sizes = [x for x in range(8,18,2)] #[8,10,12,14,16,18,20,22,24,26]
    # errors = [1.e-3, 1.e-4, 5.e-4, 1.e-5, 5.e-5, 1.e-6, 5.e-6, 1.e-7, 5.e-7, 1.e-8, 5.e-8] #, 1.e-9, 1.e-10, 1.e-13]
    errors = [10**(-s) for s in range(5, 17, 2)]  
    
    fig_num = 0
    for file_set in all_files:
        print("- data for figure name:", file_set )
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        for pnt, thrsh in enumerate(errors):
            print("- - error value:", thrsh )
            y_data = []
            Vs = ""
            for nome in file_set[:]:
                print("- - - file name:", nome )
                Vs = float(nome[5:8])
                Ls = int(nome[9:11])
            
                data = np.load(folder_path + nome)
                gsE = np.load(f"Superposition_run/raw_data/Ground_State_Energy/EDGS_{Vs}_{Ls:02}.npy") 

                loop_data = np.abs( data[0] - gsE) / Ls #np.abs(gsE)
                
                the_pos = len(loop_data[loop_data >= thrsh])
                # array = np.asarray(loop_data)
                # the_pos = (np.abs(array - thrsh)).argmin()
                print("- - - the pose:", the_pos )
                
                if the_pos != len(loop_data):
                    y_data.append( data[1][the_pos] )
                
                x_data = sizes[:len(y_data)]
                
            # ax[fig_num].plot(sizes, y_data, label=f"{thrsh}", color = f'C{pnt}', ls='', marker='o', markersize=4 )
            ax[fig_num].plot(x_data, y_data, label=f"{thrsh}", color = f'C{pnt}', ls='', marker=marks[pnt], markersize=4 )
            ax[fig_num].set_title(f"V={Vs}", y = 0.91)
            ax[fig_num].grid(which='major', axis='both', linestyle=':')
            ax[fig_num].tick_params(direction='in', labelsize=8)
            
            if scaling == "log-log" and fitting:
                fitness = np.polyfit(np.log(x_data), np.log(np.array(y_data).astype(np.float64)), 1)
                lin_data = [np.exp(fitness[1])*(x**fitness[0]) for x in sizes]
                ax[fig_num].plot(sizes, lin_data, label=rf'$L^{{{fitness[0]:.2f}}}$', color = f'C{pnt}', ls='--', linewidth=0.7, marker='None')
            
            if scaling == "log-linear" and fitting:
                fitness = np.polyfit(x_data, np.log(np.array(y_data).astype(np.float64)), 1)
                lin_data = [np.exp(fitness[1]) * np.exp(x*fitness[0]) for x in sizes]
                ax[fig_num].plot(sizes, lin_data, label=rf'$e^{{{fitness[0]:.2f}L}}$', color = f'C{pnt}', ls='--', linewidth=0.7, marker='None')
            # ax[fig_num].legend(loc='best')
        fig_num += 1
    
        
    if fitting:    
        handles,labels = ax[0].get_legend_handles_labels()
        order = [2*i for i in range(len(errors))]+[2*i+1 for i in range(len(errors))] #[0,2,4,6,8,10,12,1,3,5,7,9,11,13]
        handles = [handles[ii] for ii in order]
        labels = [labels[jj] for jj in order]
        legen_title = " "*5+"error"+" "*12+"fit"+" "*3        
        ax[0].legend(handles, labels, loc='best', ncol=2, columnspacing=1.0, title=legen_title)
        ax[-1].legend(handles, labels, loc='best', ncol=2, columnspacing=1.0, title=legen_title)
    else:
        ax[0].legend(loc='best', ncol=1, title="Error ")
        ax[-1].legend(loc='best', ncol=1, title="Error ")
            
    fig.suptitle(f" Error scaling of new method for different V's ({scaling})", fontsize='large', y=0.9001)
    fig.savefig(f"Superposition_run/plots/ErrorScaling__NEXUS__fix_V_({scaling}).png", dpi=800, bbox_inches = 'tight')

    


    
if False: ######################################################################### Ploting energy-density-convergence + colorbars for fixed V and L - 20250410
    """
    This bid plots the energy density convergence of nexus algorithm for fixed V and different L's
    """
    
    Vs, Ls = 0.3, 16
    
    folder_path = 'Superposition_run/raw_data/test/'
    nome = f'NXET_{Vs}_{Ls:02}.npy'
    print("- file name:", nome )
    
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
            gridspec_kw=dict( hspace=0.07,
                ),
            subplot_kw=dict( 
                xscale = 'linear', #yscale ='log',               
                # ylabel = r'$|\frac{ E-E_{_{ed}} }{ L }|$', #r'$|\frac{ E-E_{_*} }{ E_{_*} }|$', 
                xlabel = r'$|M|$',#r'$\frac{ \# \text{states} }{ N_L }$',
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                )
            )
        
        
    maxdim = sp.special.binom(Ls, Ls//2)    

    new_data = np.load(folder_path + nome)
    old_data = np.load(folder_path + f'TRNC_{Vs}_{Ls:02}.npy')
    
    gsE = np.load(f"Superposition_run/raw_data/Ground_State_Energy/EDGS_{Vs}_{Ls:02}.npy") 

    new_y = np.abs( new_data[0] - gsE) / np.abs(Ls) # np.abs(gsE)
    old_y = np.abs( old_data[0] - gsE) / np.abs(Ls) # np.abs(gsE)

    x_data = new_data[1] #/ maxdim
    
    ax[0].plot(x_data, new_y, label=f"{Ls}  ", color = f'C{0}', ls='-.', linewidth=0.5, marker='*', markersize=4, zorder=1) #f"new={Ls}"
    ax[0].plot(x_data, old_y, label=" ", color = f'C{0}', ls=':', linewidth=0.5, marker='.', markersize=5.5, zorder=-10) # f"old={Ls}"
    
    ax[0].set(
            # title=f"comparing old truncation vs. new method for L={Ls}, V={Vs} & |m|={0}", 
            yscale='log', ylabel=r'$|\frac{E-E_{_{ed}}}{L}|$', ylim=(1.e-17, 1e-4), yticks=[10**(-s) for s in range(4,17,1)],         
            xticks=x_data, 
        )

    ax[0].legend(loc='best', markerscale=0.7, fontsize='small',markerfirst=False)
    
    
    legen_title = " "*4+"L"+" "*3+"new"+" "*5+"old"+" "*3
    ax[0].legend(loc='best', ncol=2, columnspacing=1.0, title=legen_title, fontsize='small', markerfirst=False)

            
    ################# creating colorbar plots
    
    DSTNS = np.arange(Ls//2+1)
    COUNTS = np.zeros((len(x_data),Ls//2+1), dtype=np.float16)
    PERCNS = np.zeros((len(x_data),Ls//2+1), dtype=np.float16)

    nexus_bonds = np.load(folder_path + f'NXBL_{Vs}_{Ls:02}.npy', allow_pickle=True)
    # print("- NEXUS bonds:", nexus_bonds)
    for indx, bonds in enumerate(nexus_bonds):
        # print(f"- - bond {indx} has {len(bonds)}: ")
        # print(bonds)
        
        dists, counts = np.unique(hf.basis_distance(bonds, Ls), return_counts=True)
        percents = (counts/np.sum(counts))*100
        # counts = (counts/np.sum(counts))*100
        # print("- - ", counts)
        # dicd = dict(zip(dists,percents))
        dicd = dict(zip(dists,counts))
        dicp = dict(zip(dists,percents))
        for n in dists:
            COUNTS[indx, n] = dicd[n]
            PERCNS[indx, n] = dicp[n]


    for ii in range(Ls//2+1):
        # ########### to remove the zero values from the plot
        f = PERCNS[:, ii] != 0 
        
        # bottom = np.sum(COUNTS[f, 0:ii], axis=1)
        # stackbars = ax[1].bar(x_data[f], COUNTS[f, ii], bottom=bottom, width=3.0, label= f"{DSTNS[ii]}")

        bottom = np.sum(PERCNS[f, 0:ii], axis=1)
        stackbars = ax[1].bar(x_data[f], PERCNS[:, ii][f], bottom=bottom, width=220.0, label= f"{DSTNS[ii]}")
        
        ax[1].bar_label(stackbars, labels=COUNTS[f, ii].astype(np.int64), label_type="center", fontsize=5, rotation=90)#, padding=3)

        
    ax[1].set(
            # title=f"percentage of distances in new method for L={Ls}, V={Vs} & |m|={0}", 
            yscale='linear', ylabel=r'$\% $ of distance in $|M|$', ylim=(0, 102),
            xticks=x_data, xticklabels=[str(int(x)) for x in x_data],
        )
    
    ax[1].legend(bbox_to_anchor=(1.015, 0.5), loc='center', borderaxespad=0., markerscale=0.8, fontsize=8, title='distance', title_fontsize=6)
    ax[1].yaxis.label.set(fontsize=8) #, position=(0, 0.9))


    for axs in ax:
        axs.grid(which='major', axis='both', linestyle=':', linewidth=0.5, alpha=0.7,zorder=-10)
        axs.tick_params(axis='y', direction='in', labelsize=7)
        axs.tick_params(axis='x', labelrotation=90)
    
    
    fig.suptitle(f"Distribution of distances for best |M| selected + accuracy comparison - L={Ls}, V={Vs}", fontsize='medium', y=0.9001)


    # fig.savefig(f"Superposition_run/plots/_A_new_method_(dist+compar)_V{Vs}_L{Ls}_J{job_number}.png", dpi=800, bbox_inches = 'tight')
    # fig.savefig(f"Superposition_run/plots/Distance_Distribution_new_method_(+compar)_V{Vs}_L{Ls}.png", dpi=800, bbox_inches = 'tight')
    # fig.savefig(f"Superposition_run/plots/Distance_Distribution_new_method_(+compar)_V{Vs}_L{Ls}.pdf", dpi=500, bbox_inches = 'tight')


    
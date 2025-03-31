import numpy as np # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import glob
import sys


## array_number = int(sys.argv[1])
job_number = int(sys.argv[1])


folder_path = 'parallel_data/'
# folder_path = 'parallel_data_2D/'
all_files =  sorted( glob.glob('*.npy', root_dir=folder_path) )


particle_labels, energy_labels = tuple(), tuple()
E_DATA, P_DATA, p_data = [], [], []
for nome in all_files:
    # if nome[-6:-4] != "00":
    data = np.load(folder_path + nome)
    # print("   - ",nome)
    # print("   - - ",data)
    
    E_DATA.append(data[0,0])
    p_data.append(data[0,1])
    # P_DATA.append(data[0,2])
    
    if nome[:4] not in energy_labels:
        energy_labels += (nome[:4],)
    if nome[:4] not in particle_labels:
        particle_labels += (nome[:4],)


print(np.shape(E_DATA))
print(energy_labels)
print(np.shape(p_data))
print(particle_labels)
print("- - - - - - - - -")


E_DATA = np.reshape(E_DATA, (2,40))
p_data = np.reshape(p_data, (2,40))

print(np.shape(E_DATA))
print(energy_labels) # print(E_DATA[0,:]) # print(E_DATA[1,:])
print(np.shape(p_data))
print(particle_labels)


# cores = tuple(f'C{indx}' for indx in range() )

if True: #################################################################################################################### Saving the data
    arcivo1 = open(f'data/E_DATA_L100_(HFvsPC)_(V:0,10,.25)_J{job_number}.npy', 'wb')
    np.save(arcivo1, E_DATA)
    arcivo1.close()
    arcivo2 = open(f'data/p_data_L100_(HFvsPC)_(V:0,10,.25)_J{job_number}.npy', 'wb')
    np.save(arcivo2, p_data)
    arcivo2.close()
    
    # arcivo3 = open(f'data/P_DATA_test_J{job_number}.npy', 'wb')
    # np.save(arcivo3, P_DATA)
    # arcivo3.close()

if False: ##################################################################################################################### Ploting the data
    # E_DATA = np.load('data/E_DATA_4x4_J1625103.npy')
    # p_data = np.load('data/p_data_4x4_J1625103.npy')
    # energy_labels = ["ED0","ED1","ED2","ED3","ED4","ED5","HF","PC"]

    # VS = np.arange(0.03,8,0.2).T
    VS = np.arange(0,10,0.25)

    # fig, ax = plt.subplots(1,2, figsize=(18, 4) ,subplot_kw=dict( 
    #     xscale = 'linear', yscale ='linear', # xscale='log', yscale ='linear',               
    #     # ylabel = r'$A(t)$', 
    #     xlabel = r'$V$', #'$\log(\frac{t}{1})$',
    #     xticks = np.arange(0,22,2),
    #     # ylim=(1.e-18, 1.85), xlim = (1.e-12,100),            
    #     ))

    # fig.suptitle("HF, PC, ED for Large V and 4x4 lattice", y=1.01)

    # # for n in range(4):
    # #     E_DATA[n,:20] = E_DATA[n,:20] - E_DATA[0,:20]
    
    # ax[0].plot(VS, np.transpose(p_data), label = ( "ED", "HF","PC1", "PC2" ) )#particle_labels)#, color = cores)
    # ax[0].set_ylabel(r'$|n_1 - n_2|$')
    # ax[1].plot(VS, np.transpose(E_DATA), label = ( "ED", "HF","PC1", "PC2" ) )#energy_labels)#, color = cores)
    # ax[1].set_ylabel(r'$E$')
    # # ax[2].plot(VS, np.transpose(P_DATA), label = particle_labels)#, color = cores)

    # # ax[0].set_title('Target Energy')
    # # ax[1].set_title(r'$n_1=\left<c_1^\dagger c_1 \right>$')
    # # ax[2].set_title(r'$N = \sum_i \: n_i$')

    # ax[0].legend(loc='upper left')
    # ax[1].legend(loc='lower left')
    # # ax[2].legend(loc='lower left')
    # fig.savefig(f"data/ED_HF_PC_large_V_comparisons_2D_J{job_number}.pdf", bbox_inches = 'tight')
    # fig.show()
    
    ###################################### automized version of above plotting code! ##############################
    
    fig, ax = plt.subplots(1,2, figsize=(16, 5),
                           subplot_kw=dict(
                               # xscale = 'linear', yscale ='linear',
                               # ylim=(1.e+2, 1.e-12), # xlim = (0,21),# ylabel = r'$\Delta E = E - E_{ 0 }$',
                               xlabel = r'$V$', xticks = np.arange(0,11),
                               )
                           )
    
    # fig.suptitle("Comparison of optimization methods with respect to the exact diagonalization results for a 1D system with length 16.", y=0.95, fontsize=11) #for frank plot
    fig.suptitle(" ED & MF & HF & PC vs V for L=16 in 1D (log) (ned)", y=.95)

    # ax[0].plot(VS, np.transpose(p_data), label = ("HF","PC") )#particle_labels)#, color = cores)
    # ax[1].plot(VS, np.transpose(E_DATA), label = ("HF","PC") )#energy_labels)#, color = cores)
    
    # cores = colors.colormaps['tab10']
    # marks = ['None','None','None','1','2','3','4','|','o','s']
    # lines = ['-','--',':','None','None','None','None','None','None','None']
    # marks = ['None','None','1','2','3','4','o','*','^','v','<']
    # lines = ['-','--',':',':',':',':','None','None','None','None','None'] #for normal plot
    # widths = [1.7,1.7,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6] #this is for the case where markers were joined
    
    marks = ['None','None','None','2','3','4','o','*','^','v','<'] #for frank plot
    lines = ['-','--','-.',':',':',':',':',':',':',':',':'] #for frank plot
    widths = [1.7,1.7,1.7,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6,.6] #for frank plot
    # energy_labels = [r"$\text{GS}_\text{ED}$",r"${1^{st}}_\text{ED}$",r"${2^{nd}}_\text{ED}$",r"${3^{rd}}_\text{ED}$",r"${4^{th}}_\text{ED}$",r"${5^{th}}_\text{ED}$","H.F.","P.C.F.F",r"$\text{fVQE}_\text{1 layer}$",r"$\text{fVQE}_\text{2 layer}$",r"$\text{fVQE}_\text{4 layer}$"]
    for inx, pn in enumerate([0,1,2,6,7,8]):
        # particle_data = []
        # if pn < 6:
        #     var=0.5
        # else:
        #     var=0.0
        # particle_data.append( p_data[pn,:] - var ) 
        energy_data = []
        energy_data.append( E_DATA[pn,:] - E_DATA[0,:] ) 
        # energy_data.append( [abs( (E_DATA[pn,nn] - E_DATA[0,nn])/E_DATA[0,nn]) for nn in range(40)] ) #for frank plot
        # energy_data.append( E_DATA[pn,:]) 
        ax[1].plot(VS, np.transpose(energy_data), label = energy_labels[pn], color = f'C{inx}', ls=lines[pn], lw=widths[pn], marker=marks[pn], mfc='None')
        ax[0].plot(VS, np.transpose(energy_data), label = energy_labels[pn], color = f'C{inx}', ls=lines[pn], lw=widths[pn], marker=marks[pn], mfc='None')
        # ax[0].plot(VS, np.transpose(p_data[pn,:]), label = energy_labels[pn], color = f'C{inx}', ls=lines[pn], lw=widths[pn], marker=marks[pn], mfc='None')
    
    
            
    ax[0].set_yscale('linear')
    # ax[0].set_ylabel(r'$|n_1 - n_2|$')
    ax[0].set_ylabel(r'$\Delta E = \langle H \rangle - E_\text{GS}$')
    
    # ax[1].set_yscale('linear')
    # ax[1].set_ylabel('Method Energy') 
    ax[1].set_yscale('log')
    ax[1].set_ylim(1.e-4,2.e+1)
    ax[1].grid(visible=True, which='major',axis='y')
    # ax[1].set_ylabel(r'$\Delta E = \frac{\langle H \rangle - E_\text{GS} }{E_\text{GS}}$')
    ax[1].set_ylabel(r'$\Delta E = \langle H \rangle - E_\text{GS}$')
        

    # ax[0].legend(loc='upper left')#, fontsize='large') #(loc='center right', ncols=2, markerscale=1.4)
    ax[1].legend(loc='best', ncols=2, markerscale=1.1)#, fontsize='large') #(loc='center left', ncols=2, markerscale=1.4)

    fig.savefig(f"data/ED_HF_MF_PC_changing_V_1D_Ned_(log)_J{job_number}.pdf", bbox_inches = 'tight')

    fig.show()
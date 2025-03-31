import numpy as np # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import glob
import sys


## array_number = int(sys.argv[1])
job_number = int(sys.argv[1])


folder_path = 'data/'

######### read the data(s)
p_data_1d = np.load(folder_path+'p_data_L12_J1725757.npy')
e_data_1d = np.load(folder_path+'E_DATA_L12_J1725757.npy')
# p_data_2d = np.load(folder_path+'p_data_10x10_J1609893.npy')
# e_data_2d = np.load(folder_path+'E_DATA_10x10_J1609893.npy')
# p_data_2d0 = np.load(folder_path+'p_data_10x10_J1594024.npy')
# e_data_2d0 = np.load(folder_path+'E_DATA_10x10_J1594024.npy')
# print(np.shape(p_data_1d))
# print(np.shape(e_data_1d))
# print(np.shape(p_data_2d))
# print(np.shape(e_data_2d))
print("")
print(e_data_1d)
print("")
########## Ploting the data
    
VS = np.arange(0,10,0.25)
# VS2 = np.array([0,1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,18,19])*0.5

fig, ax = plt.subplots(1,1, figsize=(8, 5),  constrained_layout=True,
        subplot_kw=dict( 
        xscale = 'linear', yscale ='linear', # xscale='log', yscale ='linear',               
        # ylabel = r'$A(t)$', 
        xlabel = r'$V$', #'$\log(\frac{t}{1})$',
        xticks = np.arange(0,11), 
        # ylim=(1.e-18, 1.85), xlim = (1.e-12,100),            
        ),
        # tick_params=dict(direction='in', which='major' )    
    )

fig.suptitle("PC vs HF comparison for large sysytem", y=1.1)

e_data = [e_data_1d[0,nn] - e_data_1d[1,nn] for nn in range(40)]
new_e_data = np.delete(e_data, 28)
new_VS = np.delete(VS, 28)
ax.plot(new_VS, np.transpose(new_e_data))#energy_labels)#, color = cores)
# ax[0,0].plot(VS, np.transpose(p_data_1d), label = ("HF", "PC") )#energy_labels)#, color = cores)
# ax[0,1].plot(VS, np.transpose(e_data_1d), label = ("HF", "PC") )#energy_labels)#, color = cores)
# ax[1,0].plot(VS2, np.transpose(p_data_2d), label = ("HF", "PC") )#energy_labels)#, color = cores)
# ax[1,1].plot(VS2, np.transpose(e_data_2d), label = ("HF", "PC") )#energy_labels)#, color = cores)
# ax[1,0].plot(VS, np.transpose(p_data_2d0), ls='dotted' ) # label = ("HF", "PC"))#, color = cores)
# ax[1,1].plot(VS, np.transpose(e_data_2d0), ls='dotted' ) #label = ("HF", "PC"))#, color = cores)
# ax[1].plot(VS, np.transpose(p_data), label = ("ED", "PC") )#particle_labels)#, color = cores)



# ax[0,0].set_title('      1D: L=120')
# ax[1,0].set_title('      2D: Lx,Ly=10,10')
# ax[1].set_title(r'$n_1=\left<c_1^\dagger c_1 \right>$')

ax.set_ylabel(r'$E_{HF}-E_{PC}$')
ax.set_ylim(-7.e-3,2.e-4)

# for n in range(2):
#     ax[n,0].set_ylabel(r'$|n_1-n_2|$')
#     ax[n,0].legend(loc='upper left')
#     ax[n,1].set_ylabel(r'$energy$')
#     ax[n,1].legend(loc='lower left')
#     ax[n,0].tick_params(direction='in', which='major')
#     ax[n,1].tick_params(direction='in', which='major')

# ax[1,0].legend(loc='upper left')
# ax[0].legend(loc='lower left')
# plt.tick_params(direction='in',which='both')
fig.savefig(f"data/HF_vs_PC_L20_J{job_number}.pdf", bbox_inches = 'tight')
fig.show()


###########################################-- just plotting energy and particle count
# st_energy_data = []
# nd_energy_data = []
# pc_energy_data = []
# hf_energy_data = []
# # for dd in range(3):
# #     pc_energy_data.append( E_DATA[3,:] - E_DATA[dd,:] ) 
# #     hf_energy_data.append( E_DATA[4,:] - E_DATA[dd,:] ) 
# st_energy_data.append( E_DATA[1,:] - E_DATA[0,:] ) 
# nd_energy_data.append( E_DATA[2,:] - E_DATA[0,:] ) 
# pc_energy_data.append( E_DATA[3,:] - E_DATA[0,:] ) 
# hf_energy_data.append( E_DATA[4,:] - E_DATA[0,:] ) 
    
# VS = np.arange(0,8,0.2).T


# fig, ax = plt.subplots(1,1, figsize=(11, 6) ,subplot_kw=dict( 
#     xscale = 'linear', yscale ='linear', # xscale='log', yscale ='linear',               
#     ylabel = r'$\Delta E = E - E_{ 0 }$', 
#     xlabel = r'$V$', #'$\log(\frac{t}{1})$',
#     # ylim=(1.e-18, 1.85), xlim = (1.e-12,100),            
#     ))

# ax.plot(VS, np.transpose(st_energy_data), label = r'${1}^{st}$ excited' )#, color = [['red'], ['green'], ['blue']] )
# ax.plot(VS, np.transpose(nd_energy_data), label = r'${2}^{nd}$ excited' )#, color = [['red'], ['green'], ['blue']] )
# ax.plot(VS, np.transpose(pc_energy_data), label = r'PC' )#, color = [['red'], ['green'], ['blue']] )
# ax.plot(VS, np.transpose(hf_energy_data), label = r'HF')#, color = 'rgb' )

# ax.set_title('relative energies for different method \n system size = 16')

# ax.legend(loc='upper left')

# fig.savefig(f"data/relative_energies_method_comparisons_J{job_number}.pdf", bbox_inches = 'tight')
# fig.show()

###################################### automized version of above plotting code! ##############################

# fig, ax = plt.subplots(1,1, figsize=(13, 6) ,subplot_kw=dict( 
#     xscale = 'linear', yscale ='linear',                
#     # ylabel = r'$\Delta E = E - E_{ 0 }$', 
#     xlabel = r'$V$', ))

# fig.suptitle('relative energy \n system size = 4x4', y=1.01)

# ax.set_title(r'$\Delta E = E - E_{ 0 }$')
# # ax[1].set_title(r'$n_1=\left<c_1^\dagger c_1 \right>$')

# VS = np.arange(0.023,8,0.2).T
# # cores = colors.colormaps['tab10']
# #plots = [2,3,4,5,6,7,8,9,11,12] #range(2,10)
# for inx, pn in enumerate([1,2,3,4,5,9,10]):
#     energy_data = []
#     energy_data.append( E_DATA[pn,:] - E_DATA[0,:] ) 
#     ax.plot(VS, np.transpose(energy_data), label = energy_labels[pn], color = f'C{inx}')#, color = [['red'], ['green'], ['blue']] )
# gs_data = [0.0 for _ in range(40)]            
# ax.plot(VS, np.transpose(gs_data), label = r'E_0', color = 'black', ls = '-.')#, color = [['red'], ['green'], ['blue']] )
# # for inx in range(9):
# #     partic_data = []
# #     partic_data.append( p_data[inx,:]) #.append( p_data[pn,:] - p_data[0,:] ) 
# #     ax[1].plot(VS, np.transpose(partic_data), label = particle_labels[inx], color = f'C{inx}')#, color = [['red'], ['green'], ['blue']] )
        

# ax.legend(loc='upper center', ncols=2)
# # ax[1].legend(loc='upper left', ncols=2)

# fig.savefig(f"data/2D_relative_energy_density_comparison_20241029_J{job_number}.pdf", bbox_inches = 'tight')
# fig.show()
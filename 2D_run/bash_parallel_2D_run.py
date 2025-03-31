import sys
import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as pl # type: ignore

from free_fermion_function import *
from general_quadratic_function import *
from gaussian_state_function import *
from exact_diagonalization_function import *
from mean_field_function import *
from HF_function import *
from circuit_vqe_function import *


array_number = int(sys.argv[1])
job_number = int(sys.argv[2])

print(f" ***** job {array_number:02} started ***** ")
# print(f" ***** Large sys small V ***** ")

Lx, Ly = 4, 4
Vs = np.arange(-5,5,0.5)
maxsteps = 500

#################################################################################################### NEW WAY with less files to save ######################################################################################################
#################################################################################################################### ED
# t_i = tt.time()
# particle_ED, energy_ED, = particle_count_2D(Lx,Ly, Vs[array_number], 1.0, K=6)
# particle_dif = abs( particle_ED[0,0] - particle_ED[0,1])
# print(" - ED Run Time: ", tt.time() - t_i,"(s)")
# print(" - - ED energy (",len(energy_ED),"): ", energy_ED[:])
# print(" - - ED particle ",np.shape(particle_ED), " , ", particle_ED[0,:4])

# # ED_DATA0 = []
# # # ED_DATA0.append([energy_ED[0], particle_ED[0,0], sum(particle_ED[0])])
# # ED_DATA0.append([energy_ED[0], particle_dif])
# # arcivo1 = open(f'parallel_data_2D/ED_{array_number:02}.npy', 'wb')
# # np.save(arcivo1, ED_DATA0)
# # arcivo1.close()

# uniq_energi, uniq_indx = np.unique( np.round(energy_ED, decimals=7), return_index=True)
# print(" - - unique energy (",len(uniq_energi),")", uniq_energi)
# # for num, indx in enumerate(uniq_indx[:8]):
# for num, indx in enumerate(range(6)):
#     ED_DATA = []
#     particle_diff = abs( particle_ED[indx,0] - particle_ED[indx,1])
#     # ED_DATA.append([energy_ED[indx], particle_ED[indx,0], sum(particle_ED[indx])])
#     ED_DATA.append([energy_ED[indx], particle_diff])
#     arcivo = open(f'parallel_data_2D/ED{num:02}__{array_number:02}.npy', 'wb')
#     np.save(arcivo, ED_DATA)
#     arcivo.close()
#     # print(" - - - the uniqe energy:", energy_ED[indx])
# print('\n')
#################################################################################################################### PC
# t_i = tt.time()
# energy_PC, particle_PC = freefermion_optimization_2d(Vs[array_number], Lx, Ly, energy_func=Sym, max_step=maxsteps)
# print(" - PC1 Run Time: ", tt.time() - t_i,"(s)")
# print(" - - PC1 energy:", energy_PC, ", |n_1-n_2|:", particle_PC)

# PC_DATA = []
# # PC_DATA.append([energy_PC, particle_PC[0], sum(particle_PC)])
# PC_DATA.append([energy_PC, particle_PC])
# arcivo1 = open(f'parallel_data_2D/PC1__{array_number:02}.npy', 'wb')
# np.save(arcivo1, PC_DATA)
# arcivo1.close()
# print('\n')

# # t_i = tt.time()
# # energy_PC2, particle_PC2 = freefermion_optimization_2d(-1.0*Vs[array_number], Lx, Ly, energy_func=Sym_ZeroDiag, max_step=maxsteps)
# # print(" - PC2 Run Time: ", tt.time() - t_i,"(s)")
# # print(" - - PC2 energy ", energy_PC2)

# # PC_DATA2 = []
# # PC_DATA2.append([energy_PC2, particle_PC2])
# # arcivo12 = open(f'parallel_data_2D/PC2__{array_number:02}.npy', 'wb')
# # np.save(arcivo12, PC_DATA2)
# # arcivo12.close()
# # print('\n')
#################################################################################################################### GS
# t_i = tt.time()
# energy_GS, particle_GS = genstate_optimization_2d(Vs[array_number], Lx, Ly, max_step=maxsteps)
# print(" - GS Run Time: ", tt.time() - t_i,"(s)")
# print(" - - GS energy ", energy_GS)

# GS_DATA = []
# # PC_DATA.append([energy_PC, particle_PC[0], sum(particle_PC)])
# GS_DATA.append([energy_GS, particle_GS])
# arcivo2 = open(f'parallel_data_2D/GS__{array_number:02}.npy', 'wb')
# np.save(arcivo2, GS_DATA)
# arcivo2.close()
# print('\n')
#################################################################################################################### MF
# t_i = tt.time()
# if Lx % 2 == 0:
#     initial_n = [1.,0.]*(Lx//2)
#     for c_y in range(1,Ly):
#         initial_n += list(np.roll([1.,0]*(Lx//2), c_y))
# else:
#     initial_n = [1.,0.]*((Lx*Ly)//2)
# initial_n += np.random.randn(Lx*Ly)

# density, C = mean_field_2d(Vs[array_number],initial_n,(Lx,Ly), max_iter=maxsteps)
# energy_MF = fullenergy_MF_2d(C, Vs[array_number], (Lx,Ly))
# # particle_HF = density[0]
# particle_MF = abs( density[0] - density[1] )
# print(" - HF Run Time: ", tt.time() - t_i,"(s)")
# print(" - - HF energy:", energy_HF, ", |n_1-n_2|:", particle_HF)
# print(" - - HF particle (",len(density),")", density)

# HF_DATA = []
# HF_DATA.append([energy_HF, particle_HF])#, sum(density)])
# arcivo1 = open(f'parallel_data_2D/HF__{array_number:02}.npy', 'wb')
# np.save(arcivo1, HF_DATA)
# arcivo1.close()

print("test of other MF function")

t_i = tt.time()
EE, pp = mean_field_optimization_2d((Vs[array_number],1.0), (Lx,Ly), max_iter= maxsteps)
print("- - - - - - - - - - - - - - - - - - HF Run Time: ", tt.time() - t_i,"(s)")
print(" - - MF energy:", EE, ", |n_1-n_2|:", pp)

print('\n')
#################################################################################################################### VQE
# t_i = tt.time()

# vqe_maxsteps = 500

# Layers = 1
# energy_VQE1, densiti_VQE1 = fermion_VQE_2d((Vs[array_number], 1.0), (Lx,Ly,Layers ), max_iter = vqe_maxsteps, grad_rate=10/(2*4*Layers*Lx*Ly) )
# print(" - VQE 1 Run Time: ", tt.time() - t_i,"(s)")
# print("energy data (", energy_VQE1.size,"): ",energy_VQE1[-1])
# particle_VQE1 = abs( densiti_VQE1[0] - densiti_VQE1[1])
# print("particle data (", densiti_VQE1.size,"): ", particle_VQE1)
# # VQE_DATA = []
# # VQE_DATA.append([energy_HF, particle_HF[0], sum(particle_HF)])
# # VQE_DATA.append([ energy_VQE[-1], particle_VQE ])
# arcivo41 = open(f'parallel_data_2D/VQE1__{array_number:02}.npy', 'wb')
# np.save(arcivo41, [[ np.min(energy_VQE1), particle_VQE1 ]]) # np.save(arcivo4, VQE_DATA)
# arcivo41.close()

# Layers = 2
# energy_VQE2, densiti_VQE2 = fermion_VQE_2d((Vs[array_number], 1.0), (Lx,Ly,Layers), max_iter = vqe_maxsteps, grad_rate=10/(2*4*Layers*Lx*Ly) )
# print(" - VQE 2 Run Time: ", tt.time() - t_i,"(s)")
# print("energy data (", energy_VQE2.size,"): ",energy_VQE2[-1])
# particle_VQE2 = abs( densiti_VQE2[0] - densiti_VQE2[1])
# print("particle data (", densiti_VQE2.size,"): ", particle_VQE2)
# arcivo42 = open(f'parallel_data_2D/VQE2__{array_number:02}.npy', 'wb')
# np.save(arcivo42, [[ np.min(energy_VQE2), particle_VQE2 ]]) # np.save(arcivo4, VQE_DATA)
# arcivo42.close()


# # energy_VQE4, densiti_VQE4 = fermion_VQE_2d((Vs[array_number], 1.0), (Lx,Ly,3), max_iter = vqe_maxsteps, grad_rate=20/(2*4*Layers*Lx*Ly) )
# # print(" - VQE 4 Run Time: ", tt.time() - t_i,"(s)")
# # print("energy data (", energy_VQE4.size,"): ",energy_VQE4[-1])
# # particle_VQE4 = abs( densiti_VQE4[0] - densiti_VQE4[1])
# # print("particle data (", densiti_VQE4.size,"): ", particle_VQE4)
# # arcivo44 = open(f'parallel_data_2D/VQE4__{array_number:02}.npy', 'wb')
# # np.save(arcivo44, [[ np.min(energy_VQE4), particle_VQE4 ]]) # np.save(arcivo4, VQE_DATA)
# # arcivo44.close()



# if True:
#     Ns = np.arange(vqe_maxsteps)
#     pl.plot(Ns, energy_VQE1,".",label="1")
#     pl.plot(Ns, energy_VQE2,".",label="2")
#     # pl.plot(Ns, energy_VQE4,".",label="3")
#     pl.ylabel("VQE energy")
#     pl.xlabel("$n_{steps}$")
#     pl.title(f'2D, Lx,Ly=4 and V={Vs[array_number]}')
#     pl.legend()
#     pl.savefig(f"parallel_data_2D/VQE_energy_conv_V={Vs[array_number] :.02}_{array_number:02}.pdf", bbox_inches = 'tight')
#     pl.show()





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

L = 100
Vs = np.arange(0,10,0.25)
maxsteps = 500

#################################################################################################### NEW WAY with less files to save ######################################################################################################
# #################################################################################################################### ED
# t_i = tt.time()
# particle_ED, energy_ED, = particle_count(L, Vs[array_number], K=6)
# # particle_dif = abs( particle_ED[0,0] - particle_ED[0,1])
# print("- - - - - - - - - - - - - - - - - - - ED Run Time: ", tt.time() - t_i,"(s)")
# print("ED energy (",len(energy_ED),")", energy_ED)
# print("ED particle:", len(particle_ED),",", particle_ED)
# # print(" - - ED total particle:", sum(particle_ED[0]),",", sum(particle_ED[1]),",", sum(particle_ED[2]),",", sum(particle_ED[3]),",", sum(particle_ED[4]), sum(particle_ED[5]),",", sum(particle_ED[6]),",", sum(particle_ED[7]) ) #,",", sum(particle_ED[8]),",", sum(particle_ED[9]))

# uniq_energi, uniq_indx = np.unique( np.round(energy_ED, decimals=8), return_index=True)
# print(" - - unique energy (",len(uniq_energi),")", uniq_energi)
# # for num, indx in enumerate(uniq_indx):
# for num, indx in enumerate(range(6)):
#     ED_DATA = []
#     particle_diff = abs( particle_ED[indx,0] - particle_ED[indx,1])
#     # ED_DATA.append([energy_ED[indx], particle_ED[indx,0], sum(particle_ED[indx])])
#     ED_DATA.append([energy_ED[indx], particle_diff, sum(particle_ED[indx])])
#     arcivo = open(f'parallel_data/ED{num:02}__{array_number:02}.npy', 'wb')
#     np.save(arcivo, ED_DATA)
#     arcivo.close()
#     # print(" - - - the uniqe energy:", energy_ED[indx])

# ED_DATA0 = []
# ED_DATA0.append([energy_ED[0], particle_ED[0,0], sum(particle_ED[0])])
# arcivo1 = open(f'parallel_data/ED0__{array_number:02}.npy', 'wb')
# np.save(arcivo1, ED_DATA0)
# arcivo1.close()
# print('\n')
# #################################################################################################################### PC
t_i = tt.time()
energy_PC, particle_PC = freefermion_optimization(Vs[array_number], L, max_step=maxsteps)
print("- - - - - - - - - - - - - - - - - PC Run Time: ", tt.time() - t_i,"(s)")
print("PC energy:", energy_PC, " - - PC particle:", particle_PC)

PC_DATA = []
# PC_DATA.append([energy_PC, particle_PC, sum(particle_PC)])
PC_DATA.append([energy_PC, particle_PC])

arcivo1 = open(f'parallel_data/PC__{array_number:02}.npy', 'wb')
np.save(arcivo1, PC_DATA)
arcivo1.close()
print('\n')
# #################################################################################################################### GQ
# t_i = tt.time()
# energy_GQ, particle_GQ = genfermion_optimization(-1.0*Vs[array_number], L, max_step=maxsteps)
# print(" - GQ Run Time: ", tt.time() - t_i,"(s)")

# GQ_DATA = []
# GQ_DATA.append([energy_GQ, particle_GQ[0], sum(particle_GQ)])

# arcivo2 = open(f'parallel_data/GQ__{array_number:02}.npy', 'wb')
# np.save(arcivo2, GQ_DATA)
# arcivo2.close()
# print('\n')
# #################################################################################################################### GS
# t_i = tt.time()
# energy_GS, particle_GS = genstate_optimization(-1.0*Vs[array_number], L, max_step=maxsteps)
# print(" - GS Run Time: ", tt.time() - t_i,"(s)")

# GS_DATA = []
# GS_DATA.append([energy_GS, particle_GS[0], sum(particle_GS)])

# arcivo3 = open(f'parallel_data/GS__{array_number:02}.npy', 'wb')
# np.save(arcivo3, GS_DATA)
# arcivo3.close()
# print('\n')

# #################################################################################################################### MF
# t_i = tt.time()

# energy_MF, densitis_MF = mean_field_optimization( (Vs[array_number],1.0), L, max_iters = maxsteps)
# print("- - - - - - - - - - - - - - - - - - MF Run Time: ", tt.time() - t_i,"(s)")
# print("energy data (", energy_MF.size,"): ",energy_MF)
# print("particle data (", densitis_MF.size,"): ",densitis_MF)

# particle_MF = abs( densitis_MF[0] - densitis_MF[1])
# # print("particle data (", densitis.size,"): ",particle_MF)

# MF_DATA = []
# # HF_DATA.append([energy_HF, particle_HF[0], sum(particle_HF)])
# MF_DATA.append([ energy_MF, particle_MF ])

# arcivo3 = open(f'parallel_data/MF__{array_number:02}.npy', 'wb')
# np.save(arcivo3, MF_DATA)
# arcivo3.close()

# print('\n')
# #################################################################################################################### HF
t_i = tt.time()

energy_HF, densitis_HF = hart_fock_optimization( (Vs[array_number], 1.0), L, max_iters = maxsteps)
print("- - - - - - - - - - - - - - - - - - - HF Run Time: ", tt.time() - t_i,"(s)")
print("energy data (", energy_HF.size,"): ",energy_HF)
print("particle data (", densitis_HF.size,"): ",densitis_HF)

particle_HF = abs( densitis_HF[0] - densitis_HF[1])

HF_DATA = []
# HF_DATA.append([energy_HF, particle_HF[0], sum(particle_HF)])
HF_DATA.append([ energy_HF, particle_HF ])

arcivo4 = open(f'parallel_data/HF__{array_number:02}.npy', 'wb')
np.save(arcivo4, HF_DATA)
arcivo4.close()

print('\n')
# #################################################################################################################### VQE
# t_i = tt.time()

# vqe_maxsteps = 5000

# Layers = 1
# energy_VQE1, densiti_VQE1 = fermion_VQE_1d((Vs[array_number], 1.0), (Layers, L), max_iter = vqe_maxsteps, grad_rate=10/(4*Layers*L) )
# print(f" - VQE Run Time ({Layers}): ", tt.time() - t_i,"(s)")
# print("energy data (", energy_VQE1.size,"): ",energy_VQE1[-1])
# particle_VQE1 = abs( densiti_VQE1[0] - densiti_VQE1[1])
# print("particle data (", densiti_VQE1.size,"): ", particle_VQE1)
# # VQE_DATA = []
# # VQE_DATA.append([energy_HF, particle_HF[0], sum(particle_HF)])
# # VQE_DATA.append([ energy_VQE[-1], particle_VQE ])
# arcivo41 = open(f'parallel_data/VQE1__{array_number:02}.npy', 'wb')
# np.save(arcivo41, [[ np.min(energy_VQE1), particle_VQE1 ]]) # np.save(arcivo4, VQE_DATA)
# arcivo41.close()


# Layers = 2
# energy_VQE2, densiti_VQE2 = fermion_VQE_1d((Vs[array_number], 1.0), (Layers, L), max_iter = vqe_maxsteps, grad_rate=10/(4*Layers*L) )
# print(f" - VQE Run Time ({Layers}): ", tt.time() - t_i,"(s)")
# print("energy data (", energy_VQE2.size,"): ",energy_VQE2[-1])
# particle_VQE2 = abs( densiti_VQE2[0] - densiti_VQE2[1])
# print("particle data (", densiti_VQE2.size,"): ", particle_VQE2)
# arcivo42 = open(f'parallel_data/VQE2__{array_number:02}.npy', 'wb')
# np.save(arcivo42, [[ np.min(energy_VQE2), particle_VQE2 ]]) # np.save(arcivo4, VQE_DATA)
# arcivo42.close()

# Layers = 4
# energy_VQE4, densiti_VQE4 = fermion_VQE_1d( (Vs[array_number], 1.0), (Layers, L), max_iter = vqe_maxsteps, grad_rate=10/(4*Layers*L) )
# print(f" - VQE Run Time ({Layers}): ", tt.time() - t_i,"(s)")
# print("energy data (", energy_VQE4.size,"): ",energy_VQE4[-1])
# particle_VQE4 = abs( densiti_VQE4[0] - densiti_VQE4[1])
# print("particle data (", densiti_VQE4.size,"): ", particle_VQE4)
# arcivo44 = open(f'parallel_data/VQE4__{array_number:02}.npy', 'wb')
# np.save(arcivo44, [[ np.min(energy_VQE4), particle_VQE4 ]]) # np.save(arcivo4, VQE_DATA)
# arcivo44.close()


# if True:
#     Ns = np.arange(vqe_maxsteps)
#     pl.plot(Ns, energy_VQE1,".",label="1")
#     pl.plot(Ns, energy_VQE2,".",label="2")
#     pl.plot(Ns, energy_VQE4,".",label="4")
#     pl.ylabel("VQE energy")
#     pl.xlabel("$n_{steps}$")
#     pl.title(f'1D, L=16 - V={Vs[array_number]}')
#     pl.legend()
#     pl.savefig(f"parallel_data/VQE_energy_conv_V={Vs[array_number] :.02}_{array_number:02}.pdf", bbox_inches = 'tight')
#     pl.show()


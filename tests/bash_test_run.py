import sys
import numpy as np
import scipy as sp
import time as tt
import matplotlib.pyplot as pl

from free_fermion_function import *
from general_quadratic_function import *
from gaussian_state_function import *
from exact_diagonalization_function import *
from mean_field_function import *


array_number = int(sys.argv[1])
job_number = int(sys.argv[2])

print(f" ***** job {array_number:02} started ***** ")

Lx, Ly = 4, 4
Vs = np.arange(0.023,8,0.2)
maxsteps = 600

#################################################################################################### NEW WAY with less files to save ######################################################################################################
#################################################################################################################### PC
t_i = tt.time()
energy_PC, particle_PC = freefermion_optimization_2d(Vs[array_number], Lx, Ly, max_step=maxsteps)
print(" - PC Run Time: ", tt.time() - t_i,"(s)")
print(" - - PC energy ", energy_PC)

PC_DATA = []
# PC_DATA.append([energy_PC, particle_PC[0], sum(particle_PC)])
PC_DATA.append([energy_PC, particle_PC])
arcivo1 = open(f'parallel_data/PC__{array_number:02}.npy', 'wb')
np.save(arcivo1, PC_DATA)
arcivo1.close()
print('\n')
#################################################################################################################### GS
# t_i = tt.time()
# energy_GS, particle_GS = genstate_optimization_2d(Vs[array_number], Lx, Ly, max_step=maxsteps)
# print(" - GS Run Time: ", tt.time() - t_i,"(s)")
# print(" - - GS energy ", energy_GS)

# GS_DATA = []
# # PC_DATA.append([energy_PC, particle_PC[0], sum(particle_PC)])
# GS_DATA.append([energy_GS, particle_GS])
# arcivo2 = open(f'parallel_data/GS__{array_number:02}.npy', 'wb')
# np.save(arcivo2, GS_DATA)
# arcivo2.close()
# print('\n')
#################################################################################################################### ED
t_i = tt.time()
particle_ED, energy_ED, = particle_count_2D(Lx,Ly, Vs[array_number], 1.0, K=34)
print(" - ED Run Time: ", tt.time() - t_i,"(s)")
# print(" - - ED energy (",len(energy_ED),")", energy_ED)
# print(" - - ED particle (",len(particle_ED),")", [particle_ED[ii,0] for ii in range(19)])

# ED_DATA0 = []
# ED_DATA0.append([energy_ED[0], particle_ED[0,0], sum(particle_ED[0])])
# arcivo1 = open(f'parallel_data/ED_{array_number:02}.npy', 'wb')
# np.save(arcivo1, ED_DATA0)
# arcivo1.close()

uniq_energi, uniq_indx = np.unique( np.round(energy_ED, decimals=7), return_index=True)
print(" - - unique energy (",len(uniq_energi),")", uniq_energi)
for num, indx in enumerate(uniq_indx[:8]):
    ED_DATA = []
    ED_DATA.append([energy_ED[indx], particle_ED[indx,0], sum(particle_ED[indx])])
    arcivo = open(f'parallel_data/ED{num:02}__{array_number:02}.npy', 'wb')
    np.save(arcivo, ED_DATA)
    arcivo.close()
    # print(" - - - the uniqe energy:", energy_ED[indx])
print('\n')
#################################################################################################################### HF
# t_i = tt.time()
# if Lx % 2 == 0:
#     initial_n = [1.,0.]*(Lx//2)
#     for c_y in range(1,Ly):
#         initial_n += list(np.roll([1.,0]*(Lx//2), c_y))
# else:
#     initial_n = [1.,0.]*((Lx*Ly)//2)
# initial_n += 0.3 * np.random.randn(Lx*Ly)

# density, C = mean_field_2d(Vs[array_number],initial_n,(Lx,Ly), max_iter=500)
# energy_HF = fullenergy_HF_2d(C, Vs[array_number], (Lx,Ly))
# particle_HF = density[0]
# print(" - HF Run Time: ", tt.time() - t_i,"(s)")
# # print(" - - HF energy", energy_HF)
# # print(" - - HF particle (",len(density),")", density)

# # HF_DATA = []
# # HF_DATA.append([energy_HF, particle_HF, sum(density)])
# # arcivo1 = open(f'parallel_data/HF_{array_number:02}.npy', 'wb')
# # np.save(arcivo1, HF_DATA)
# # arcivo1.close()
# print('\n')

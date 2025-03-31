import sys
import numpy as np
import time as tt
import matplotlib.pyplot as pl

from free_fermion_function import *
from general_quadratic_function import *
from gaussian_state_function import *
from exact_diagonalization_function import *
from mean_field_function import *


array_number = int(sys.argv[1])
job_number = int(sys.argv[2])

print(" ***** job started ***** ")

L = 20
Vs = np.arange(0,4,0.05)
maxsteps = 200

N = 40

if array_number == 1:

    # t_i = tt.time()
    # en_ff = []
    # ps_ff = []
    # for V in Vs: 
    #     energy_list, particle_counts = freefermion_optimization(V, L, max_step=maxsteps)
    #     en_ff.append(energy_list)
    #     ps_ff.append(particle_counts)
    # t_f = tt.time()
    # print("PC Run Time: ", t_f - t_i,"(s)")

    # arcivo1 = open(f'data/PC_energies_A{array_number}_J{job_number}.npy', 'wb')
    # np.save(arcivo1, en_ff)
    # arcivo1.close()
    # arcivo2 = open(f'data/PC_particles_A{array_number}_J{job_number}.npy', 'wb')
    # np.save(arcivo2, ps_ff)
    # arcivo2.close()
    
    ## avaraging particle counts
    print("PC Run Started")
    t_i = tt.time()
    
    p_ff = np.zeros(len(Vs))
    p_total_ff = np.zeros(len(Vs))
    for n in range(N):
        
        for indx, V in enumerate(Vs): 
            energy_list, particle_counts = freefermion_optimization(V, L, max_step=maxsteps)
            p_ff[indx] += particle_counts[0] / N
            p_total_ff[indx] += sum(particle_counts) / N
            
        if n % 10 == 0:
            print('-   ',n,'th loop is done')
            
    t_f = tt.time()
    
    print("PC Run Time: ", t_f - t_i,"(s)")


    arcivo1 = open(f'data/PC_energies_A{array_number}_J{job_number}.npy', 'wb')
    np.save(arcivo1, p_ff)
    arcivo1.close()
    arcivo2 = open(f'data/PC_particles_A{array_number}_J{job_number}.npy', 'wb')
    np.save(arcivo2, p_total_ff)
    arcivo2.close()
    



if array_number == 2:
    
    # t_i = tt.time()
    # en_gq = []
    # ps_gq = []
    # for V in Vs: 
    #     energy, particle = genfermion_optimization(V, L, max_step=maxsteps)
    #     en_gq.append(energy)
    #     ps_gq.append(particle)
    # t_f = tt.time()
    # print("GQ Run Time: ", t_f - t_i,"(s)")

    ## avaraging particle counts
    print("GQ Run Started")
    t_i = tt.time()
    p_gq = np.zeros(len(Vs))
    p_total_gq = np.zeros(len(Vs))
    for n in range(N):
        
        for indx, V in enumerate(Vs): 
            energy_l, particle_c = genfermion_optimization(V, L, max_step=maxsteps)
            p_gq[indx] += particle_c[0] / N
            p_total_gq[indx] += sum(particle_c) / N
        
        if n % 10 == 0:
            print('-   ',n,'th loop is done')
            
    t_f = tt.time()
    
    print("GQ Run Time: ", t_f - t_i,"(s)")


    arcivo1 = open(f'data/GQ_energies_A{array_number}_J{job_number}.npy', 'wb')
    np.save(arcivo1, p_gq)
    arcivo1.close()
    arcivo2 = open(f'data/GQ_particles_A{array_number}_J{job_number}.npy', 'wb')
    np.save(arcivo2, p_total_gq)
    arcivo2.close()



if array_number == 3:

    # t_i = tt.time()
    # en_gs = []
    # ps_gs = []
    # for V in Vs: 
    #     energy, particle = genstate_optimization(V, L, max_step=maxsteps)
    #     en_gs.append(energy)
    #     ps_gs.append(particle)
    # t_f = tt.time()
    # print("GS Run Time: ", t_f - t_i,"(s)")

    ## avaraging particle counts
    print("GS Run Started")
    t_i = tt.time()
    p_gs = np.zeros(len(Vs))
    p_total_gs = np.zeros(len(Vs))
    for n in range(N):
        
        for indx, V in enumerate(Vs): 
            energy, particle = genstate_optimization(V, L, max_step=maxsteps)
            p_gs[indx] += particle[0] / N
            p_total_gs[indx] += sum(particle) / N
        if n % 10 == 0:
            print('-   ',n,'th loop is done')
    
    t_f = tt.time()
    
    print("GS Run Time: ", t_f - t_i,"(s)")


    arcivo1 = open(f'data/GS_energies_A{array_number}_J{job_number}.npy', 'wb')
    np.save(arcivo1, p_gs)
    arcivo1.close()
    arcivo2 = open(f'data/GS_particles_A{array_number}_J{job_number}.npy', 'wb')
    np.save(arcivo2, p_total_gs)
    arcivo2.close()



# if array_number == 4:

#     n = [1.,0.]*(L//2)
#     n += np.random.randn(L)

#     t_i = tt.time()
#     energy = []
#     for V in Vs:
#         density, C, energy_list = mf(V,n,L)
#         energy.append(fullEnergy_HF(C, V, L))
#     t_f = tt.time()
#     print("HF Run Time: ", t_f - t_i,"(s)")


#     arcivo1 = open(f'data/HF_data_A{array_number}_J{job_number}.npy', 'wb')
#     np.save(arcivo1, energy)
#     arcivo1.close()
#     # arcivo2 = open(f'data/Vs_data_A{array_number}_J{job_number}.npy', 'wb')
#     # np.save(arcivo2, Vs)
#     # arcivo2.close()

print(" ***** job ended ***** ")

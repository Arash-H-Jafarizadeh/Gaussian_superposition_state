import sys
import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as pl # type: ignore
import matplotlib.colors as colors

# from general_quadratic_function import *
# from gaussian_state_function import *
# from MF_function import *
# from circuit_vqe_function import *

sys.path.append('/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/source_code/')

# import free_fermion_function as ff
import exact_diagonalization_function as ed
import hartree_fock_function as hf



array_number = int(sys.argv[1])
job_number = int(sys.argv[2])

# print(f" ***** job {array_number:02} started ***** ")
print(f" ***** job {job_number} started ***** ")
print("")

threshod = 1.e-13
maxsteps = 400


#######################################################################################################################################################################################################################################
#################################################################################################### Parallel Runs Date 18022025 ######################################################################################################
#######################################################################################################################################################################################################################################
    

if True: #################################################################################################################### HF superposition full Hamiltonian for L's and V's
    sizes = [14] #[8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,30, 32]
    VS = [0.1, 0.2, 0.3, 0.4, 0.5] # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 
    
    input = []
    for siz in sizes:
        for vs in VS:
            input.append([siz, vs])
                
    ll = input[array_number][0]
    Vs = input[array_number][1]
    maxdim = sp.special.binom(ll, ll//2) #20000 #
    
    print(f"- Full Matrix for L={ll} and V={Vs} (s)")
    print("")
    
    print("- Actual max dim:", int(sp.special.binom(ll, ll//2)))
    print("")
    
    t_i = tt.time()
    # super_ham, super_basis = hf.hart_fock_superposition( [Vs, 1.0], ll, max_iters=maxsteps, PBC=False, basis_len = maxdim, start_point=1.e-4)
    super_ham, super_basis = hf.hart_fock_superposition( [Vs, 1.0], ll, max_iters=maxsteps, PBC=False, start_point=1.e-4)
    
    print(f"- Full Matrix Creation Run {array_number} Time: ", tt.time() - t_i,"(s)")
    print("")
    
    arcivo = open(f'Superposition_run/raw_data/Full_Matrix/FulMat__{Vs}_{ll:02}.npy', 'wb')
    np.save(arcivo, super_ham)
    arcivo.close()
    
    
    distances = hf.basis_distance(super_basis, ll)
    
    arcivo2 = open(f'Superposition_run/raw_data/Amplitudes/DSTN_{Vs}_{ll:02}.npy', 'wb')
    np.save(arcivo2, distances)
    arcivo2.close()
    
    print(f"- - data file {array_number} saved")
    print('')  
        
    
if False: #################################################################################################################### ED ground-state for L's and V's
    sizes = [18, 20, 22] #[8, 10, 12, 14, 16]
    VS = [0.1, 0.2, 0.3, 0.4] #[0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #
    
    input = []
    for siz in sizes:
        for vs in VS:
            input.append([siz, vs])
                
    ll = input[array_number][0]
    Vs = input[array_number][1]
    
    print(f"- Full Matrix for L={ll} and V={Vs} (s)")
    print("")
    
    
    t_i = tt.time()
    _, ED_GS = ed.particle_count(ll, [Vs, 1.0], K=2, PBC=False)

    
    print(f"- ED ground state Run {array_number} Time: ", tt.time() - t_i,"(s)")
    print("")
    
    print(f"- ground state energy for L={ll} and V={Vs} is:", ED_GS[0])
    print("")
    
    arcivo = open(f'Superposition_run/raw_data/EDGS_{Vs}_{ll:02}.npy', 'wb')
    np.save(arcivo, ED_GS[0])
    arcivo.close()
    print(f"- - data file {array_number} saved")
    print('')  
    
    
if False: #################################################################################################################### DMRG ground-state for L's and V's
    sizes = [20, 22, 24, 26] #[8, 10, 12, 14, 16]
    VS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #
    
    input = []
    for siz in sizes:
        for vs in VS:
            input.append([siz, vs])
                
    ll = input[array_number][0]
    Vs = input[array_number][1]
    
    print(f"DMRG ground-state for L={ll} and V={Vs}")
    print("")
    
    
    t_i = tt.time()
    
    DMRG_GS = 'none'
    
    print(f"- DMRG ground state Run {array_number} Time: ", tt.time() - t_i,"(s)")
    print("")
    
    print(f"- ground state energy for L={ll} and V={Vs} is:", DMRG_GS)
    print("")
        
    
        
    


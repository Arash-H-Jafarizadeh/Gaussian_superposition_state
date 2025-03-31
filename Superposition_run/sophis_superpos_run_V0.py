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

sys.path.append('/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/')

# import free_fermion_function as ff
import exact_diagonalization_function as ed
import HF_function as hf



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
    

if False: #################################################################################################################### HF superposition full Hamiltonian for L's and V's
    sizes = [8, 10, 12, 14, 16]
    VS = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    input = []
    for siz in sizes:
        for vs in VS:
            input.append([siz, vs])
                
    ll = input[array_number][0]
    Vs = input[array_number][1]
    
    print(f"- Full Matrix for L={ll} and V={Vs} (s)")
    print("")
    
    
    t_i = tt.time()
    supam_HF, full_order = hf.hart_fock_superposition( [Vs, 1.0], ll, max_iters=maxsteps, PBC=False,  start_point=1.e-4)
    
    print(f"- Full Matrix Creation Run {array_number} Time: ", tt.time() - t_i,"(s)")
    print("")
    
    # arcivo = open(f'Superposition/raw_data/FulMat__{array_number:02}.npy', 'wb')
    arcivo = open(f'Superposition/raw_data/FulMat__{Vs}_{ll}.npy', 'wb')
    np.save(arcivo, supam_HF)
    arcivo.close()
    print(f"- - data file {array_number} saved")
    print('')  
        
    
if True: #################################################################################################################### ED ground-state for L's and V's
    sizes = [8, 10, 12, 14, 16]
    VS = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    input = []
    for siz in sizes:
        for vs in VS:
            input.append([siz, vs])
                
    ll = input[array_number][0]
    Vs = input[array_number][1]
    
    print(f"- Full Matrix for L={ll} and V={Vs} (s)")
    print("")
    
    
    t_i = tt.time()
    supam_HF, full_order = hf.hart_fock_superposition( [Vs, 1.0], ll, max_iters=maxsteps, PBC=False,  start_point=1.e-4)
    
    print(f"- Full Matrix Creation Run {array_number} Time: ", tt.time() - t_i,"(s)")
    print("")
    
    # arcivo = open(f'Superposition/raw_data/FulMat__{array_number:02}.npy', 'wb')
    arcivo = open(f'Superposition/raw_data/FulMat__{Vs}_{ll}.npy', 'wb')
    np.save(arcivo, supam_HF)
    arcivo.close()
    print(f"- - data file {array_number} saved")
    print('')  
        
    


#######################################################################################################################################################################################################################################
#################################################################################################### Parallel Runs Date 03022025 ######################################################################################################
#######################################################################################################################################################################################################################################

if False:
    phys0 = 0.20, 1.0
    phys1 = 1.80, 1.0

    num_dims = 15
    input = []

    for L in [8, 10, 12, 14, 16, 18]:
        maxdim = sp.special.binom(L, L//2)
        truncation = np.linspace(1, maxdim, num_dims, dtype=np.int64)
        for trnc in truncation:
            input.append([L, trnc, trnc/maxdim ])
            
    # print(input)
    # print("")

    # #################################################################################################################### ED Part
    t_i = tt.time()

    ll = input[array_number][0]
    trc = input[array_number][1]
    scaled_size = input[array_number][2]
    
    print("the inputs are:")
    print(f"L={ll} and truncation= {trc} ----> {scaled_size}" )
    print("")
    
    _, GS_OBC_0 = ed.particle_count(ll, phys0, K=2, PBC=False)
    _, GS_OBC_1 = ed.particle_count(ll, phys1, K=2, PBC=False)

    print("ground state energies: V0=",GS_OBC_0[0]," , V1=", GS_OBC_1[0])
    print("")
    
    print(f"- - - - - - - - - - - - - - - - - - - ED Array {array_number} Run Time: ", tt.time() - t_i,"(s)")
    print("")

    # #################################################################################################################### HF superposition truncation

    GS0, GS1 = GS_OBC_0[0], GS_OBC_1[0]
    
    truncated_data = []
        
    t_i = tt.time()

    
    supam_HF0, _ = hf.hart_fock_superposition(phys0, ll, max_iters=maxsteps, PBC=False, basis_len = trc,  start_point=1.e-3)
    HF_E0, HF_U0 = np.linalg.eigh(supam_HF0)
    truncated_data.append(np.abs((GS0 - HF_E0[0]) / GS0))
    
    
    supam_HF1, _ = hf.hart_fock_superposition(phys1, ll, max_iters=maxsteps, PBC=False, basis_len = trc,  start_point=1.e-3)
    HF_E1, HF_U1 = np.linalg.eigh(supam_HF1)
    truncated_data.append(np.abs((GS1 - HF_E1[0]) / GS1))
    
    
    print(f"- - - - - - - - - - - - - - - - - - - Truncate size ({ll},{trc}) Run Time: ", tt.time() - t_i,"(s)")
    print("")

    print("trunc_enery: ", truncated_data)
    
    arcivo0 = open(f'Superposition/raw_data/TRN__{array_number:06}.npy', 'wb')
    np.save(arcivo0, truncated_data)
    arcivo0.close()
    
    print(f"Truncation Data file saved")
    print("")
    
    if scaled_size == 1.0: ################################################################################################ Amplitudes and Infidality
        infedal_data = []
        
        maxdim = sp.special.binom(ll, ll//2)
        truncation = np.linspace(1, maxdim, num_dims, dtype=np.int64)
        
        HF_fin0 = 1.0000000000001 - np.cumsum(HF_U0[:,0]**2)
        HF_inf0 = np.array(HF_fin0)[np.array(truncation,dtype=np.int64)-1]
        
        # arcivo1 = open(f'Superposition/raw_data/INFD_0_{ll:02}__{array_number:06}.npy', 'wb')
        # np.save(arcivo1, HF_inf0)
        # arcivo1.close()
        
        infedal_data.append(HF_inf0)
        
        HF_fin1 = 1.0000000000001 - np.cumsum(HF_U1[:,0]**2)
        HF_inf1 = np.array(HF_fin1)[np.array(truncation,dtype=np.int64)-1]
        
        # arcivo2 = open(f'Superposition/raw_data/INFD_1_{ll:02}_{array_number:06}.npy', 'wb')
        # np.save(arcivo2, HF_inf1)
        # arcivo2.close()
        
        infedal_data.append(HF_inf1)
        
        arcivo = open(f'Superposition/raw_data/INFD_{ll:02}_{array_number:06}.npy', 'wb')
        np.save(arcivo, infedal_data)
        arcivo.close()
        
        print(f"Infidality Data file saved")
        print('')

        amp_data = []

        amps0 = np.sort(np.abs(HF_U0[:,0]))[::-1]
        print("lenth of sorted", len(amps0))
        amps0 = amps0[ amps0 > threshod ]
        print("lenth of cutted and sorted", len(amps0))
        print("")
        amp_data.append(amps0)
        
        # arcivo0s = open(f'Superposition/raw_data/AMPS_0_{ll}_{array_number:06}.npy', 'wb')
        # np.save(arcivo0s, np.insert( amps0 , 0 , Vs[array_number] ))
        # arcivo0s.close()    

        amps1 = np.sort(np.abs(HF_U1[:,0]))[::-1]
        print("lenth of sorted", len(amps1))
        amps1 = amps1[ amps1 > threshod ]
        print("lenth of cutted and sorted", len(amps1))
        print("")        
        amp_data.append(amps1)
        
        # arcivo1s = open(f'Superposition/raw_data/AMPS_0_{ll}_{array_number:06}.npy', 'wb')
        # np.save(arcivo1s, np.insert( amps1 , 0 , Vs[array_number] ))
        # arcivo1s.close()    
        
        arcivo1s = open(f'Superposition/raw_data/AMPS_{ll:02}_{array_number:06}.npy', 'wb')
        np.save(arcivo1s, amp_data)
        arcivo1s.close()    

        print(f"Amplitudes Data file saved")
        print('')
    

# #################################################################################################################### HF superposition amplitudes + plot
if False:

    t_i = tt.time()

    Vs =  [0.2, 0.6, 1.0, 1.4, 1.8] 
    
    loop_t = tt.time()
    
    sup_hf, ord_hf = hf.hart_fock_superposition([Vs[array_number], 1.0], L, PBC=False, max_iters=maxsteps, start_point=1.e-5)
    _, vec_hf = np.linalg.eigh(sup_hf)
    
    sup_ff, ord_ff = ff.free_fermion_superposition([Vs[array_number], 1.0], L, PBC=False, max_iters=maxsteps)
    _, vec_ff = np.linalg.eigh(sup_ff)
        
    print(f"- - - - - - - - - - - - - - - - - - - Data point V={Vs[array_number]} Run Time: ", tt.time() - loop_t,"(s)")
    print("")


    arcivo1 = open(f'Superposition/raw_data/AMP__{array_number:06}.npy', 'wb')
    np.save(arcivo1, np.insert( np.abs(vec_hf[:,0]) , 0, Vs[array_number] ))
    # np.save(arcivo, np.abs(vec[:,0]))
    arcivo1.close()
    
    arcivo1s = open(f'Superposition/raw_data/AMPS__{array_number:06}.npy', 'wb')
    np.save(arcivo1s, np.insert( np.sort(np.abs(vec_hf[:,0]))[::-1] , 0 , Vs[array_number] ))
    arcivo1s.close()    

    arcivo1ss = open(f'Superposition/raw_data/AMPSS__{array_number:06}.npy', 'wb')
    np.save(arcivo1ss, np.insert( np.abs(vec_hf[:,0])[ord_hf.astype(int)] , 0 , Vs[array_number] ))
    arcivo1ss.close()    

    arcivo2 = open(f'Superposition/raw_data/FMP__{array_number:06}.npy', 'wb')
    np.save(arcivo2, np.insert( np.abs(vec_ff[:,0]) , 0, Vs[array_number] ))
    # np.save(arcivo, np.abs(vec[:,0]))
    arcivo2.close()
    
    arcivo2s = open(f'Superposition/raw_data/FMPS__{array_number:06}.npy', 'wb')
    np.save(arcivo2s, np.insert( np.sort(np.abs(vec_ff[:,0]))[::-1] , 0 , Vs[array_number] ))
    arcivo2s.close()
    
    arcivo2ss = open(f'Superposition/raw_data/FMPSS__{array_number:06}.npy', 'wb')
    np.save(arcivo2ss, np.insert( np.sort(np.abs(vec_ff[:,0]))[ord_ff.astype(int)] , 0 , Vs[array_number] ))
    arcivo2ss.close()
    
    print(f"Data file saved")
    print('\n')    
    
    

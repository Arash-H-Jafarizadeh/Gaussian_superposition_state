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

import free_fermion_function as ff
import exact_diagonalization_function as ed
import HF_function as hf



array_number = int(sys.argv[1])
job_number = int(sys.argv[2])

# print(f" ***** job {array_number:02} started ***** ")
print(f" ***** job {job_number} started ***** ")
print("")

# L = 14
# maxsteps = 400
# maxdim = sp.special.binom(L, L//2)


# phys = 0.5731, 1.0

# phys0 = 0.20, 1.0
# phys1 = 1.80, 1.0

#######################################################################################################################################################################################################################################
#################################################################################################### Parallel Runs Date 03022025 ######################################################################################################
#######################################################################################################################################################################################################################################

if False:
    
    num_dims = 14
    truncation = np.linspace(1, maxdim, num_dims, dtype=np.int64)
    
    # #################################################################################################################### ED Part
    t_i = tt.time()

    print("len: ", int(maxdim))
    print("")

    _, GS_OBC_0 = ed.particle_count(L, phys0, K=2, PBC=False)
    _, GS_OBC_1 = ed.particle_count(L, phys1, K=2, PBC=False)

    print("- - - - - - - - - - - - - - - - - - - ED Run Time: ", tt.time() - t_i,"(s)")
    print("")
    print("ground state energies: V0=",GS_OBC_0[0]," , V1=", GS_OBC_1[0])
    print("")
    

    # #################################################################################################################### HF superposition truncation

    GS0, GS1 = GS_OBC_0[0], GS_OBC_1[0]
    
    truncated_data = []
        
    loop_t = tt.time()

    sup_ham_HF0, _ = hf.hart_fock_superposition(phys0, L, max_iters=maxsteps, PBC=False, basis_len = truncation[array_number])
    HF_E0, HF_U0 = np.linalg.eigh(sup_ham_HF0)
    truncated_data.append(np.abs((GS0 - HF_E0[0]) / GS0))
    
    
    sup_ham_HF1, _ = hf.hart_fock_superposition(phys1, L, max_iters=maxsteps, PBC=False, basis_len = truncation[array_number])
    HF_E1, HF_U1 = np.linalg.eigh(sup_ham_HF1)
    truncated_data.append(np.abs((GS1 - HF_E1[0]) / GS1))
    
    sup_ham_FF0, _ = ff.free_fermion_superposition(phys0, L, max_iters=maxsteps, PBC=False, basis_len = truncation[array_number])
    FF_E0, FF_U0 = np.linalg.eigh(sup_ham_FF0)
    truncated_data.append(np.abs((GS0 - FF_E0[0]) / GS0))

    sup_ham_FF1, _ = ff.free_fermion_superposition(phys1, L, max_iters=maxsteps, PBC=False, basis_len = truncation[array_number])
    FF_E1, FF_U1 = np.linalg.eigh(sup_ham_FF1)
    truncated_data.append(np.abs((GS1 - FF_E1[0]) / GS1))

    print(f"- - - - - - - - - - - - - - - - - - - Truncate size={truncation[array_number]} Run Time: ", tt.time() - loop_t,"(s)")
    print("")

    arcivo0 = open(f'Superposition/raw_data/TRN__{array_number:06}.npy', 'wb')
    np.save(arcivo0, truncated_data)
    arcivo0.close()
    
    print(f"Truncation Data file saved")
    print('\n')
    
    if truncation[array_number] == maxdim:
        HF_fin0 = 1.0000000000001 - np.cumsum(HF_U0[:,0]**2)
        HF_inf0 = np.array(HF_fin0)[np.array(truncation,dtype=np.int64)-1]
        arcivo1 = open(f'Superposition/raw_data/IFD__0__{array_number:06}.npy', 'wb')
        np.save(arcivo1, HF_inf0)
        arcivo1.close()
        
        HF_fin1 = 1.0000000000001 - np.cumsum(HF_U1[:,0]**2)
        HF_inf1 = np.array(HF_fin1)[np.array(truncation,dtype=np.int64)-1]
        arcivo2 = open(f'Superposition/raw_data/IFD__1__{array_number:06}.npy', 'wb')
        np.save(arcivo2, HF_inf1)
        arcivo2.close()
        
        FF_fin0 = 1.0000000000001 - np.cumsum(FF_U0[:,0]**2)
        FF_inf0 = np.array(FF_fin0)[np.array(truncation,dtype=np.int64)-1]
        arcivo3 = open(f'Superposition/raw_data/IFD__2__{array_number:06}.npy', 'wb')
        np.save(arcivo3, FF_inf0)
        arcivo3.close()
        
        FF_fin1 = 1.0000000000001 - np.cumsum(FF_U1[:,0]**2)
        FF_inf1 = np.array(FF_fin1)[np.array(truncation,dtype=np.int64)-1]
        arcivo4 = open(f'Superposition/raw_data/IFD__3__{array_number:06}.npy', 'wb')
        np.save(arcivo4, FF_inf1)
        arcivo4.close()
        
        print(f"Infidality Data file saved")
        print('\n')
    
    

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
    
    
# #################################################################################################################### Matrix elements plot
if False:
    
    t_i = tt.time()
    
    threshod=1.e-12

    if array_number == 0:
    
        supmat, _ = hf.hart_fock_superposition(phys0, L, max_iters=400, PBC=False)
        supmat[np.logical_and( supmat < threshod , supmat > -threshod ) ] = 0.0
    
        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
    
        arcivo = open(f'Superposition/raw_data/MEP__{array_number:02}.npy', 'wb')
        np.save(arcivo, supmat)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
    
    if array_number == 1:
    
        supmat, _ = hf.hart_fock_superposition(phys1, L, max_iters=400, PBC=False)
        supmat[np.logical_and( supmat < threshod , supmat > -threshod ) ] = 0.0
    
        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
        
        arcivo = open(f'Superposition/raw_data/MEP__{array_number:02}.npy', 'wb')
        np.save(arcivo, supmat)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
    
    if array_number == 2:
    
        supmat, _ = ff.free_fermion_superposition(phys0, L, max_iters=400, PBC=False)
        supmat[np.logical_and( supmat < threshod , supmat > -threshod ) ] = 0.0
    
        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
        
        arcivo = open(f'Superposition/raw_data/MEP__{array_number:02}.npy', 'wb')
        np.save(arcivo, supmat)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
        
    if array_number == 3:
    
        supmat, _ = ff.free_fermion_superposition(phys1, L, max_iters=400, PBC=False)
        supmat[np.logical_and( supmat < threshod , supmat > -threshod ) ] = 0.0

        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
        
        arcivo = open(f'Superposition/raw_data/MEP__{array_number:02}.npy', 'wb')
        np.save(arcivo, supmat)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
    
    elif array_number > 3:
        print(f"- - - - - - - - - - - - - - - - - - - Empty array {array_number} Run ")
        print("")
        
        
# #################################################################################################################### Shadow Matrix elements
if False:
    
    t_i = tt.time()
    
    # threshod=1.e-10

    if array_number == 0:
    
        shdwmat_hf = hf.hart_fock_shadowing(phys0, L, max_iters=400, PBC=False)
    
        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
    
        arcivo = open(f'Superposition/raw_data/SHW__{array_number:02}.npy', 'wb')
        np.save(arcivo, shdwmat_hf)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
    
    if array_number == 1:
    
        shdwmat_hf = hf.hart_fock_shadowing(phys1, L, max_iters=400, PBC=False)
    
        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
        
        arcivo = open(f'Superposition/raw_data/SHW__{array_number:02}.npy', 'wb')
        np.save(arcivo, shdwmat_hf)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
    
    if array_number == 2:
    
        shdwmat = ff.free_fermion_shadowing(phys0, L, max_iters=400, PBC=False)
    
        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
        
        arcivo = open(f'Superposition/raw_data/SHW__{array_number:02}.npy', 'wb')
        np.save(arcivo, shdwmat)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
        
    if array_number == 3:
    
        shdwmat = ff.free_fermion_shadowing(phys1, L, max_iters=400, PBC=False)

        print(f"- - - - - - - - - - - - - - - - - - - Matrix Element Run {array_number} Time: ", tt.time() - t_i,"(s)")
        print("")
        
        arcivo = open(f'Superposition/raw_data/SHW__{array_number:02}.npy', 'wb')
        np.save(arcivo, shdwmat)
        arcivo.close()
        print(f"- Data file {array_number} saved")
        print('\n')    
    
    if array_number > 3:
        print(f"- - - - - - - - - - - - - - - - - - - Empty array {array_number} Run ")
        print("")
        
        
        
# #################################################################################################################### Timing EigenValue Problem
if False:

    L = 18
    maxdim = sp.special.binom(L, L//2).astype(int)
    print("- ",L," - ", maxdim)
    print("")
    
    t_i = tt.time()
    mat = np.random.rand( maxdim , maxdim )
    H = np.tril(mat) + np.tril(mat, -1).T
    print(f"- Cearting Matrix {array_number} Time: ", tt.time() - t_i,"(s)")


    t_i = tt.time()
    E, U = np.linalg.eigh(H)
    print(f"- Diagonalization {array_number} Time: ", tt.time() - t_i,"(s)")

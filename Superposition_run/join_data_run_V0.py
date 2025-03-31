########################################################################################################################################################################################
################################################################################# Join & Plot 19022025 #################################################################################
########################################################################################################################################################################################
"""
V:0.0 created on 28032025
loading each (full) matrix in parallel (arrays) to:
    - produce datas for:
        - truncated energies
        - amplitudes
        - distances
"""

import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore
import glob
import sys

if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0


# folder_path = 'raw_data/'
# folder_path = '/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/raw_data/'


if False: ########################################################################### Loading the Hamiltoniand and saving Truncation, Infidality and Amps data - 19022025
    folder_path = 'Superposition_run/raw_data/'
    # all_files =  sorted( glob.glob('FulMat'+'*.npy', root_dir=folder_path) )
    # all_files2 = all_files[25:]
    
    all_files =  np.array(['FulMat__0.1_16.npy','FulMat__0.2_16.npy','FulMat__0.3_16.npy','FulMat__0.4_16.npy','FulMat__0.5_16.npy',
                            'FulMat__0.1_26.npy','FulMat__0.2_26.npy','FulMat__0.3_26.npy','FulMat__0.4_26.npy','FulMat__0.5_26.npy',
                           'FulMat__0.1_28.npy','FulMat__0.2_28.npy','FulMat__0.3_28.npy','FulMat__0.4_28.npy','FulMat__0.5_28.npy'])
    
    nome = all_files[array_number]
    
    num_datas = 20

    Vs = float(nome[8:11])
    Ls = int(nome[12:14])
    threshod = 1.e-12
        
    print(f"- Loading Full Matrix for L={Ls} and V={Vs} - data file name:",nome)
    print("")
    
    t_i = tt.time()
    ham = np.load(folder_path + nome)
    print(f"- Full Matrix Loding Time: ", tt.time() - t_i,"(s)")
    print("")
    maxdim = np.shape(ham)[0]    
    MAXDIM = sp.special.binom(Ls, Ls//2)    
    print(f"- Matrix size = {maxdim} - Full size = {MAXDIM}")
    print("")
    
    T_I = tt.time()
    trncs = np.linspace(1, maxdim, num_datas, dtype=np.int64)
    truncated_energy = []
    for l in trncs:
        t_i = tt.time()
        Es, vec = np.linalg.eigh(ham[:l,:l])
        print(f"- - - Eigenvalue Time: ", tt.time() - t_i,"(s) for ", l,"",l/maxdim)
        truncated_energy.append(Es[0])
        
        if l/maxdim == 1.0:
        
            GS = vec[:,0]
            amps = np.sort(np.abs(GS))[::-1] # np.abs(GS) # 
            amps = amps[ amps > threshod ]
            print("- - Length of sorted Amps:", len(amps)," Length of Half size:", maxdim //2)
            print("")
            
            arcivo = open(f'Superposition_run/raw_data/AMPS_{Vs}_{Ls:02}.npy', 'wb')
            np.save(arcivo, amps)
            arcivo.close()    
                    
            # infedality = 1.0000000000001 - np.cumsum(GS**2)
            # infedality = np.array(infedality)[np.array(trncs,dtype=np.int64)-1]
            
            # arcivo = open(f'Superposition_run/raw_data/INFD_{Vs}_{Ls:02}.npy', 'wb')
            # np.save(arcivo, infedality)
            # arcivo.close()
            
    print(f"- All Eigenvalues Time: {tt.time() - T_I} (s) for L={Ls} and V={Vs}")
    print("")
        
    arcivo = open(f'Superposition_run/raw_data/TRNC_{Vs}_{Ls:02}.npy', 'wb')
    np.save(arcivo, np.array([truncated_energy, trncs]))
    arcivo.close()
    print(f"- Truncation Data file saved")
    print("")
        
    
    # t_i = tt.time()
    # Es, vec = np.linalg.eigh(ham)
    # print(f"- Eigenvalue Calculation Time: ", tt.time() - t_i,"(s)")
    # print("")
    

if False: ########################################################################### Read the Hamiltoniand and saving Infidality and Amps data - 19022025
    Amplitude = True
    Fidality = False
    
    folder_path = 'Superposition_run/raw_data/Full_Matrix/'
    # all_files =  sorted( glob.glob('FulMat'+'*.npy', root_dir=folder_path) )
    # all_files2 = all_files[25:]
    
    all_files =  np.array([
                            #'FulMat__0.1_08.npy','FulMat__0.2_08.npy','FulMat__0.3_08.npy','FulMat__0.4_08.npy','FulMat__0.5_08.npy',
                            #'FulMat__0.1_10.npy','FulMat__0.2_10.npy','FulMat__0.3_10.npy','FulMat__0.4_10.npy','FulMat__0.5_10.npy',
                            'FulMat__0.1_12.npy','FulMat__0.2_12.npy','FulMat__0.3_12.npy','FulMat__0.4_12.npy','FulMat__0.5_12.npy',
                            'FulMat__0.1_14.npy','FulMat__0.2_14.npy','FulMat__0.3_14.npy','FulMat__0.4_14.npy','FulMat__0.5_14.npy',
                            'FulMat__0.1_16.npy','FulMat__0.2_16.npy','FulMat__0.3_16.npy','FulMat__0.4_16.npy','FulMat__0.5_16.npy'
                           ])
    
    nome = all_files[array_number]

    Vs = float(nome[8:11])
    Ls = int(nome[12:14])
    threshod = 1.e-12
        
    print(f"- Loading Full Matrix for L={Ls} and V={Vs} - data file name:",nome)
    print("")
    
    t_i = tt.time()
    ham = np.load(folder_path + nome)
    print(f"- Full Matrix Loding Time: ", tt.time() - t_i,"(s)")
    print("")
    maxdim = np.shape(ham)[0]    
    MAXDIM = sp.special.binom(Ls, Ls//2)    
    print(f"- Matrix size = {maxdim} - Full size = {MAXDIM}")
    print("")
    
    # ######################### Amplitudes DATA ##########################
    if Amplitude:
        Es, vec = np.linalg.eigh(ham)
        print(f"- Eigenvalue Time: ", tt.time() - t_i,"(s) for ", maxdim,"",MAXDIM)
        GS = vec[:,0]
        amps = np.abs(GS) 
        # np.sort(np.abs(GS))[::-1]
        # amps = amps[ amps > threshod ]
        # print("- Length of sorted Amps:", len(amps)," Length of Half size:", maxdim//2)
        # print("")        
        arcivo = open(f'Superposition_run/raw_data/Amplitudes/AMP_{Vs}_{Ls:02}.npy', 'wb')
        np.save(arcivo, amps)
        arcivo.close()    
        print(f"- Amplitudes Data file saved")
        print("")
    
    # ######################### INFIDALITY DATA ##########################
    if Fidality:
        truncation = np.linspace(1, maxdim, 5, dtype=np.int64)
        infedality = 1.0000000000001 - np.cumsum(GS**2)
        infedality = np.array(infedality)[np.array(truncation,dtype=np.int64)-1]
        arcivo = open(f'Superposition_run/raw_data/INFD_{Vs}_{Ls:02}.npy', 'wb')
        np.save(arcivo, infedality)
        arcivo.close()
        print(f"- Infidality Data file saved")
        print('')


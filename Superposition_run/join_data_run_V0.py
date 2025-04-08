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


if False: ########################################################################### Loading the Hamiltoniand and saving Truncation, Infidality, Amplitudes and Distances - 20250401
    folder_path = 'Superposition_run/raw_data/'
    # all_files =  sorted( glob.glob('FulMat'+'*.npy', root_dir=folder_path) )
    # all_files2 = all_files[25:]
    
    all_files =  np.array(['FulMat__0.1_16.npy','FulMat__0.2_16.npy','FulMat__0.3_16.npy','FulMat__0.4_16.npy','FulMat__0.5_16.npy',
                            'FulMat__0.1_14.npy','FulMat__0.2_14.npy','FulMat__0.3_14.npy','FulMat__0.4_14.npy','FulMat__0.5_14.npy'
                            # 'FulMat__0.1_28.npy','FulMat__0.2_28.npy','FulMat__0.3_28.npy','FulMat__0.4_28.npy','FulMat__0.5_28.npy'
                           ])
    
    nome = all_files[array_number]
    
    num_datas = 25

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
            amps = np.abs(GS) # np.sort(np.abs(GS))[::-1] #   
            # amps = amps[ amps > threshod ]
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
    

if False: ########################################################################### Read the Hamiltoniand and saving Infidality and Amplitudes and Distances data - 20250401
    Amplitude = True
    Fidality = False
    
    folder_path = 'Superposition_run/raw_data/Full_Matrix/'
    # all_files =  sorted( glob.glob('FulMat'+'*.npy', root_dir=folder_path) )
    # all_files2 = all_files[25:]
    
    all_files =  np.array([
                            #'FulMat__0.1_08.npy','FulMat__0.2_08.npy','FulMat__0.3_08.npy','FulMat__0.4_08.npy','FulMat__0.5_08.npy',
                            #'FulMat__0.1_10.npy','FulMat__0.2_10.npy','FulMat__0.3_10.npy','FulMat__0.4_10.npy','FulMat__0.5_10.npy',
                            #'FulMat__0.1_12.npy','FulMat__0.2_12.npy','FulMat__0.3_12.npy','FulMat__0.4_12.npy','FulMat__0.5_12.npy',
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


if True: ########################################################################### Read and save data from array runs - new algorithm for fixed step_size/L/V - 20250407
    
    mid_plot = False
    fin_plot = False
    
    folder_path = 'Superposition_run/raw_data/test/'
    all_files =  sorted( glob.glob('DATA'+'*0.2'+'*14'+'*.npy', root_dir=folder_path) )
    # print(all_files)
    # print("")
    
    Vs, Ls = 0.2, 14
    step = 100 #data_dic['step_size']
    # threshod = 1.e-12
    
    maxdim = sp.special.binom(Ls, Ls//2)    
    # bonds = [1] + [b for b in range(150, int(maxdim), 150)] + [int(maxdim)] #np.linspace(1, maxdim, num_datas, dtype=np.int64)
    # bonds = np.array(bonds, dtype=np.int64)
    # print("size of bonds ", len(bonds))
    
    ed_energy = np.load('Superposition_run/raw_data/Ground_State_Energy/' + f"EDGS_{Vs}_{Ls:02}.npy")

    # nome = 'DATA_0.2_14_10.npy'
    # Vs = float(nome[5:8])
    # Ls = int(nome[9:11])
    # print("physical is: ",Vs,", size is ",Ls)
    # print("")

    if mid_plot:
        mid_fig, mid_ax = plt.subplots(1,1, figsize=(9, 7), 
                subplot_kw=dict( 
                    yscale = 'log', xscale ='linear',
                    title = f"All |M| convergence for L = {Ls}, V={Vs} & |m|={step}",
                    ylabel = r'$|\frac{E-E_{ed}}{L}|$', xlabel = r'$\#\: iterations$',
                    # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                    xticks= [x for x in range(0, int(maxdim/step +5), 5)], #xticklabels=[str(x) for x in range(0, int(maxdim/step +5), 5)],
                    yticks=[10**(-s) for s in range(4,17,1)],    
                    ),
                )
    

    new_truncated_energy = []
    old_truncated_energy = []
    new_basis_set = []
    # new_distan_counts = []
    bond_set = []
    for indx, nome in enumerate(all_files):
        print(" - file name is ",nome)
        data_dic = np.load(folder_path + nome , allow_pickle=True).item()
        
        bond = data_dic['bond_size']
        bond_set.append(bond)
        
        new_energies = data_dic['new']
        print(" - size of new energies ", len(new_energies))
        if np.abs(new_energies[-1] - ed_energy) <= 0.0:
            print("  ****new energies problem****")
            print(f" - - ground state  energy for bond {bond} is {ed_energy:.22f}")
            print(f" - - new truncated energy for bond {bond} is {new_energies[-1] :.22f}")
            print(f" - - old truncated energy for bond {bond} is {data_dic['old'] :.22f}")
            print("")
            new_energies[-1] = ed_energy + 1.e-16
        new_truncated_energy.append(new_energies[-1])
            
        old_truncated_energy.append(data_dic['old'])
        
        # new_distan_counts.append(data_dic['dstn_count'])
        
        new_basis_set.append(data_dic['basis_set'])
        
        # print(f" - ground state  energy for bond {bonds[indx]} is {ed_energy:.22f}")
        # print(f" - new truncated energy for bond {bonds[indx]} is {new_energies[-1] :.22f}")
        # print(f" - old truncated energy for bond {bonds[indx]} is {data_dic['old'] :.22f}")
        # print("")
        
        if mid_plot:
            mid_energy = np.abs(np.array(new_energies) - ed_energy)/Ls
            mid_ax.plot(mid_energy, marker='o', label=f"M={bond}", markersize=5)
                

    arcivo = open(f'Superposition_run/raw_data/test/NBTS_{Vs}_{Ls:02}.npy', 'wb') #NBTS : new best truncated state    #BTGS : best truncated ground state
    # np.save(arcivo, np.array([new_truncated_energy, bonds]))
    np.save(arcivo, np.array([new_truncated_energy, bond_set]))
    arcivo.close()
    print(f"New Bond-Search Data file saved")
    print("")
    
    arcivo = open(f'Superposition_run/raw_data/test/TRNC_{Vs}_{Ls:02}.npy', 'wb')
    np.save(arcivo, np.array([old_truncated_energy, bond_set]))
    arcivo.close()
    print(f"Old Truncated Data file saved")
    print("")
    
    arcivo = open(f'Superposition_run/raw_data/test/BBST_{Vs}_{Ls:02}.npy', 'wb') #BBST : best basis set
    np.save(arcivo, new_basis_set)
    arcivo.close()
    print(f"Best Basis Set file saved")
    print("")
    
    
    
    if mid_plot:
        mid_ax.grid(which='major', axis='y', linestyle=':')
        mid_ax.legend(loc='best', ncol=2, fontsize='small', markerscale=0.9)
        mid_fig.savefig(f"Superposition_run/output/All_Bonds_mid_converge_V{Vs}_L{Ls}_J{job_number}.pdf", bbox_inches = 'tight')
        print(f"Mid convergence plot saved")
        print("")
    
    if fin_plot:    
        fig, ax = plt.subplots(1,1, figsize=(8, 6), 
                subplot_kw=dict( 
                    yscale= 'log', xscale= 'linear',
                    title= f"comparing the convergence new vs old for L = {Ls}, V={Vs} & |m|={step}",
                    ylabel= r'$|\frac{E-E_{ed}}{L}|$', xlabel = r'$|M|$',
                    xticks= bond_set[::2], #xticklabels=[str(x) for x in bonds],
                    yticks=[10**(-s) for s in range(4,17)],    
                    # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                    ),
                )
        
        new_truncated_energy = np.abs(np.array(new_truncated_energy) - ed_energy)/Ls
        # ax.plot( bonds, new_truncated_energy, label="new", ls='--', linewidth=0.7, marker='o')
        ax.plot( bond_set, new_truncated_energy, label="new", ls='--', linewidth=0.7, marker='o')
        old_truncated_energy = np.abs(np.array(old_truncated_energy) - ed_energy)/Ls
        ax.plot( bond_set, old_truncated_energy, label="old", ls='--', linewidth=0.7, marker='d')
        ax.grid(which='major', axis='y', linestyle=':')
        ax.legend(loc='best')
        fig.savefig(f"Superposition_run/output/__comapring_new_vs_old_method_V_{Vs}_L{Ls}_J{job_number}.pdf", bbox_inches = 'tight')
        print(f"A convergence plot saved")
        print("")
        
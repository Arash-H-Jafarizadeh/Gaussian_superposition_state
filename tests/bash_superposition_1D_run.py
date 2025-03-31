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


# sys.path.append('/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/')


import free_fermion_function as ff
import exact_diagonalization_function as ed
import HF_function as hf


# array_number = int(sys.argv[1])
job_number = int(sys.argv[1])

# print(f" ***** job {array_number:02} started ***** ")
print(f" ***** job {job_number} started ***** ")
print("")

L = 8

num_dims = 10

maxsteps = 200

maxdim = sp.special.binom(L, L//2)

phys = 0.5731, 1.0

phys0 = 0.20, 1.0
phys1 = 1.80, 1.0

#################################################################################################### test run 31012025 ######################################################################################################
# #################################################################################################################### ED
if False:
    t_i = tt.time()

    print("len: ", int(maxdim))
    print("")

    _, GS_PBC = ed.particle_count(L, phys0, K=6, PBC=True)
    _, GS_OBC = ed.particle_count(L, phys0, K=6, PBC=False)

    print("- - - - - - - - - - - - - - - - - - - ED Run Time: ", tt.time() - t_i,"(s)")
    print("")
    # print("ED energy (",len(energy_ED),")", energy_ED)
    # print("ground state energies: PBC=",GS_PBC[0]," , OBC=", GS_OBC[0])
    # print("")
    print("PBC ground state energies: PBC=",GS_PBC)
    print("")
    print("OBC ground state energies: OBC=",GS_OBC)
    print("")
    # print("ED particle:", len(particle_ED),",", particle_ED)
    # print("")

    # uniq_energi, uniq_indx = np.unique( np.round(energy_ED, decimals=10), return_index=True)
    # print(" - - unique energy (",len(uniq_energi),")", uniq_energi)
    # print("")

    t_i = tt.time()
    sup_ham = hf.hart_fock_shadowing(phys, L, PBC=False, max_iters=maxsteps, start_point=1.e-5, extra_check=False)
    print(sup_ham)
    print("")
    # sup_ham, new_order = hf.hart_fock_superposition(phys, L, PBC=False, max_iters=maxsteps, start_point=1.e-8)
    # sup_ham = ff.free_fermion_superposition(phys, L, PBC=False, max_iters=maxsteps, start_point=1.e-1)
    # super_energy, vec = sp.linalg.eigh(sup_ham,  subset_by_index=[0,11])

    print("- - - - - - - - - - - - - - - - - - - SUP Run Time: ", tt.time() - t_i,"(s)")
    print("")
    

# #################################################################################################################### HF superposition truncation + Plot
if False:
    t_i = tt.time()

    # pnt_list =  [1]+[10*x for x in range(1,6)]+[70]
    pnt_list = np.linspace(1,maxdim, num_dims, dtype=np.int64)
    
    super_data_PBC = []
    super_ff_PBC = []
    super_data_OBC = []
    super_ff_OBC = []

    for trunc in pnt_list:
        
        loop_t = tt.time()

        # sup_ham_PBC, _ = hf.hart_fock_superposition(phys0, L, max_iters=maxsteps, PBC=True, basis_len = trunc)
        # new_E_PBC, new_U_PBC = np.linalg.eigh(sup_ham_PBC)
        # super_data_PBC.append(np.abs((GS_PBC[0] - new_E_PBC[0]) / GS_PBC[0]))
        
        # ff_ham_PBC, _ = ff.free_fermion_superposition(phys0, L, max_iters=maxsteps, PBC=True, basis_len = trunc)
        # ff_E_PBC, ff_U_PBC = np.linalg.eigh(ff_ham_PBC)
        # super_ff_PBC.append(np.abs((GS_PBC[0] - ff_E_PBC[0]) / GS_PBC[0]))

        sup_ham_OBC, _ = hf.hart_fock_superposition(phys0, L, max_iters=maxsteps, PBC=False, basis_len = trunc)
        new_E_OBC, new_U_OBC = np.linalg.eigh(sup_ham_OBC)
        super_data_OBC.append(np.abs((GS_OBC[0] - new_E_OBC[0]) / GS_OBC[0]))
        print("HF DATA: ",super_data_OBC)
        
        ff_ham_OBC, _ = ff.free_fermion_superposition(phys0, L, max_iters=maxsteps, PBC=False, basis_len = trunc)
        ff_E_OBC, ff_U_OBC = np.linalg.eigh(ff_ham_OBC)
        super_ff_OBC.append(np.abs((GS_OBC[0] - ff_E_OBC[0]) / GS_OBC[0]))
        print("FF DATA: ", super_ff_OBC)
        
        print("Loop Run Time: ", tt.time() - loop_t,"(s)")

    print("- - - - - - - - - - - - - - - - - - - Superposition Run Time: ", tt.time() - t_i,"(s)")
    print("")

    t_i = tt.time()

    # pl.plot(pnt_list, super_data_PBC, label=f'HF PBC', ls='--',linewidth=0.8, marker='s')
    pl.plot(pnt_list, super_data_OBC, label=f'HF OBC', ls='--',linewidth=0.8, marker='D')
    # pl.plot(pnt_list, super_ff_PBC, label=f'FF PBC', ls='--',linewidth=0.8, marker='o')
    pl.plot(pnt_list, super_ff_OBC, label=f'FF OBC', ls='--',linewidth=0.8, marker='D', fillstyle='none')

    pl.title(f"Superposition energy for V={phys0[0]} and L={L} ")
    pl.xlabel(r"number of states in superposition basis")    
    pl.xticks(pnt_list)
    # pl.xscale('log')

    pl.ylabel(r"$|\frac{E - E_{ed}}{E_{ed}}| $")    
    pl.yscale('log')
    pl.grid(which='major', axis='both', linestyle=':')

    pl.legend(loc='best')
    pl.savefig(f"data/SuperEnrgy_NEW_L{L}_(PBC,OBC)(HF,FF)_J{job_number}.pdf", bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Plotting Run Time: ", tt.time() - t_i,"(s)")
    print("")

# #################################################################################################################### HF superposition amplitudes + plot
if False:

    t_i = tt.time()

    V_list =  [0.2, 0.6, 1.0, 1.4, 1.8] 
    
    # fig = pl.figure(figsize=(12, 7))
    # blues = pl.cm.get_cmap('Blues_r',num_plts+3)
    # blues = pl.cm.get_cmap('autumn',num_plts)
    marks = ['o','*','^','v','p','s','D','h','8']

    for plt, Vs in enumerate(V_list):
        
        loop_t = tt.time()
        
        sup_ham = hf.hart_fock_superposition([Vs, 1.0], L, PBC=False, max_iters=maxsteps, start_point=1.e-10)
        super_energy, vec = np.linalg.eigh(sup_ham)
        pl.plot(np.abs(vec[:,0]), label=f'V={Vs:.2f}', color=f'C{plt}', marker=marks[plt], linestyle='', linewidth=0.8, alpha=1-(plt*0.75)/len(V_list))
        
        # sup_ham2 = hart_fock_superposition([Vs,1.0], L, PBC=False, max_iters=400)
        # super_energy2, vec2 = np.linalg.eigh(sup_ham2)
        # pl.plot(np.abs(vec2[:,0]), label=f'left V={Vs:.2f}', color=f'C{plt+1}', marker=marks[plt+1], linestyle='', linewidth=0.8, alpha=1-(plt*0.5)/num_plts)
        
        print("Loop Run Time: ", tt.time() - loop_t,"(s)")

    print("- - - - - - - - - - - - - - - - - - - Superposition Run Time: ", tt.time() - t_i,"(s)")
    print("")

    t_i = tt.time()

    pl.title("Amplitudes for different V's for L=8 (HF)")
    pl.xlabel(r"$n$ as in $|n\rangle $", fontsize='large')    
    pl.ylabel(r"$|a_n| $", rotation='horizontal', fontsize='large')    
    pl.yscale('log')
    pl.ylim(10**-18,10)
    pl.legend(loc='best',ncols=2)
    pl.grid(which='major', axis='y', linestyle=':')
    pl.savefig(f"data/SuperAmps_L{L}_(HF)(OBC)(log)_J{job_number}.pdf", bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Plotting Run Time: ", tt.time() - t_i,"(s)")
    print("")
    
    
# #################################################################################################################### Matrix elements + plot
if False:
    
    t_i = tt.time()
    
    threshod=1.e-8

    # supmat0, _ = hf.hart_fock_superposition(phys0, L, max_iters=400, PBC=False)
    # v_fac0 =  np.round(np.max([abs(np.max(supmat0)), abs(np.min(supmat0))]), decimals = 5)
    # supmat0[np.logical_and( supmat0 < threshod , supmat0 > -threshod ) ] = 0.0

    # v_fac1 =  np.round(np.max([abs(np.max(supmat1)), abs(np.min(supmat1))]), decimals = 5)
    # supmat1[np.logical_and( supmat1 < threshod , supmat1 > -threshod ) ] = 0.0

    # v_fac = np.max([v_fac1,v_fac2])
    v_fac = 9.0
    # supmat0 = hf.hart_fock_shadowing(phys1, L, max_iters=400, PBC=False)
    # supmat1 = hf.hart_fock_shadowing(phys1, L, max_iters=400, PBC=False, extra_check = True)

    print("- - - - - - - - - - - - - - - - - - - Matrix Element Run Time: ", tt.time() - t_i,"(s)")
    print("")

    t_i = tt.time()

    fig, ax = pl.subplots(1,2, figsize=(8, 10) )

    norm=colors.SymLogNorm(linthresh=1.e-5, linscale=1e-1, vmin=-v_fac, vmax=v_fac)

    im0 = ax[0].imshow(supmat0, norm=norm, cmap='bwr', interpolation='nearest')
    ax[0].set_title(f'matrix elements for V={phys0[0]}', pad=0)
    # ax[0].xaxis.tick_top()

    im1 = ax[1].imshow(supmat1, norm=norm, cmap='bwr', interpolation='nearest')
    ax[1].set_title(f'matrix elements for V={phys1[0]}', pad=0)
    # ax[1].xaxis.tick_top()

    fig.colorbar(im0, ax=ax, fraction=0.0215, pad=0.04, ticks=[-10,-1.0,-0.1,-0.01,-1.e-3,-1.e-6,0,1.e-6,1.e-5,1.e-3,0.01,0.1,1.0,10])#orientation='vertical')
    # fig.suptitle('Set a Single Main Title for All the Subplots', fontsize=16, y=.84)

    pl.savefig(f"data/SuperMatShadow_L{L}_(HF)(OBC)_J{job_number}.pdf", dpi=300, bbox_inches = 'tight')

    print("- - - - - - - - - - - - - - - - - - - Matrix Element Ploting Time: ", tt.time() - t_i,"(s)")
    print("")
    
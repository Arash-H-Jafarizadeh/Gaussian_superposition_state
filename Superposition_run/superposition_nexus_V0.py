import sys
import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors # type: ignore

sys.path.append('/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/source_code/')

# import free_fermion_function as ff
import exact_diagonalization_function as ed
import hartree_fock_function as hf


if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0


print(f" ***** job {job_number} started ***** ")
print("")

threshod = 1.e-13
hf_iters = 400

if False: ###################################################################### Testing improved version of the nexus method
    print(" ***** Testing improved version of the nexus method ***** ")
    mid_plot = True
    
    L = 10
    physical = 0.2, 1.0 
    maxdim = sp.special.binom(L, L//2)    
    ed_energy = {8:-4.670424529225037, 10:-5.908793067783503, 12:-7.14882332720711}

    t_0 = tt.time()
    HF_E, HF_U = hf.new_hart_fock_optimization(physical, L, max_iters=hf_iters, PBC=False, start_point=1.e-4)
    print(f"- Hartree Fock Optimization Time:", tt.time() - t_0,"(s)")
    
    t_1 = tt.time()
    super_ham, super_basis = hf.based_ham(physical, L, HF_E, HF_U, PBC=False)# #, basis_len = bond
    print(f"- Full Matrix Creation Time: ", tt.time() - t_1,"(s)")
    print("")

    bonds = np.array([1] + [b for b in range(30, int(maxdim), 30)] + [int(maxdim)]) 
    step = 20

    new_data = np.zeros(len(bonds), dtype=np.float128)
    old_data = np.zeros(len(bonds), dtype=np.float128)
    blind_data = np.zeros(len(bonds), dtype=np.float128)
    # DSTNS = np.arange(L//2+1)
    # COUNTS = np.zeros((len(bonds),L//2+1), dtype=np.float16)

    if mid_plot:
        mid_fig, mid_ax = plt.subplots(1,1, figsize=(9, 7), 
                subplot_kw=dict( 
                    yscale = 'log', xscale ='linear',
                    title = f"All |M| convergence for L = {L}, V={physical[0]} & |m|={step}",
                    ylabel = r'$|\frac{E-E_{ed}}{L}|$', xlabel = r'$\#\: iterations$',
                    # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                    xticks= bonds, #xticklabels=[str(x) for x in range(0, int(maxdim/step +5), 5)],
                    yticks=[10**(-s) for s in range(4,17,1)],    
                    ),
                )

    t_2 = tt.time()
    for indx, bond in enumerate(bonds):
        print(f"* * Bond is: {bond}")

        t_2_0 = tt.time()
        nexus_energy, nexus_bond, nexus_amps = hf.nexus_optimization(physical, L, bond, HF_E, HF_U, size_step = step, PBC=False, max_iters = hf_iters)
        print("- - Time for nexus:", tt.time()-t_2_0)
        nexus_energy = np.abs(np.array(nexus_energy) - ed_energy[L])/L
        new_data[indx] = nexus_energy[-1]
                
        t_2_1 = tt.time()
        test_energy, test_bond, test_amps = hf.new_hf_optimization(physical, L, bond, size_step = step, PBC=False, max_iters = hf_iters)
        print("- - Time for old:", tt.time()-t_2_1)
        test_energy = np.abs(np.array(test_energy) - ed_energy[L])/L
        old_data[indx] = test_energy[-1]

        # dists, counts = np.unique(hf.basis_distance(test_bond, L),return_counts=True)
        # counts = (counts/np.sum(counts))*100
        # # print("- - ", dists)        # print("- - ", counts)
        # dicd = dict(zip(dists,counts))
        # for n in dists:
        #     COUNTS[indx, n] = dicd[n]

        t_2_2 = tt.time()
        Es, _ = np.linalg.eigh(super_ham[:bond,:bond])
        print(f"- - Eigenvalue Time: ", tt.time() - t_2_2,"(s) for ", bond,",",bond/maxdim)
        blind_data[indx] = (Es[0] - ed_energy[L])/L

        print("- - - nexus:", nexus_energy[-1])
        print("- - - older:", test_energy[-1])
        print("- - - blind:", (Es[0] - ed_energy[L])/L)
        
        print("")

        if mid_plot:
            # mid_energy_nexus = np.abs(np.array(new_energies) - ed_energy)/Ls
            mid_ax.plot(nexus_energy, marker='o', label=f"new, M={bond}", linestyle=":", markersize=4)
            mid_ax.plot(test_energy, marker='d', label=f"old, M={bond}", linestyle=":", markersize=4)
            # mid_ax.plot(mid_energy, marker='*', label=f"blnd, M={bond}", linestyle=":", markersize=4)


    print(f"- Full Run Time: ", tt.time() - t_2,"(s)")

    # print(COUNTS)
    
    
    
    if mid_plot:
        mid_ax.grid(which='major', axis='y', linestyle=':')
        mid_ax.legend(loc='best', ncol=2, fontsize='small', markerscale=0.9)
        mid_fig.savefig(f"Superposition_run/output/_test_nexus_vs_old_converge_V{physical[0]}_L{L}_J{job_number}.pdf", bbox_inches = 'tight')
        print(f"Mid convergence plot saved")
        print("")

    fig, ax = plt.subplots(1,1, figsize=(8, 6), 
            subplot_kw=dict( 
                yscale= 'log', xscale= 'linear',
                title= f"comparing the convergence nexus vs old vs blind for L = {L}, V={physical[0]} & |m|={step}",
                ylabel= r'$|\frac{E-E_{ed}}{L}|$', xlabel = r'$|M|$',
                xticks= bonds, #xticklabels=[str(x) for x in bonds],
                yticks=[10**(-s) for s in range(4,17)],    
                # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                ),
            )
    
    # new_truncated_energy = np.abs(np.array(new_truncated_energy) - ed_energy)/Ls
    # ax.plot( bonds, new_truncated_energy, label="new", ls='--', linewidth=0.7, marker='o')
    ax.plot( bonds, new_data, label="nexus", ls='--', linewidth=0.7, marker='o')
    # old_truncated_energy = np.abs(np.array(old_truncated_energy) - ed_energy)/Ls
    ax.plot( bonds, old_data, label="old", ls='--', linewidth=0.7, marker='d')
    ax.plot( bonds, blind_data, label="blind", ls='--', linewidth=0.7, marker='*')
    ax.grid(which='major', axis='y', linestyle=':')
    ax.legend(loc='best')
    fig.savefig(f"Superposition_run/output/_test_comapring_nexus_vs_old_method_V_{physical[0]}_L{L}_J{job_number}.pdf", bbox_inches = 'tight')
    print(f"A convergence plot saved")
    print("")    

if True: ###################################################################### creating the data for each array job in a dict format
    mid_plot = True
    save_data = True
    
    Ls = 16
    Vs, Js = 0.3, 1.0 
    ed_energy = np.load('Superposition_run/raw_data/Ground_State_Energy/' + f"EDGS_{Vs}_{Ls:02}.npy")
    
    # num_pnts = 23
    # b_step = int( 10 * np.round((maxdim/num_pnts)/10))
    b_step = 700
    step = 1000
    
    maxdim = sp.special.binom(Ls, Ls//2)    
     
    bonds = np.array( [1] + [b for b in range(b_step, int(maxdim), b_step)] + [int(maxdim)] , dtype=np.int64) # B should come here
    print(f"- length of bonds is: {np.size(bonds)}")
    print(f"- Array {array_number}th bond is {bonds[array_number]}")
    
    t_0 = tt.time()
    HF_E, HF_U = hf.new_hart_fock_optimization([Vs, Js], Ls, max_iters=hf_iters, PBC=False, start_point=1.e-14)
    print(f"- Hartree Fock Optimization Time:", tt.time() - t_0,"(s)")
    

    array_dict = {}
    array_dict['step_size'] = step
    COUNTS = np.zeros((Ls//2+1), dtype=np.float16)

    t_2 = tt.time()
    bond = bonds[array_number]
    array_dict['bond_size'] = bond
    
    t_3 = tt.time()
    nexus_energy, nexus_basis, nexus_amps = hf.nexus_optimization([Vs, Js], Ls, bond, HF_E, HF_U, size_step = step, PBC=False, max_iters = hf_iters)
    print("- - Time for nexus run:", tt.time()-t_3)
    print("")
    array_dict['new_energy'] = nexus_energy
    array_dict['new_basis'] = nexus_basis
    array_dict['new_amps'] = nexus_amps
        # nexus_energy = np.abs(np.array(nexus_energy) - ed_energy[L])/L
        # new_data[indx] = nexus_energy[-1]
    print("- - TESTING if bond is: ",len(nexus_basis))
    
    
    """
    **IMPROVEMENT**:
        this part went down to avoid making the full matrix and just using submatrix of it. Or just call a saved matrix for old method
    """
    # t_1 = tt.time() 
    # super_ham, super_basis = hf.based_ham([Vs, Js], Ls, HF_E, HF_U, PBC=False) #, basis_len = bond
    # print(f"- Full Matrix Creation Time: ", tt.time() - t_1,"(s)")
    # print("")
    
    # t_1 = tt.time() ######## IMPROVEMENT: below I just use a saved matrix to avoid making the full matrix and use the order_basis function to get full basis set
    # super_ham, super_basis = hf.based_ham([Vs, Js], Ls, HF_E, HF_U, PBC=False, basis_len = bond)
    # print(f"- Bond Matrix Creation Time: ", tt.time() - t_1,"(s)")
    # print("")
    
    super_ham = np.load(f'Superposition_run/raw_data/Full_Matrix/FulMat__{Vs}_{Ls:02}.npy')
    super_basis = hf.ordered_basis(Ls, HF_E)
    
    t_4 = tt.time()
    Es, Us = np.linalg.eigh(super_ham[:bond,:bond])
    print(f"- - Eigenvalue Time (old): ", tt.time() - t_4,"(s) for ", bond,",",bond/maxdim)
    print("")
    array_dict['old_energy'] = Es[0] #(Es[0] - ed_energy)/Ls
    # array_dict['old_amps'] = Us[:,0] 
    
    if bond == bonds[-1]:
        array_dict['full_amps'] = Us[:,0] 
        array_dict['full_basis'] = super_basis 


    dists, counts = np.unique(hf.basis_distance(nexus_basis, Ls),return_counts=True)
    dicd = dict(zip(dists,counts))
    for n in dists:
        COUNTS[n] = dicd[n]

    print(f"- - counts for bond {bond} is {COUNTS}")
    print(f"- - order of amplitudes for bond {bond} is {np.floor(np.log10(np.abs(nexus_amps)))}")
    print("")

    if mid_plot:
        mid_energy = np.abs(np.array(nexus_energy) - ed_energy)/Ls
        plt.plot(mid_energy, marker='o', label=f"M = {bond}")
        blnd_energy = np.abs(Es[0] - ed_energy)/Ls
        plt.plot(np.full(np.size(mid_energy), blnd_energy), marker='d', label=f" blind")
        plt.yscale('log')
        plt.legend(loc='best', fontsize='small', markerscale=0.9)
        plt.title(f"L = {Ls}, V={Vs} & |m|={step}")
        plt.savefig(f"Superposition_run/plots/mid_plots/_mid_converge_V{Vs}_L{Ls}_B{bond:04}_A{array_number:02}.pdf", bbox_inches = 'tight')
    
    print(f"- Full Run {array_number} Time: ", tt.time() - t_2,"(s)")
    print("")
    
    print("- Size of full new data:", len(nexus_energy))
    print("")
    
    if save_data:       
        arcivo = open(f'Superposition_run/raw_data/test/DATA_{Vs}_{Ls:02}_{array_number:02}.npy', 'wb')
        np.save(arcivo, array_dict)
        arcivo.close()
        
        print(f"- Data files are saved")
        print("") 

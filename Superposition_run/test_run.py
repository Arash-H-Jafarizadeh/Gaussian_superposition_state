import sys
import numpy as np # type: ignore
import scipy as sp # type: ignore
import time as tt
import matplotlib.pyplot as plt # type: ignore
import matplotlib.colors as colors

# from general_quadratic_function import *
# from gaussian_state_function import *
# from MF_function import *
# from circuit_vqe_function import *

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
maxsteps = 400




L = 10
physical = 0.2, 1.0 
MAXDIM = sp.special.binom(L, L//2)    
ed_energy = {8:-4.670424529225025, 10:-5.908793067783503, 12:-7.14882332720711}
# print(ed_energy[L])

if True:#####################################################################################################################################################    
    t_i = tt.time()
    super_ham, super_basis = hf.hart_fock_superposition( physical, L, max_iters=maxsteps, PBC=False, start_point=1.e-4) #, basis_len = bond
    print(f"- Full Matrix Creation Run {array_number} Time: ", tt.time() - t_i,"(s)")
    print("")


    x_data = np.arange(20, 230, 20)
    step = 20

    new_data = np.zeros(len(x_data), dtype=np.float64)
    old_data = np.zeros(len(x_data), dtype=np.float64)
    DSTNS = np.arange(L//2+1)
    COUNTS = np.zeros((len(x_data),L//2+1), dtype=np.float16)

    T_i = tt.time()
    for indx, bond in enumerate(x_data):
        t0 = tt.time()
        print(f"- - Bond is: {bond}")
        # bond = 30

        test_energy, test_bond, test_amps = hf.new_hf_optimization(physical, L, bond, size_step = step, PBC=False, max_iters = 200)
        print("- - Time for bond:", tt.time()-t0)
        print("- - Bond len is: ",len(test_bond))

        test_energy = (np.array(test_energy) - ed_energy[L])/L
        test_energy[-1] = 1.e-17 if test_energy[-1] == 0.0 else test_energy[-1]
        new_data[indx] = test_energy[-1]
        
        # print(len(test_bond)," - ",test_bond)
        # print(basis_distance(test_bond, L))
        dists, counts = np.unique(hf.basis_distance(test_bond, L),return_counts=True)
        counts = (counts/np.sum(counts))*100
        print("- - ", dists)
        print("- - ", counts)
        dicd = dict(zip(dists,counts))
        for n in dists:
            COUNTS[indx, n] = dicd[n]

        plt.plot(test_energy, marker='o', label=f"M = {bond}")
        plt.yscale('log')
        plt.legend(loc='best', fontsize='small', markerscale=0.9)
        plt.title(f"L = {L}, V={physical[0]} & |m|={step}")
        # plt.show()
        # plt.savefig(f"Superposition_run/output/mid_plots/_{indx}_mid_energyplot_new.pdf", bbox_inches = 'tight')
        if indx == len(x_data)-1:
            plt.savefig(f"Superposition_run/output/mid_plots/_mid_energyplot_J{job_number}.pdf", bbox_inches = 'tight')


        t_i = tt.time()
        Es, _ = np.linalg.eigh(super_ham[:bond,:bond])
        print(f"- - Eigenvalue Time: ", tt.time() - t_i,"(s) for ", bond,"",bond/MAXDIM)
        old_data[indx] = (Es[0] - ed_energy[L])/L


    print(f"- Full Run {array_number} Time: ", tt.time() - T_i,"(s)")

    print(COUNTS)
    # print(new_data)
    # print(old_data)

    fig, ax = plt.subplots(1,1, figsize=(8, 5), #sharex=True, 
                # gridspec_kw=dict( hspace=0.0,
                #     ),
                subplot_kw=dict( 
                    # yscale = 'log', xscale ='linear',
                    # title = f"comparing old truncation vs. new method for L={L}, V={physical[0]} & |m|={step}",               
                    # ylabel = r'$|\frac{E-E_{*}}{L}|$', xlabel = r'$|M|$',
                    title = f"percentage of distances in new method for L={L}, V={physical[0]} & |m|={step}",               
                    ylabel = f'percentage', xlabel = r'$|M|$',
                    # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                    xticks= x_data, xticklabels=[str(x) for x in x_data],
                    # yticks=[10**(-s) for s in range(2,17,2)],    
                    ),
                )

    # ax.plot( x_data, new_data, label="new", ls='--', linewidth=0.7, marker='o')
    # ax.plot( x_data, old_data, label="old", ls='--', linewidth=0.7, marker='d')
    # ax.legend(loc='best')
    # ax.grid(which='major', axis='y', linestyle=':')
    # fig.savefig(f"Superposition_run/output/_test_Comparing_new_vs_old_V_{physical[0]}_L{L}_J{job_number}.pdf", bbox_inches = 'tight')

    for ii in range(L//2+1):
        bottom = np.sum(COUNTS[:, 0:ii], axis=1)
        erzeugung = ax.bar(x_data, COUNTS[:, ii], bottom=bottom, width=4.0, label=f"{DSTNS[ii]}")
    # ax.bar_label(erzeugung)#, padding=3)
    ax.grid(which='major', axis='y', linestyle=':')
    ax.set_xticks(x_data)
    ax.legend(loc='best')
    # ax.set_xticklabels(x)
    
    fig.savefig(f"Superposition_run/output/_test_distances_new_method_V_{physical[0]}_L{L}_J{job_number}.pdf", bbox_inches = 'tight')


if False:#####################################################################################################################################################    

    x_data = np.arange(20, 150, 20)
    new_data = [] #np.zeros((5,), dtype=np.float64)
    bond = 100

    fig, ax = plt.subplots(1,1, figsize=(8, 5), #sharex=True, 
                subplot_kw=dict( 
                    yscale = 'log', xscale ='linear',
                    title = f"comparing different step-sizes for L={L}, V={physical[0]} & |M|={bond}",               
                    ylabel = r'$|\frac{E-E_{ed}}{L}|$', xlabel = r'$\#\: iterations$',
                    # ylim = (1.e-17, 1e-2), #xlim = (1.e-12,100),   
                    xticks= [x for x in range(0, int((MAXDIM-bond)/x_data[0]), 10)], xticklabels=[str(x) for x in range(0, int((MAXDIM-bond)/x_data[0]), 10)],
                    # yticks=[10**(-s) for s in range(2,17,2)],    
                    ),
                )


    T_i = tt.time()
    for indx, step in enumerate(x_data):
        t0 = tt.time()
        print(f"- - Step is: {step}")
        # step = 10

        test_energy, test_bond, test_amps = hf.new_hf_optimization(physical, L, bond, size_step = step, PBC=False, max_iters = 250)

        test_energy = (np.array(test_energy) - ed_energy[L])/L
        
        # test_energy[-1] = 1.e-17 if test_energy[-1] == 0.0 else test_energy[-1]
        new_data.append(test_energy) #.indx] = test_energy[-1]
        
        
        print("- - Bond len is: ",len(test_bond))
        dists = np.unique(hf.basis_distance(test_bond, L),return_counts=True)
        print("- - ", dists)
        print("- - Time for bond:", tt.time()-t0)

        ax.plot(test_energy, marker='o', label=f"m={step}")

        # plt.legend(loc='best')
        # plt.title(f"M = {bond} & m={step}", y=0.9001)
        # plt.show()
        # if indx == 10:
        #    plt.savefig(f"Superposition_run/output/mid_plots/_{indx}_mid_energyplot_new.pdf", bbox_inches = 'tight')


    print(f"- Full Run {array_number} Time: ", tt.time() - T_i,"(s)")



    # ax.plot( x_data, new_data, label="new", ls='--', linewidth=0.8, marker='o')
    # ax.plot( x_data, old_data, label="old", ls='--', linewidth=0.8, marker='d')
    ax.grid(which='major', axis='y', linestyle=':')
    ax.legend(loc='best')
    fig.savefig(f"Superposition_run/output/_test_changing_step_sizes_new_method_V_{physical[0]}_L{L}_J{job_number}.pdf", bbox_inches = 'tight')

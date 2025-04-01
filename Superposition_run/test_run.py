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



print(f" bash inputs: ", sys.argv)
print("")

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
amp_thrsh = 1.e-5
ed_energy = {8:-4.670424529225025, 10:-5.908793067783503, 12:-7.14882332720711}
print(ed_energy[L])

x_data = np.arange(20, 240, 20)
new_data = np.zeros((,), dtype=np.float64)
old_data = np.zeros()

for bond in x_data:
    t0 = tt.time()
    print(f"Bond is: {bond}")
    # bond = 30
    step = 10

    test_energy, test_bond, test_amps = hf.new_hf_optimization(physical, L, bond, size_step = step, PBC=False, max_iters = 200)

    test_energy = (np.array(test_energy) - ed_energy[L])/L
    
    test_energy[-1] = 1.e-17 if test_energy[-1] == 0.0 else test_energy[-1]
    
    data.append(test_energy[-1])
    
    print("  - Bond len is: ",len(test_bond))
    # print(len(test_bond)," - ",test_bond)
    # print(basis_distance(test_bond, L))
    dists = np.unique(basis_distance(test_bond, L),return_counts=True)
    # print(zip(dists))
    print("  - Time for bond:", tt.time()-t0)

    pl.plot(test_energy)
    pl.yscale('log')
    # pl.xscale('log')
    pl.title(f"M = {bond}", y=0.9001)
    pl.show()
    
    t_i = tt.time()
    super_ham, super_basis = hf.hart_fock_superposition( [Vs, 1.0], ll, max_iters=maxsteps, PBC=False, basis_len = maxdim, start_point=1.e-4)
    

    
pl.plot( range(10,220,40), data)
# pl.ylim([1.e-24, 1.e-3])
pl.yscale('log')
pl.show()
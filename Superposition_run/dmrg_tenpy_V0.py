import sys
import time as tt
import numpy as np # type: ignore
import scipy as sp # type: ignore
import matplotlib.pyplot as plt # type: ignore

import tenpy
# from tenpy.networks.site import SpinHalfSite
from tenpy.models.spins import SpinModel 
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

import logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.disable(logging.INFO)

saving = True

sys.path.append('/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/')

if len(sys.argv) == 3:
    job_number = int(sys.argv[1])
    array_number = int(sys.argv[2])
if len(sys.argv) == 2:
    job_number = int(sys.argv[1])
    array_number = 0



print(f" ***** job {job_number} started ***** ")
print("")

error_in_energy = 1.e-18
max_bond = 1000

sizes = [40, 50] #[20, 22, 24, 26, 28, 30]
VS = [0.1, 0.2, 0.3] #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    
input = []
for siz in sizes:
    for vs in VS:
        input.append([siz, vs])
            
ll = input[array_number][0]
Vs = input[array_number][1]
    
print(f"DMRG ground-state for L={ll} and V={Vs}")
print("")
    
    
t_i = tt.time()

h_boundary = Vs/4. 
model_params = {
    'S':0.5,
    'Jx': 2*1.0, 'Jy': 2*1.0, 'Jz': Vs/2.,
    'hz': -Vs,
    'L': ll,
    'conserve':'Sz',
    'bc_x': 'open',
    'bc_MPS': 'finite',
}

# site = SpinHalfSite()

# Create the spin model
model = SpinModel(model_params)
# Add boundary terms
model.add_onsite_term(h_boundary, 0, "Sz") #left boundary
model.add_onsite_term(h_boundary, ll-1, "Sz") #right boundary
model.init_H_from_terms()

psi = MPS.from_lat_product_state(model.lat, [['up'],['down']])

dmrg_params = {
    'mixer': True,
    'max_E_err': error_in_energy,
    'trunc_params': {
        'chi_max': max_bond,
        'svd_min': error_in_energy,
    },
    # 'verbose': True,
    'combine': True,
}

results = dmrg.run(psi, model, dmrg_params)

DMRG_GS = results['E'] + Vs*(ll-1)/8.

if saving:
    arcivo = open(f'Superposition_run/raw_data/Ground_State_Energy/EDGS_{Vs}_{ll:02}.npy', 'wb')
    np.save(arcivo, DMRG_GS)
    arcivo.close()
    print(f"- - data file {array_number} saved")
    print("")
    
print(f"- dmrg ground state Run {array_number} Time: ", tt.time() - t_i,"(s)")
print("")
    
print(f"- ground state energy for L={ll} and V={Vs} is:", DMRG_GS)
print("")
    
MAX_BOND = results['sweep_statistics']['max_chi'][-1]
# arcivo = open(f'Superposition/raw_data/DMRG_BOND_{Vs}_{ll:02}.npy', 'wb')
# np.save(arcivo, MAX_BOND)
# arcivo.close()
# print(f"- - data file {array_number} saved")
# print("")

print(f"- max bond for L={ll} and V={Vs} is:", MAX_BOND)
print("")

# print("")

# print("DMRG results:", results)
# print("")
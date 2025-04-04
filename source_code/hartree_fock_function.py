import numpy as np  # type: ignore
import scipy as sp # type: ignore

####################################################################################################################################################################################################################
########################################################################################################## 1D functions ##########################################################################################################
####################################################################################################################################################################################################################

def HF_hamil(physical, cmat, L, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True    
    V, J = physical[:2]
    
    H = np.zeros((L,L))
    
    if PBC == True:
        for i in range(L):
            H[i,i] = +V/2*(cmat[np.mod(i+1,L),np.mod(i+1,L)] + cmat[np.mod(i-1,L),np.mod(i-1,L)])
            H[i,np.mod(i+1,L)] = -J - V/2*cmat[np.mod(i+1,L),i]
            H[np.mod(i+1,L),i] = -J - V/2*cmat[i,np.mod(i+1,L)]
    
    if PBC == False:
        for i in range(L - 1):
            H[i,np.mod(i+1,L)] = -J - V/2*cmat[np.mod(i+1,L),i]
            H[np.mod(i+1,L),i] = -J - V/2*cmat[i,np.mod(i+1,L)]
        
        for i in range(1, L - 1):
            H[i,i] = +V/2*(cmat[np.mod(i+1,L),np.mod(i+1,L)] + cmat[np.mod(i-1,L),np.mod(i-1,L)])
        
        H[0,0] = +V/2*cmat[1,1] 
        H[L-1,L-1] = +V/2*cmat[L-2,L-2]
    
    return(H)


def fullEnergy_HF(input_C, physical, L, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True
    
    V, J = physical[:2]
    E = 0
    for j in range(L-int(not PBC)):
        E += - J*(input_C[j,np.mod(j+1,L)] + input_C[np.mod(j+1,L),j])
        E += + V/2*(input_C[j,j]*input_C[np.mod(j+1,L),np.mod(j+1,L)] - input_C[j,np.mod(j+1,L)]*input_C[np.mod(j+1,L),j])

    return(E)


def hart_fock_optimization(physical,L, **kws):
    max_iters = kws['max_iters'] if 'max_iters' in kws.keys() else 400
    startp = kws['start_point'] if 'start_point' in kws.keys() else 0.0157
    
    c_mat = np.diag([1.,0.]*(L//2) + startp * np.random.randn(L))
    
    last_E = np.empty((L))
    output_energy = 0.0
    for _ in range(max_iters):
        H = HF_hamil(physical, c_mat, L, **kws)
        last_E, U = np.linalg.eigh(H)
        c_mat = np.dot(U[:,:L//2],np.conj(U[:,:L//2].T))
        output_energy = fullEnergy_HF(c_mat, physical, L, **kws)
    
    return(output_energy, np.diag(c_mat)) #return(last_E, np.diag(c_mat))
        
        
########################################################################################################## Superposition functions

def energy_func(n, mode_energis):
    L = len(mode_energis)
    K_list = np.array([int(i) for i in np.binary_repr(n, width=L)])
    if len(K_list) != L:
        print("Error - -")
        return( ) 
    return( np.array(K_list) @ np.array(mode_energis) )


def basis_set(L):
    N = L//2
    basis_N = []
    for n in range(2**(L//2)-1,2**L - 2**(L//2) +1): # range(2**L):
        bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
        if sum(bin_state) == N:
            basis_N.append(n)
    return basis_N


def C_term(U_mat, kk, qq, L, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True
    
    nn = kk ^ qq
    if nn == 0:
        indxs = np.nonzero([int(i) for i in np.binary_repr(kk, width=L)])[0]
        out_mat = np.dot(U_mat[:,indxs], np.conj(U_mat[:,indxs].T))
        out_val = np.trace(out_mat, offset=1) + np.trace(out_mat, offset=-1) + int(PBC)*np.trace(out_mat, offset=L-1) + int(PBC)*np.trace(out_mat, offset=-L+1)
        return out_val 
    else:        
        k_set = np.nonzero([int(i) for i in np.binary_repr(kk, width=L)])[0]
        q_set = np.nonzero([int(i) for i in np.binary_repr(qq, width=L)])[0]
        k,q = np.setdiff1d(k_set,q_set), np.setdiff1d(q_set,k_set)
            
        if len(k)==len(q) and len(q)==1:
            sign_k = k_set[k_set>k].size
            sign_q = q_set[q_set>q].size
            out_sign = ((-1)**(sign_k))*((-1)**(sign_q))
                
            out_mat = np.dot(U_mat[:,k], np.conj(U_mat[:,q]).T)
            out_val = np.trace(out_mat, offset=1) + np.trace(out_mat, offset=-1) + int(PBC)*np.trace(out_mat, offset=L-1) + int(PBC)*np.trace(out_mat, offset=-L+1)
            
            return out_sign * out_val
        
        return 0.0
        

def V_term(U_mat, kk, qq, L, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True

    nn = kk ^ qq
    if nn == 0:
        indxs = np.nonzero([int(i) for i in np.binary_repr(kk, width=L)])[0]
        out_mat = np.dot(U_mat[:,indxs], np.conj(U_mat[:,indxs].T))
        out_val = 0.0
        for j in range(L-int(not PBC)):
            out_val += out_mat[j,j]*out_mat[np.mod(j+1,L),np.mod(j+1,L)] - out_mat[j,np.mod(j+1,L)]*out_mat[np.mod(j+1,L),j]
            
        return out_val 
    
    else:        
        k_set = np.nonzero([int(i) for i in np.binary_repr(kk, width=L)])[0]
        q_set = np.nonzero([int(i) for i in np.binary_repr(qq, width=L)])[0]
        dif_k,dif_q = np.setdiff1d(k_set,q_set), np.setdiff1d(q_set,k_set)
                
        if len(dif_k) == len(dif_q) and len(dif_k) == 1:
            equl = np.intersect1d(k_set,q_set)
            inner_c = np.dot(U_mat[:,equl], np.conj(U_mat[:,equl]).T)
            
            sign_k = k_set[k_set > dif_k].size
            sign_q = q_set[q_set > dif_q].size
            out_sign = ((-1)**(sign_k))*((-1)**(sign_q))
            
            out_val = 0.0
            for j in range(L-int(not PBC)):
                out_val += -1*( U_mat[np.mod(j+1,L),dif_k] * inner_c[np.mod(j  ,L),np.mod(j+1,L)] * np.conj(U_mat[np.mod(j  ,L),dif_q]) )
                out_val += -1*( U_mat[np.mod(j  ,L),dif_k] * inner_c[np.mod(j+1,L),np.mod(j  ,L)] * np.conj(U_mat[np.mod(j+1,L),dif_q]) )
                out_val += +1*( U_mat[np.mod(j  ,L),dif_k] * inner_c[np.mod(j+1,L),np.mod(j+1,L)] * np.conj(U_mat[np.mod(j  ,L),dif_q]) )
                out_val += +1*( U_mat[np.mod(j+1,L),dif_k] * inner_c[np.mod(j  ,L),np.mod(j  ,L)] * np.conj(U_mat[np.mod(j+1,L),dif_q]) )            
        
            return out_sign * out_val[0]
        
        if len(dif_k) == len(dif_q) and len(dif_k) == 2:
            sign_k = k_set[k_set > dif_k[0]]
            sign_k = sign_k[sign_k < dif_k[1]].size
            
            sign_q = q_set[q_set > dif_q[0]]
            sign_q = sign_q[sign_q < dif_q[1]].size
            
            out_sign = ((-1)**(sign_k))*((-1)**(sign_q))
            
            out_val = 0.0
            for j in range(L-int(not PBC)):
                out_val += +1*( U_mat[np.mod(j,L),dif_k[0]] * U_mat[np.mod(j+1,L),dif_k[1]] * np.conj(U_mat[np.mod(j ,L),dif_q[0]]) * np.conj(U_mat[np.mod(j+1,L),dif_q[1]]) )
                out_val += -1*( U_mat[np.mod(j,L),dif_k[0]] * U_mat[np.mod(j+1,L),dif_k[1]] * np.conj(U_mat[np.mod(j ,L),dif_q[1]]) * np.conj(U_mat[np.mod(j+1,L),dif_q[0]]) )
                out_val += -1*( U_mat[np.mod(j,L),dif_k[1]] * U_mat[np.mod(j+1,L),dif_k[0]] * np.conj(U_mat[np.mod(j ,L),dif_q[0]]) * np.conj(U_mat[np.mod(j+1,L),dif_q[1]]) )
                out_val += +1*( U_mat[np.mod(j,L),dif_k[1]] * U_mat[np.mod(j+1,L),dif_k[0]] * np.conj(U_mat[np.mod(j ,L),dif_q[1]]) * np.conj(U_mat[np.mod(j+1,L),dif_q[0]]) )
                
            return out_sign * out_val
        
        return 0.0


def ordered_basis(L, mode_energis, **kwargs):
    basis_len = kwargs['basis_len'] if 'basis_len' in kwargs.keys() else None
    
    N = L//2
    basis_N = []
    order_set = []
    for n in range(2**(L//2)-1,2**L - 2**(L//2) +1): #range(2**L):
        bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
        if sum(bin_state) == N:
            basis_N.append(n)
            order_set.append( np.array(bin_state) @ np.array(mode_energis) )
    order = np.argsort(order_set) 
    output = np.array(basis_N)[order.astype(int)]
    return(output[0:basis_len])
        

def ordered_basis_interaction(L, physical, **kwargs):
    basis_len = kwargs['basis_len'] if 'basis_len' in kwargs.keys() else None
    V, J = physical[:2]
    
    N = L//2
    basis_N = []
    order_set = []
    for n in range(2**(L//2)-1,2**L - 2**(L//2) +1): #range(2**L):
        bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
        rolo_state = np.roll(bin_state, -1)
        if sum(bin_state) == N:
            basis_N.append(n)
            order_set.append( 0.5 * V * rolo_state @ bin_state  )
    order = np.argsort(order_set) 
    output = np.array(basis_N)[order.astype(int)]
    return(output[0:basis_len])



def ordered_full_energy(L, U_mat, physical, **kwargs):
    basis_len = kwargs['basis_len'] if 'basis_len' in kwargs.keys() else None
    
    N = L//2
    basis_N = []
    order_set = []
    for n in range(2**(L//2)-1,2**L - 2**(L//2) +1): #range(2**L):
        bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
        if sum(bin_state) == N:
            basis_N.append(n)
            colum_indexs = np.where(bin_state == 1)[0]
            c_mat = np.dot(U_mat[:,colum_indexs],np.conj(U_mat[:,colum_indexs].T))
            order_set.append( fullEnergy_HF(c_mat, physical, L, **kwargs)  )    
    
    order = np.argsort(order_set) 
    output = np.array(basis_N)[order.astype(int)]
    return(output[0:basis_len], order)
        

def basis_distance(basis_set, L, **kwargs):
    distance_set = []
    for n in basis_set:
        n0 = basis_set[0]
        m = np.bitwise_and(n0, n)
        bin_state = np.array([int(i) for i in np.binary_repr(m, width=L)])
        distance = int(L/2 - np.sum(bin_state))
        distance_set.append(distance)
        
    return(distance_set)
                

def based_ham(physical, L, energy_list, u_mat, **kwargs):
    # basic_order = kwargs['basic_order'] if 'basic_order' in kwargs.keys() else True
    
    V, J = physical[:2]
    
    # if basic_order:
    #     base_set = ordered_basis(L, energy_list, **kwargs)
    # else:
    #     base_set = ordered_full_energy(L, u_mat, physical, **kwargs)
    base_set = ordered_basis(L, energy_list, **kwargs)
    el = len(base_set)
    
    new_ham = np.zeros((el,el))    
    for indx_k, K in enumerate(base_set):
        for indx_q, Q in enumerate(base_set):
            new_ham[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs)
            new_ham[indx_k,indx_q] += -J*C_term(u_mat, K, Q, L, **kwargs)
            
    return(new_ham, base_set)        
        
        

def hart_fock_superposition(physical, L, **kws):
    max_iters = kws['max_iters'] if 'max_iters' in kws.keys() else 200
    startp = kws['start_point'] if 'start_point' in kws.keys() else 0.0157
    
    c_mat = np.diag([1.,0.]*(L//2) + startp * np.random.randn(L))
    
    last_E = np.empty((L))
    last_U = np.empty((L,L))
    # HF_energy = 0.0
    for _ in range(max_iters):
        H = HF_hamil(physical, c_mat, L, **kws)
        last_E, last_U = np.linalg.eigh(H)
        c_mat = np.dot(last_U[:,:L//2],np.conj(last_U[:,:L//2].T))
        # HF_energy = fullEnergy_HF(c_mat, physical, L, **kws)
    
    reorder_basis, new_order = ordered_full_energy(L, last_U, physical, **kws)
    # old_order = ordered_basis(L, last_E, **kws)
    new_ham, super_basis = based_ham( physical, L, last_E, last_U, **kws)
    
    return(new_ham, super_basis) #new_order)


######################################################################################## new algorithm for superposition optimization

def new_based_ham(physical, L, bond_list, u_mat, **kwargs):
    
    V, J = physical[:2]
    el = len(bond_list)

    new_ham = np.zeros((el,el))    
    for indx_k, K in enumerate(bond_list):
        for indx_q, Q in enumerate(bond_list):
            new_ham[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs)
            new_ham[indx_k,indx_q] += -J*C_term(u_mat, K, Q, L, **kwargs)
            
    return(new_ham, bond_list)        
        

def new_hart_fock_optimization(physical, L, **kws):
    max_iters = kws['max_iters'] if 'max_iters' in kws.keys() else 100
    startp = kws['start_point'] if 'start_point' in kws.keys() else 0.0157
    
    c_mat = np.diag([1.,0.]*(L//2) + startp * np.random.randn(L))
    
    last_E = np.empty((L))
    last_U = np.empty((L,L))
    # output_energy = []
    for _ in range(max_iters):
        H = HF_hamil(physical, c_mat, L, **kws)
        last_E, last_U = np.linalg.eigh(H)
        c_mat = np.dot(last_U[:,:L//2],np.conj(last_U[:,:L//2].T))
        # output_energy.append( fullEnergy_HF(c_mat, physical, L, **kws) )

    return(last_E, last_U)


def new_hf_optimization(physical, L, bond_size, **kws):
    max_steps = kws['max_steps'] if 'max_steps' in kws.keys() else 200
    size_step = kws['size_step'] if 'size_step' in kws.keys() else 5
    # amp_trshld = kws['amp_trshld'] if 'amp_trshld' in kws.keys() else 1.e-12
    # output_amps = kws['return_amps'] if 'return_amps' in kws.keys() else False
    
    hf_E, hf_U = new_hart_fock_optimization(physical, L, **kws)
    bond_list = ordered_basis(L, hf_E, **kws)
    # basis = bond_list[:bond_size + size_step]
    basis = bond_list[:bond_size]

    output_amps = np.zeros((bond_size,), dtype=np.float64)
    output_energy = []
    step = 0
    
    while step < max_steps and len(bond_list) > 0:
        
        # print(f"*** step {step}")
        ham, _ = new_based_ham(physical, L, basis, hf_U, **kws)
        _, U = np.linalg.eigh(ham)
        
        amps = np.abs(U[:,0])
        order = np.argsort(amps)[::-1] 
        new_basis = np.array(basis)[order.astype(int)]

        bond_list = np.setdiff1d(bond_list, new_basis, assume_unique=True)
        basis = np.concatenate((new_basis[:bond_size], bond_list[:size_step]), axis=None)

        ham, _ = new_based_ham(physical, L, basis[:bond_size], hf_U, **kws)
        
        # E = np.linalg.eigvalsh(ham)
        
        E, nU = np.linalg.eigh(ham)
        output_amps = nU[:,0]
        # N_amps = np.abs(nU[:,0])
        # res_amps = np.floor(np.log10(N_amps))
        
        output_energy.append( E[0] )
        
        step += 1
        
    return(output_energy, basis[:bond_size], output_amps)
        
 


#~~~~~~~~~~~~~~~~~~~~~~~~~~ below can be removed X
         
def shadow_ham(physical, L, energy_list, u_mat, **kwargs):
    extra_check = kwargs['extra_check'] if 'extra_check' in kwargs.keys() else False
    treshold = kwargs['treshod'] if 'treshold' in kwargs.keys() else 1.e-13
    # basic_order = kwargs['basic_order'] if 'basic_order' in kwargs.keys() else True
    
    V, J = physical[:2]
    
    E_max = np.sum( np.abs( energy_list[[0,1,L-2,L-1]] )) 
    
    base_set = ordered_basis(L, energy_list, **kwargs)
    el = len(base_set)

    shadow = np.zeros((el,el))    
    for indx_k, K in enumerate(base_set):
        for indx_q, Q in enumerate(base_set):
            K_state = np.array([int(i) for i in np.binary_repr(K, width=L)])
            E_k = np.array(K_state) @ np.array(energy_list)
            
            Q_state = np.array([int(i) for i in np.binary_repr(Q, width=L)])
            E_q = np.array(Q_state) @ np.array(energy_list)
            
            delta_E = np.abs( E_k - E_q)
            mat_element = 0.5*V*V_term(u_mat, K, Q, L, **kwargs) + -J*C_term(u_mat, K, Q, L, **kwargs)
            # shadow[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs)
            # shadow[indx_k,indx_q] += -J*C_term(u_mat, K, Q, L, **kwargs)
            if delta_E >= E_max :
            
                if not extra_check:
                    shadow[indx_k,indx_q] += 1
                    
                if extra_check and np.abs(mat_element) > treshold:
                    # shadow[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs) + -J*C_term(u_mat, K, Q, L, **kwargs)
                    shadow[indx_k,indx_q] += 1
            
    return(shadow)        


def hart_fock_shadowing(physical, L, **kws):
    max_iters = kws['max_iters'] if 'max_iters' in kws.keys() else 200
    startp = kws['start_point'] if 'start_point' in kws.keys() else 0.15
    
    c_mat = np.diag([1.,0.]*(L//2) + startp * np.random.randn(L))
    
    last_E = np.empty((L))
    last_U = np.empty((L,L))
    for _ in range(max_iters):
        H = HF_hamil(physical, c_mat, L, **kws)
        last_E, last_U = np.linalg.eigh(H)
        c_mat = np.dot(last_U[:,:L//2],np.conj(last_U[:,:L//2].T))

    new_ham = shadow_ham( physical, L, last_E, last_U, **kws)
    
    # return(new_ham, new_order)
    return(new_ham)


# def free_fermion_superposition(physical,L, **kws):    
#     # last_E = np.empty((L))
#     # last_U = np.empty((L,L))

#     H = FF_hamil(physical, L, **kws)
#     last_E, last_U = np.linalg.eigh(H)
#     c_mat = np.dot(last_U[:,:L//2],np.conj(last_U[:,:L//2].T))
#     FF_energy = fullEnergy_HF(c_mat, physical, L, **kws)
    
#     new_ham = based_ham( physical, L, last_E, last_U, **kws)
#     # new_E, new_U = np.linalg.eigh(new_ham)
    
#     return(new_ham)

####################################################################################################################################################################################################################
########################################################################################################## 2D functions ##########################################################################################################
####################################################################################################################################################################################################################


def MF_ham_2d(physical, cmat, dims, **kws):
    
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True
    
    Lx, Ly = dims
    V, J = physical[:2]

    # this is for off-diagonal terms of hamiltonian or Fock contribution
    H_mat = np.zeros((Lx*Ly, Lx*Ly))
    for x in range(Lx):
        for y in range(Lx):
            nl = x+Lx*y
            
    # this is for diagonal terms of hamiltonian or Hartree contribution
    for x in range(Lx):
        for y in range(Lx):
            org = x+Lx*y
            l,r = np.mod(x+1, Lx)+Lx*np.mod(y, Ly), np.mod(x-1, Lx)+Lx*np.mod(y, Ly)
            u,d = np.mod(x, Lx)+Lx*np.mod(y-1, Ly), np.mod(x, Lx)+Lx*np.mod(y+1, Ly)
            H_mat[org,org] = + V/2*( cmat[l,l] + cmat[r,r] + cmat[u,u] + cmat[d,d] )
        
    return(H_mat)




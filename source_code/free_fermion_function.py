import numpy as np # type: ignore

########################################################################################################################################################################################
##################################################################################### 1D functiond #####################################################################################
########################################################################################################################################################################################

def Sym(input, L):
    """_summary_: Creates a symmetric matrix from a vectro of parameters with length L(L+1)/2.
    Returns:
        _numpy.ndarray_: symetric matrix with shape of (L, L)
    """
    if input.size != L * (L+1)//2:
        input = np.resize(input, L*(L+1)//2)
    
    out = np.zeros((L,L))
    indxs = np.triu_indices_from(out)
    out[ indxs[1], indxs[0] ]= input
    out[indxs] = input
        
    return  out


def Sym_ZeroDiag(input, L):
    """
    Returns:
        _nump.ndarray_: L by L symmetric numpy array with zero diagonal elements
    """    
    if input.size != L * (L-1)//2:
        input = np.resize(input, L*(L-1)//2)
    
    out = np.zeros((L,L))
    indxs = np.triu_indices_from(out, k=1)
    out[ indxs[1], indxs[0] ]= input
    out[indxs] = input
        
    return  out


def cmat(inputlist, L, filing=2):
    
    # if len(inputlist) != L*L:
    #     np.resize(inputlist, L*L)
    # Wmat = np.reshape(inputlist,(L,L))
    # H = (Wmat + Wmat.T)/2
    
    H = Sym(inputlist,L)
    # H = Sym_ZeroDiag(inputlist)
    
    E, U = np.linalg.eigh(H)
    C = np.dot(U[:,:L//filing],np.conj(U[:,:L//filing].T))
    
    return(C)


def fullEnergy_FF(input_C, physical, L, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True
    
    V, J = physical[:2]
    E = 0
    for j in range(L-int(not PBC)):
        E += - J*(input_C[j,np.mod(j+1,L)] + input_C[np.mod(j+1,L),j])
        E += + V/2*(input_C[j,j]*input_C[np.mod(j+1,L),np.mod(j+1,L)] - input_C[j,np.mod(j+1,L)]*input_C[np.mod(j+1,L),j])

    return(E)


def GetGradian_FF(input_array, physical, L, **kwargs):
    p_shift = kwargs['parameter_shift'] if 'parameter_shift' in kwargs.keys() else 0.001
    
    grad_array = np.zeros(input_array.shape)
    for i_th in np.arange(input_array.size):
        input_array[i_th] += p_shift
        C_mat = cmat(input_array, L)
        energi1 = fullEnergy_FF(C_mat, physical, L, **kwargs)
        
        input_array[i_th] += -2 * p_shift
        C_mat = cmat(input_array, L)
        energi2 = fullEnergy_FF(C_mat, physical, L, **kwargs)
        
        grad_array[i_th] += (energi1-energi2)/(2*p_shift)
        input_array[i_th] += p_shift
        
    return(grad_array)



def freefermion_optimization(physical, L, **kwargs):
    grad_rate = kwargs['grad_rate'] if 'grad_rate' in kwargs.keys() else 1.637
    max_step = kwargs['max_step'] if 'max_step' in kwargs.keys() else 200
    
    # params = np.random.rand( L * L )
    params = np.random.uniform(-1.0, +1.0, L * (L+1) // 2 )
    # es = []
    # ps = []
    es = 0.0
    ps = 0.0
    for n in np.arange(max_step):
        cm = cmat(params, L)
        # es.append(fullEnergy_FF(cm, Vs, L))
        # ps.append(np.diag(cm)[0])
        es = fullEnergy_FF(cm, physical, L, **kwargs)
        ps = abs(np.diag(cm)[0] - np.diag(cm)[1])
        gradian = GetGradian_FF(params, physical, L, **kwargs)
        params += -1 * grad_rate * gradian
    return(es, ps) #(es[-1], ps[-1])


########################################################################### Superposition functions

def FF_hamil(physical, L, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True

    V, J = physical[:2]
    
    H = np.zeros((L,L))
    for i in range(L-int(not PBC)):
        H[i,np.mod(i+1,L)] = -J 
        H[np.mod(i+1,L),i] = -J
    
    return(H)


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
    for n in range(2**L):
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
    for n in range(2**L):
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
            order_set.append( fullEnergy_FF(c_mat, physical, L, **kwargs)  )    
    
    order = np.argsort(order_set) 
    output = np.array(basis_N)[order.astype(int)]
    return(output[0:basis_len], order)
                

def based_ham(physical, L, energy_list, u_mat, **kwargs):
    
    V, J = physical[:2]
    base_set = ordered_basis(L, energy_list, **kwargs)
    el = len(base_set)

    new_ham = np.zeros((el,el))    
    for indx_k, K in enumerate(base_set):
        for indx_q, Q in enumerate(base_set):
            new_ham[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs)
            new_ham[indx_k,indx_q] += -J*C_term(u_mat, K, Q, L, **kwargs)
            
    return(new_ham)        
        

def free_fermion_superposition(physical, L, **kws):    
    # gs_output = kws['gs_output'] if 'gs_output' in kws.keys() else False

    H = FF_hamil(physical, L, **kws)
    last_E, last_U = np.linalg.eigh(H)
    c_mat = np.dot(last_U[:,:L//2],np.conj(last_U[:,:L//2].T))
    # full_energy = fullEnergy_FF(c_mat, physical, L, **kws)
    
    reorder_basis, new_order = ordered_full_energy(L, last_U, physical, **kws)
    
    new_ham = based_ham( physical, L, last_E, last_U, **kws)
    
    # if not gs_output:
    #     return(new_ham)
        
    # else:
    #     sup_E, sup_U = np.linalg.eigh(new_ham)
    #     return(full_energy, sup_E[0], sup_U[0])
    
    # return(new_ham)
    return(new_ham, new_order)



def shadow_ham(physical, L, energy_list, u_mat, **kwargs):
    extra_check = kwargs['extra_check'] if 'extra_check' in kwargs.keys() else False
    treshold = kwargs['treshod'] if 'treshold' in kwargs.keys() else 1.e-12
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
            # shadow[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs)
            # shadow[indx_k,indx_q] += -J*C_term(u_mat, K, Q, L, **kwargs)
            mat_element = 0.5*V*V_term(u_mat, K, Q, L, **kwargs) + -J*C_term(u_mat, K, Q, L, **kwargs)
            
            if delta_E >= E_max :
            
                if not extra_check:
                    shadow[indx_k,indx_q] += 1
                    
                if extra_check and mat_element > treshold:
                    # shadow[indx_k,indx_q] += 0.5*V*V_term(u_mat, K, Q, L, **kwargs) + -J*C_term(u_mat, K, Q, L, **kwargs)
                    shadow[indx_k,indx_q] += 1
                    
            # print("*")
            # print(K,",",K_state," -> ",E_k)
            # print(Q,",",Q_state," -> ",E_q)
            # print("                                                 ", delta_E,"><",E_max)
            # print("                                                 ", 0.5*V*V_term(u_mat, K, Q, L, **kwargs) - J*C_term(u_mat, K, Q, L, **kwargs))
    return(shadow)        

def free_fermion_shadowing(physical, L, **kws):
    max_iters = kws['max_iters'] if 'max_iters' in kws.keys() else 200
        
    last_E = np.empty((L))
    last_U = np.empty((L,L))
    for _ in range(max_iters):
        H = FF_hamil(physical, L, **kws)
        last_E, last_U = np.linalg.eigh(H)
    
    new_ham = shadow_ham( physical, L, last_E, last_U, **kws)
    
    # return(new_ham, new_order)
    return(new_ham)

########################################################################################################################################################################################
##################################################################################### 2D functiond #####################################################################################
########################################################################################################################################################################################

def Sym(input, L):
    """_summary_: Creates a symmetric matrix from a vectro of parameters with length L(L+1)/2.
    Returns:
        _numpy.ndarray_: symetric matrix with shape of (L, L)   
    """
    if input.size != L * (L+1)//2:
        input = np.resize(input, L*(L+1)//2)
    
    out = np.zeros((L,L))
    indxs = np.triu_indices_from(out)
    out[ indxs[1], indxs[0] ]= input
    out[indxs] = input
        
    return  out


def Sym_ZeroDiag(input, L):
    """
    Returns:
        _nump.ndarray_: L by L symmetric numpy array with zero diagonal elements
    """    
    if input.size != L * (L-1)//2:
        input = np.resize(input, L*(L-1)//2)
    
    out = np.zeros((L,L))
    indxs = np.triu_indices_from(out, k=1)
    out[ indxs[1], indxs[0] ]= input
    out[indxs] = input
        
    return  out


def cmat_2d(inputlist, *dims, **kw):
    
    filing = kw['filing'] if 'filing' in kw.keys() else 2
    func = kw['energy_func'] if 'energy_func' in kw.keys() else Sym
    
    Lx,Ly = dims[:2]
    L = Lx*Ly
    
    H = func(inputlist, L)
    
    E, U = np.linalg.eigh(H)
    C = np.dot(U[:,:L//filing],np.conj(U[:,:L//filing].T))
    
    return C

# def cmat_2d(inputlist, *dims, **kw):
#     filing = kw['filing'] if 'filing' in kw.keys() else 2

#     Lx,Ly = dims[:2]
#     L = Lx*Ly
    
#     if len(inputlist) != L*L:
#         np.resize(inputlist, L*L)
     
#     Wmat = np.reshape(inputlist,(L,L))
    
#     H = (Wmat + Wmat.T)/2
    
#     E, U = np.linalg.eigh(H)
#     C = np.dot(U[:,:L//filing],np.conj(U[:,:L//filing].T))
    
#     return C

def FullEnergy_PC_2d(C, *physical, **kw):
    
    PBC = kw['PBC'] if 'PBC' in kw.keys() else True
    
    Vs, Lx, Ly = physical[:3]
    
    ad_X = np.eye(Lx, k=1) + np.eye(Lx, k=-1)  + int(PBC)*np.eye(Lx, k=Lx-1)  + int(PBC)*np.eye(Lx, k=-Lx+1)
        
    if Ly==2:
        ad_Y = np.eye(Ly, k=1) + np.eye(Ly, k=-1)
    else:
        ad_Y = np.eye(Ly, k=1) + np.eye(Ly, k=-1)  + int(PBC)*np.eye(Ly, k=Ly-1)  + int(PBC)*np.eye(Ly, k=-Ly+1)

    ad_mat = np.kron(ad_Y, np.eye(Lx)) + np.kron(np.eye(Ly), ad_X)
    
    E = 0.0
    for xx in range(Lx*Ly):
        for yy in range(xx,Lx*Ly):
            E += - 1.0 *ad_mat[xx,yy] * (C[xx,yy] + C[yy,xx])
            E += - Vs/2*ad_mat[xx,yy] * (C[xx,xx]*C[yy,yy] - C[xx,yy]*C[yy,xx])

    return(E)


def GetGradian_PC_2d(input_array, *parameter, **kw):
    
    p_shift = kw['p_shift'] if 'p_shift' in kw.keys() else 0.001
    
    Vs, Lx, Ly = parameter[:3]
    
    grad_array = np.zeros(input_array.shape)
    for i_th in range(input_array.size):
        
        input_array[i_th] += p_shift
        C_mat = cmat_2d(input_array, Lx, Ly, **kw)
        energi1 = FullEnergy_PC_2d(C_mat, Vs, Lx, Ly, **kw)
        
        input_array[i_th] += -2 * p_shift
        C_mat = cmat_2d(input_array, Lx, Ly, **kw)
        energi2 = FullEnergy_PC_2d(C_mat, Vs, Lx, Ly, **kw)
        
        grad_array[i_th] += (energi1-energi2)/(2*p_shift)
        input_array[i_th] += p_shift
        
    return(grad_array)


def freefermion_optimization_2d(*physical, **kw):
    
    grad_rate = kw['grad_rate'] if 'grad_rate' in kw.keys() else 1.673
    max_step = kw['max_step'] if 'max_step' in kw.keys() else 200
    target_energy = kw['target_energy'] if 'target_energy' in kw.keys() else 0.0
    func = kw['energy_func'] if 'energy_func' in kw.keys() else Sym
    # energy_output = kw['energy_output'] if 'energy_output' in kw.keys() else True
    # particle_number = kw['particle_number'] if 'particle_number' in kw.keys() else 0
    
    Vs, Lx, Ly = physical[:3]

    if func.__qualname__ == 'Sym_ZeroDiag':
        params = np.random.uniform(-1., +1., Lx*Ly * (Lx*Ly-1) // 2 )
    else:
        params = np.random.uniform(-1., +1., Lx*Ly * (Lx*Ly+1) // 2 )
    
    es = 0.0 #[]
    ps = 0.0 #[]
    for _ in range(max_step):
        CMAT = cmat_2d(params, Lx, Ly, **kw)
        energia_optimizado = FullEnergy_PC_2d(CMAT, Vs, Lx, Ly, **kw)
        
        if target_energy != 0.0 and type(target_energy) == float:
            # es.append(abs( energia_optimizado - target_energy) )
            es = abs( energia_optimizado - target_energy)
        else:
            # es.append(energia_optimizado)
            es = energia_optimizado
        
        # ps.append(np.round(np.trace(CMAT) / Lx*Ly, 5))
        ps = abs( np.diag(CMAT)[0] - np.diag(CMAT)[1] ) # ps.append(np.diag(CMAT)[0])

        gradian = GetGradian_PC_2d(params, Vs, Lx, Ly, **kw)
        params += -1 * grad_rate * gradian
    
    return (es, ps) #(es[-1], ps[-1]) #, params


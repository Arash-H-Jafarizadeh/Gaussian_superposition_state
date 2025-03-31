import numpy as np

def R_mat(inputlist, L):
    
    if inputlist.size != L * L:
        inputlist=np.resize(inputlist, L*L)
     
    R_mat = (np.reshape(inputlist,(L,L)) - np.reshape(inputlist,(L,L)).T)/2
    norm = np.sqrt(np.linalg.det(np.eye(L) + np.dot(R_mat.conj().T, R_mat)))
    
    Q_mat = np.linalg.inv( np.eye(L,L) - np.dot(R_mat.conj(), R_mat))
    
    C = ( np.eye(L) - Q_mat) #/ norm
    F = (np.dot(R_mat.conj(), Q_mat.T)) #/ norm
    
    return C, F


def FullEnergy_GS(C, F, Vs, L):
    E=0
    for j in range(L):
        E += -(C[j,np.mod(j+1,L)] + C[np.mod(j+1,L),j])
        E += -Vs/2*(C[j,j]*C[np.mod(j+1,L),np.mod(j+1,L)] -F[j,np.mod(j+1,L)]*F[j,np.mod(j+1,L)] - C[j,np.mod(j+1,L)]*C[np.mod(j+1,L),j])

    return E


def GetGradian_GS(input_array, Vs, L, p_shift = 0.001):
    grad_array = np.zeros(input_array.shape)
    
    for i_th in np.arange(input_array.size):    
        input_array[i_th] += p_shift
        C_mat, F_mat = R_mat(input_array, L)
        energi1 = FullEnergy_GS(C_mat, F_mat, Vs, L)
        
        input_array[i_th] += -2 * p_shift
        C_mat, F_mat = R_mat(input_array, L)
        energi2 = FullEnergy_GS(C_mat, F_mat, Vs, L)
        
        grad_array[i_th] += (energi1-energi2)/(2 * p_shift)
        
    return(grad_array)



def genstate_optimization(Vs, L, grad_rate = 1.637, max_step = 200):
    params = np.random.rand( L * L )
    es = []
    ps = []
    for _ in np.arange(max_step):
        cmat,fmat = R_mat(params, L)
        es.append( FullEnergy_GS(cmat, fmat, Vs, L))
        ps.append( np.diag(cmat) )
        gradian = GetGradian_GS(params, Vs, L)
        params += -1 * grad_rate * gradian
    
    return(es[-1], ps[-1])

########################################################################################################################################################################################
##################################################################################### 2D functiond #####################################################################################
########################################################################################################################################################################################

def AntiSym(input, L):
    """_summary_: Creates an ant-symmetric matrix from an array of parameters.
    parameters:
        input: 1D array with length L(L-1)/2
        L: size of the out put matrix 
    Returns:
        _numpy.ndarray_: symetric matrix with shape of (L, L).   
    """
    if input.size != L * (L-1)//2:
        input = np.resize(input, L*(L-1)//2)
    
    out = np.zeros((L,L))
    indxs = np.triu_indices_from(out, k=1)
    out[ indxs[1], indxs[0] ]= input
    out[indxs] = -1*input
        
    return  out


def r_mat_2d(inputlist, *dims, **kw):

    Lx,Ly = dims[:2]
    L = Lx*Ly

    R_mat = AntiSym(inputlist, L)
    
    Q_mat = np.linalg.inv( np.eye(L,L) - np.dot(R_mat.conj(), R_mat))
    
    C = ( np.eye(L) - Q_mat) #/ norm
    F = (np.dot(R_mat.conj(), Q_mat.T)) #/ norm
    
    return C, F


def FullEnergy_GS_2d(C, F, *physical, **kw):
    
    PBC = kw['PBC'] if 'PBC' in kw.keys() else True
    
    Vs, Lx, Ly = physical[:3]
    
    ad_X = np.eye(Lx, k=1) + np.eye(Lx, k=-1)  + int(PBC)*np.eye(Lx, k=Lx-1)  + int(PBC)*np.eye(Lx, k=-Lx+1)    
    if Ly==2:
        ad_Y = np.eye(Ly, k=1) + np.eye(Ly, k=-1)
    else:
        ad_Y = np.eye(Ly, k=1) + np.eye(Ly, k=-1)  + int(PBC)*np.eye(Ly, k=Ly-1)  + int(PBC)*np.eye(Ly, k=-Ly+1)

    ad_mat = np.kron(ad_Y, np.eye(Lx)) + np.kron(np.eye(Ly), ad_X)
    
    E = 0
    for xx in range(Lx*Ly):
        for yy in range(xx, Lx*Ly):
            E += - 1.0*ad_mat[xx,yy] * (C[xx,yy] + C[yy,xx])
            E += - Vs/2*ad_mat[xx,yy] * (C[xx,xx]*C[yy,yy] - C[xx,yy]*C[yy,xx] - F[xx,yy]*F[yy,xx].conj() )

    return E


def GetGradian_GS_2d(input_array, *parameter, **kw):
    
    p_shift = kw['p_shift'] if 'p_shift' in kw.keys() else 0.001
    
    Vs, Lx, Ly = parameter[:3]
    
    grad_array = np.zeros(input_array.shape)
    for i_th in np.arange(input_array.size):    
        input_array[i_th] += p_shift
        C_mat, F_mat = r_mat_2d(input_array, Lx, Ly, **kw)
        energi1 = FullEnergy_GS_2d(C_mat, F_mat, Vs, Lx, Ly, **kw)
        
        input_array[i_th] += -2 * p_shift
        C_mat, F_mat = r_mat_2d(input_array, Lx, Ly, **kw)
        energi2 = FullEnergy_GS_2d(C_mat, F_mat, Vs, Lx, Ly, **kw)
        
        grad_array[i_th] += (energi1-energi2)/(2 * p_shift)
        input_array[i_th] += p_shift
        
    return(grad_array)


def genstate_optimization_2d(*physical, **kw):
    """_summary_
    parameters(*):
        physical: V, Lx, Ly (given in order) 
    Keywords(**):
        grad_rate: gradient descent rate, a.k.a update rate.
        max_step: maximum optimization steps.
        target_energy: targer energy to reach.
        energy_output: type of energy output. 'True' for solely optimized energy or 'Falsw' for difference between target energy and optimized energy. 
    Returns:
        _list_: energy, particle_count 
    """
    grad_rate = kw['grad_rate'] if 'grad_rate' in kw.keys() else 0.343
    max_step = kw['max_step'] if 'max_step' in kw.keys() else 200
    target_energy = kw['target_energy'] if 'target_energy' in kw.keys() else 0.0
    # energy_output = kw['energy_output'] if 'energy_output' in kw.keys() else True
    
    Vs, Lx, Ly = physical[:3]
    
    # params = np.random.rand( Lx*Ly * (Lx*Ly - 1) // 2 )
    # params = np.random.randn( Lx*Ly * (Lx*Ly - 1) // 2 )
    params = np.random.uniform(-1, 1, Lx*Ly * (Lx*Ly - 1) // 2 )

    Vs, Lx, Ly = physical[:3]
    
    es = 0.0#[]
    ps = 0.0#[]
    for _ in np.arange(max_step):
        cmat, fmat = r_mat_2d(params, Lx, Ly, **kw)
        optimized_energy = FullEnergy_GS_2d(cmat, fmat, Vs, Lx, Ly, **kw)
        
        if target_energy != 0.0 and type(target_energy) == float:
            # es.append(abs( energia_optimizado - target_energy) )
            es = abs(optimized_energy - target_energy)
        else:
            # es.append(energia_optimizado)
            es = optimized_energy
        
        # ps.append(np.trace(cmat, dtype=np.float16) / L)
        ps = np.diag(cmat)[0] #ps.append(np.diag(cmat)[0])
        
        gradian = GetGradian_GS_2d(params, Vs, Lx, Ly, **kw)
        params += -1 * grad_rate * gradian
        
    return(es, ps) #(es[-1], ps[-1])


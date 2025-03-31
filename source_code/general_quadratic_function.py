import numpy as np



def gen_mat(inputlist, L, some_list):
    
    if inputlist.size != 2 * L * L:
        inputlist=np.resize(inputlist, 2*L*L)
    
    a_mat, b_mat =  np.split(inputlist, 2) 
    
    a_mat = np.reshape(a_mat,(L,L))
    b_mat = np.reshape(b_mat,(L,L))
    
    A_mat = (a_mat + a_mat.T)/2
    B_mat = (b_mat - b_mat.T)/2
    
    H = np.block([ [A_mat, B_mat],
        [-1*B_mat.conj(), -1*A_mat.conj()] ])
    
    E, U = np.linalg.eigh(H)
    
    ordr=np.arange(2*L)
    ordr[0:L] = ordr[0:L][::-1]
    U = U[:,ordr]
    
    g_mat, h_mat = U[0:L,0:L].T, U[L:, :L].T
    
    if some_list.size != L:
        some_list = np.resize(some_list, L)
    II = np.diag(some_list)
    C = np.dot(h_mat.T.conj(), h_mat) + np.dot(g_mat.T, np.dot(II, g_mat.conj())) - np.dot(h_mat.T.conj(), np.dot(II, h_mat))
    F = np.dot(h_mat.T.conj(), g_mat) + np.dot(g_mat.T, np.dot(II, h_mat.conj())) - np.dot(h_mat.T.conj(), np.dot(II, g_mat))
    
    # C = np.dot(h_mat.T.conj(), h_mat)
    # F = np.dot(h_mat.T.conj(), g_mat)
    
    # return(E, g_mat, h_mat)#np.diag(np.dot(U.T,np.dot(H, U))))
    return C, F


def FullEnergy_GQ(C, F, Vs, L):
    E=0
    for j in range(L):
        E += -(C[j,np.mod(j+1,L)] + C[np.mod(j+1,L),j])
        E += -Vs/2*(C[j,j]*C[np.mod(j+1,L),np.mod(j+1,L)] -F[j,np.mod(j+1,L)]*F[j,np.mod(j+1,L)] - C[j,np.mod(j+1,L)]*C[np.mod(j+1,L),j])

    return E


def GetGradian_GQ(input_array, Vs, L, some_list, p_shift = 0.001):
    
    grad_array = np.zeros(input_array.shape)
    
    for i_th in np.arange(input_array.size):
        
        input_array[i_th] += p_shift 
        C_mat, F_mat = gen_mat(input_array, L, some_list)
        energi1 = FullEnergy_GQ(C_mat, F_mat, Vs, L)
        
        input_array[i_th] += -2 * p_shift 
        C_mat, F_mat = gen_mat(input_array, L, some_list)
        energi2 = FullEnergy_GQ(C_mat, F_mat, Vs, L)
        
        grad_array[i_th] += (energi1-energi2)/(2*p_shift)
        
    return(grad_array)


def genfermion_optimization(Vs, L, some_list, grad_rate = 1.637, max_step = 200):
    params = np.random.rand( 2 * L * L )
    es = []
    ps = []
    for _ in np.arange(max_step):
        cmat,fmat = gen_mat(params, L, some_list)
        
        es.append(FullEnergy_GQ(cmat, fmat, Vs, L))
        ps.append(np.diag(cmat))
        
        gradian = GetGradian_GQ(params, Vs, L, some_list)
        
        params += -1*grad_rate * gradian
    
    return es[-1], ps[-1]


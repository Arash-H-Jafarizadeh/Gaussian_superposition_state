import numpy as np # type: ignore

########################################################################################################################################################################################
##################################################################################### 1D functiond #####################################################################################
########################################################################################################################################################################################

def modul_hoping_gate(parametrs):
    th1,th2,th3,th4 = parametrs[:4]    

    return np.array([[th1,th3+1j*th4],[th3-1j*th4,th2]])



def modul_circuit_correlation(parameters, geometrical, **kws):
    
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True

    N_layer, L = geometrical[:2]
    parameters = np.reshape(parameters, (N_layer*(L-int(not PBC)),4) )
    
    CIRC_MAT = np.eye(L)
    OUT_COR = np.diag([1,0]*(L//2)) 
    # OUT_COR = np.diag([1]+[0]*(L-1)) 
    
    for nl in range(N_layer):
        C_MAT_O = np.zeros((L,L), dtype=np.complex64)
        for oi in range(L//2):
            C_MAT_O[2*oi:2*oi+2,2*oi:2*oi+2] = modul_hoping_gate(parameters[oi+nl*(L-int(not PBC))])
        val_O, vec_O = np.linalg.eigh(C_MAT_O / 1.0)
        OP_O = np.linalg.multi_dot([vec_O, np.diag(np.exp( + 1j * val_O )), np.conjugate(np.transpose(vec_O)) ])
        
        C_MAT_E = np.zeros((L,L), dtype=np.complex64)
        for ei in range(L//2 - 1):
            C_MAT_E[2*ei+1:2*ei+3,2*ei+1:2*ei+3] = modul_hoping_gate(parameters[ei+nl*(L-int(not PBC))+L//2])
            
        if PBC:
            C_MAT_E[::L-1,::L-1] = modul_hoping_gate(parameters[L+nl*(L-1)-1])
        val_E, vec_E = np.linalg.eigh(C_MAT_E / 1.0)
        OP_E = np.linalg.multi_dot([vec_E, np.diag(np.exp( + 1j * val_E )), np.conjugate(np.transpose(vec_E)) ])
        
        CIRC_MAT = OP_E @ OP_O @ CIRC_MAT
        # CIRC_MAT = CIRC_MAT @ OP_O @ OP_E
        
    OUT_COR = np.linalg.multi_dot([CIRC_MAT, OUT_COR, np.conjugate(np.transpose(CIRC_MAT)) ])    
    return OUT_COR 



def fullenergy_QC_1d(input_mat, L, physical, **kws):
    
    V, J = physical[:2]    
    
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True
    
    out=0
    for n in range(L - int(not PBC)):
        out += -J*(input_mat[np.mod(n,L), np.mod(n+1,L)] + input_mat[np.mod(n+1,L), np.mod(n,L)]) 
        out += -0.5*V*( input_mat[np.mod(n,L), np.mod(n,L)]*input_mat[np.mod(n+1,L), np.mod(n+1,L)] - input_mat[np.mod(n,L), np.mod(n+1,L)]*input_mat[np.mod(n+1,L), np.mod(n,L)] )
    return out



def GetGradian_QC_1d(input_array, physical, geometrical, **kw):
    
    shift = kw['p_shift'] if 'p_shift' in kw.keys() else 0.001
    
    _, L = geometrical[:2]
    
    grad_array = np.zeros(input_array.shape)
    for i_th in range(input_array.size):
        input_array[i_th] += shift
        C_mat1 = modul_circuit_correlation(input_array, geometrical, **kw)
        energi1 = fullenergy_QC_1d(C_mat1, L, physical, **kw)
        
        input_array[i_th] += -2 * shift
        C_mat2 = modul_circuit_correlation(input_array, geometrical, **kw)
        energi2 = fullenergy_QC_1d(C_mat2, L, physical, **kw)
        
        grad_array[i_th] += np.real((energi1-energi2)/(2*shift))
        input_array[i_th] += shift
        
    return(grad_array)



def fermion_VQE_1d(physical, geometrical, **kws):
    
    grad_rate = kws['grad_rate'] if 'grad_rate' in kws.keys() else 0.1357
    max_iter = kws['max_iter'] if 'max_iter' in kws.keys() else 400
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True

    N_layer, L = geometrical[:2]
    parameters = 0.01*np.random.randn(N_layer*(L-int(not PBC))*4)
    
    Energy = []
    Density = np.zeros(L)    
    for itr in range(max_iter):
        
        c_mat = modul_circuit_correlation(parameters, geometrical,**kws)
        energy = fullenergy_QC_1d(c_mat,L, physical, **kws)
        
        Energy.append(energy.real)
        Density = np.diag(c_mat)
        
        grad_desc = GetGradian_QC_1d(parameters, physical, geometrical,**kws)
        parameters += -1 * grad_rate * grad_desc
    
    return(np.array(Energy), Density)



########################################################################################################################################################################################
##################################################################################### 2D functiond #####################################################################################
########################################################################################################################################################################################


def modul_hoping_gate(parametrs):

    th1,th2,th3,th4 = parametrs[:4]    

    return np.array([[th1,th3+1j*th4],[th3-1j*th4,th2]])



def modul_circuit_2d(parameters, geometrical, **kws):
    
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True

    Lx, Ly, N_layer = geometrical[:3] #convesion is Lx, Ly, N_layer
    
    L = Lx*Ly
    num_gates = Lx*(Ly-int(not PBC))+Ly*(Lx-int(not PBC))
    
    param_shap = (N_layer * num_gates, 4)
    parameters = np.reshape(parameters, param_shap)
    
    CIRC_MAT = np.eye(Lx*Ly)
    OUT_COR = np.kron(np.diag([1,0]*(Ly//2)), np.diag([1,0]*(Lx//2))) + np.kron(np.diag([0,1]*(Ly//2)), np.diag([0,1]*(Lx//2))) 
    
    norm = 1.0
    for nl in range(N_layer):
        C_MAT_1 = np.zeros((L,L), dtype=np.complex64)
        for l1 in range(L//2):
            C_MAT_1[2*l1:2*l1+2,2*l1:2*l1+2] = modul_hoping_gate(parameters[l1+nl*num_gates ])
        val_1, vec_1 = np.linalg.eigh(C_MAT_1 / norm)
        OP_1 = np.linalg.multi_dot([vec_1, np.diag(np.exp( + 1j * val_1 )), np.conjugate(np.transpose(vec_1)) ])
        
        C_MAT_2 = np.zeros((L,L), dtype=np.complex64)
        for y in range(Ly):
            for x in range(Lx//2-1):
                idxs = np.ix_([2*x+y*Lx +1, 2*x+y*Lx +2], [2*x+y*Lx +1, 2*x+y*Lx +2])
                C_MAT_2[idxs] = modul_hoping_gate(parameters[x + y*(Lx//2-1) + nl*num_gates + L//2])
        if PBC:
            for y in range(Ly):
                idxs = np.ix_([y*Lx+Lx-1, y*Lx], [y*Lx+Lx-1, y*Lx])
                C_MAT_2[idxs] = modul_hoping_gate(parameters[y + nl*num_gates + L//2 + Ly])
        val_2, vec_2 = np.linalg.eigh(C_MAT_2 / norm)
        OP_2 = np.linalg.multi_dot([vec_2, np.diag(np.exp( + 1j * val_2 )), np.conjugate(np.transpose(vec_2)) ])

        C_MAT_3 = np.zeros((L,L), dtype=np.complex64)
        for y in range(Ly//2):
            for x in range(Lx):
                idxs = np.ix_([2*Lx*y+x, 2*Lx*y+x+Lx], [2*Lx*y+x, 2*Lx*y+x+Lx])
                C_MAT_3[idxs] = modul_hoping_gate(parameters[x + y*(Lx) + nl*num_gates + L])
        val_3, vec_3 = np.linalg.eigh(C_MAT_3 / norm)
        OP_3 = np.linalg.multi_dot([vec_3, np.diag(np.exp( + 1j * val_3 )), np.conjugate(np.transpose(vec_3)) ])
                
        C_MAT_4 = np.zeros((L,L), dtype=np.complex64)
        for y in range(Ly//2-1):
            for x in range(Lx):
                idxs = np.ix_([Lx+x, 2*Lx+x], [Lx+x, 2*Lx+x])
                C_MAT_4[idxs] = modul_hoping_gate(parameters[x + y*(Lx) + nl*num_gates + 3*L//2])
        if PBC:
            for x in range(Lx):
                idxs = np.ix_([(Ly-1)*Lx+x, x], [(Ly-1)*Lx+x, x])
                C_MAT_4[idxs] = modul_hoping_gate(parameters[x + nl*num_gates + 3*L//2 + Ly])
        val_4, vec_4 = np.linalg.eigh(C_MAT_4 / norm)
        OP_4 = np.linalg.multi_dot([vec_4, np.diag(np.exp( + 1j * val_4 )), np.conjugate(np.transpose(vec_4)) ])
        
        CIRC_MAT = OP_4 @ OP_3 @OP_2 @ OP_1 @ CIRC_MAT # CIRC_MAT = CIRC_MAT @ OP_O @ OP_E
        
    OUT_COR = np.linalg.multi_dot([CIRC_MAT, OUT_COR, np.conjugate(np.transpose(CIRC_MAT)) ])
    
    return OUT_COR 



def fullenergy_QC_2d(input_C, physical, geometrical, **kws): # new and fast
    
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True
    
    Vs, J = physical[:2]
    Lx, Ly = geometrical[:2]
    
    ad_X = np.eye(Lx, k=1) + np.eye(Lx, k=-1)  + int(PBC)*np.eye(Lx, k=Lx-1)  + int(PBC)*np.eye(Lx, k=-Lx+1)    
    ad_Y = np.eye(Ly, k=1) + np.eye(Ly, k=-1)  + int(PBC)*np.eye(Ly, k=Ly-1)  + int(PBC)*np.eye(Ly, k=-Ly+1)
    ad_mat = np.kron(ad_Y, np.eye(Lx)) + np.kron(np.eye(Ly), ad_X)
    
    if Lx ==2 or Ly ==2:
        ad_mat = np.where(ad_mat,1,0)
    
    matJ = input_C + np.transpose(input_C)
    matV1 = np.outer(np.diag(input_C), np.diag(input_C))
    matV2 = input_C * np.transpose(input_C)

    E_mat = np.where(ad_mat, -J*matJ -(Vs/2)*matV1 +(Vs/2)*matV2, 0.0)
    
    return(np.sum(E_mat)/2)



def GetGradian_QC_2d(input_array, physical, geometrical, **kw):
    
    shift = kw['p_shift'] if 'p_shift' in kw.keys() else 0.0001
        
    grad_array = np.zeros(input_array.shape)
    for i_th in range(input_array.size):
        input_array[i_th] += shift
        C_mat1 = modul_circuit_2d(input_array, geometrical, **kw)
        energi1 = fullenergy_QC_2d(C_mat1, physical, geometrical, **kw)
        
        input_array[i_th] += -2 * shift
        C_mat2 = modul_circuit_2d(input_array, geometrical, **kw)
        energi2 = fullenergy_QC_2d(C_mat2, physical, geometrical, **kw)
        
        grad_array[i_th] += np.real((energi1-energi2)/(2*shift))
        input_array[i_th] += shift
        
    return(grad_array)



def fermion_VQE_2d(physical, geometrical, **kws):
    
    grad_rate = kws['grad_rate'] if 'grad_rate' in kws.keys() else 0.0789
    max_iter = kws['max_iter'] if 'max_iter' in kws.keys() else 200
    PBC = kws['PBC'] if 'PBC' in kws.keys() else True

    Lx, Ly, N_layer = geometrical[:3] #convesion is Lx, Ly, N_layer
    num_gates = Lx*(Ly-int(not PBC))+Ly*(Lx-int(not PBC))

    parameters = 0.01*np.random.randn(N_layer*num_gates*4)
    
    Energy = []   
    Density = np.zeros(Lx*Ly)     
    for itr in range(max_iter):
        
        c_mat = modul_circuit_2d(parameters, geometrical,**kws)
        energy = fullenergy_QC_2d(c_mat,physical, geometrical, **kws)
        
        Energy.append(energy.real)
        Density = np.diag(c_mat)

            
        grad_desc = GetGradian_QC_2d(parameters, physical, geometrical,**kws)
        parameters += -1 * grad_rate * grad_desc
    
    return(np.array(Energy), Density)


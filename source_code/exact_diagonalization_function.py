import numpy as np # type: ignore
import scipy as sp # type: ignore

############################################################################################## 1D functions ##############################################################################################

def fc(a,j,L):
    
    spin = np.array([[0,1],[1,0]]), np.array([[0,-1j],[+1j,0]]), np.array([[1,0],[0,-1]]) 
    out = 1
    for _ in range(j-1):
        out = np.kron(out,spin[2])
    
    op = 0.5*(spin[0]+a*1j*spin[1])
    out = np.kron(out, np.kron(op, np.eye(2**(L-j))))
    return out

def full_fermion_Ham(V, L, J=1, BC=True):
    out = np.zeros((2**L,2**L), dtype=np.complex_)    
    for n in range(1,L):
        out += -J*fc(+1,n,L) @ fc(-1,n+1,L) 
        out += -J*fc(+1,n+1,L) @ fc(-1,n,L)
        out += 0.5*V*( fc(+1,n,L) @ fc(-1,n,L) @ fc(+1,n+1,L) @ fc(-1,n+1,L))
    if BC:
        out += -J*fc(+1,L,L) @ fc(-1,1,L) 
        out += -J*fc(+1,1,L) @ fc(-1,L,L)
        out += 0.5*V*( fc(+1,L,L) @ fc(-1,L,L) @ fc(+1,1,L) @ fc(-1,1,L) )
    return out 


def full_total_particle_counts(L):
    
    out = np.zeros((2**L,2**L), dtype=np.complex_)  
    for n in range(1,L+1):    
        out += fc(+1,n,L) @ fc(-1,n,L)     
    
    return out


def full_particle_counts(input_state,L):
    
    input_state = np.array(input_state)
    Ns = np.zeros((L,) ) 
    for n in range(1,L+1):    
        n_th = fc(+1,n,L) @ fc(-1,n,L) 
        Ns[int(n-1)] += input_state @ n_th @ input_state.conj()
        
    return Ns



def bit_flip(input, indx1, indx2):
    temp_state = np.array(input[:])
    temp_state[ [indx1,indx2] ] = 1 - temp_state[ [indx1,indx2] ]
    return temp_state


def acting_ham(n:int, physical, L:int, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True
    
    V, J = physical
    
    state = []
    weight = []
    
    bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
    rolo_state = np.roll(bin_state, -1)
    
    if PBC:
        diag = 0.5 * V * ( rolo_state @ bin_state )
    else:
        diag = 0.5 * V * ( rolo_state[:-1] @ bin_state[:-1] )
        
    state.append(n)
    weight.append(float(diag))

    for l in range(L - int(not PBC)):
        if bin_state[ l ] != rolo_state[ l ]:
            new_state = bit_flip(bin_state, np.mod(l,L), np.mod(l+1,L))
            m = int("".join(str(i) for i in new_state),2)
            sign_factor = -1 if sum(bin_state) % 2 ==0 and int(abs(n-m)/(2**(L-1)-1))==1 else +1
            state.append( m )
            weight.append(- J * sign_factor ) ## - or + ? check this!

    return state, weight


def basis_set_N(N,L):
    
    basis_N = []
    for n in range(2**(L//2)-1,2**L - 2**(L//2) +1): #range(2**L):
        bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
        if sum(bin_state) == N:
            basis_N.append(n)
        
    return basis_N


def build_hamiltonian(L, parameters, **kwargs):
    sector = kwargs['sector'] if 'sector' in kwargs.keys() else L//2

    basis_N = basis_set_N(sector,L)
    HN = np.zeros( (np.size(basis_N), np.size(basis_N)) )
    
    for ind1, n in enumerate(basis_N):
        out_states, out_weights = acting_ham(n, parameters, L, **kwargs)

        for ii, m in enumerate(out_states):
            ind2 = basis_N.index(m)
            HN[ind2, ind1] += out_weights[ii]
                    
    return HN


def particle_count(L, parameters, K = 2, **kwargs):
    sector = kwargs['sector'] if 'sector' in kwargs.keys() else L//2

    energies, vectors = sp.linalg.eigh(build_hamiltonian(L, parameters, **kwargs), subset_by_index=[0,K-1]) 
    
    particle_stat = np.zeros((K,L))
    for w, n in enumerate(basis_set_N(sector,L)):
        for k in range(K):
            particle_stat[k,:] += (np.conjugate(vectors[w,k])*vectors[w,k]) * np.array([int(i) for i in np.binary_repr(n, width=L)])
    
    return particle_stat, energies


############################################################################################## 2D functions ##############################################################################################

def bit_flip_2D(input, indxs1:tuple, indxs2:tuple):
    temp_state = np.array(input[:])
    temp_state[indxs1] = 1 - temp_state[indxs1]
    temp_state[indxs2] = 1 - temp_state[indxs2]
    return temp_state


def basis_set_2D(L):
    
    N = L//2
    basis_N = []
    for n in range(2**N - 1, 2**L):
        bin_state = np.array([int(i) for i in np.binary_repr(n, width=L)])
        if sum(bin_state) == N:
            basis_N.append(n)
        
    return basis_N


def acting_ham_2D(n:int, dims:tuple, *physical, **kwargs):
    PBC = kwargs['PBC'] if 'PBC' in kwargs.keys() else True

    Lx, Ly = dims
    V, J = physical[:2]
    state_weight = []
    
    bin_state = np.array([int(i) for i in np.binary_repr(n, width = Lx*Ly)[::-1]]).reshape(Ly,Lx)
    
    diag = 0
    if PBC:
        for ly in range(Ly):
            rolled = np.roll(bin_state[ly,:], -1)
            diag += + 0.5 * V * (rolled @ bin_state[ly,:])
            diag += + 0.5 * V * (bin_state[ly,:] @ bin_state[np.mod(ly+1,Ly),:])
        if Ly == 2:
            diag += + 0.5 * V * (bin_state[ly,:] @ bin_state[np.mod(ly+1,Ly),:])
    else:
        for ly in range(Ly):
            rolled = np.roll(bin_state[ly,:], -1)
            diag += + 0.5 * V * (rolled[:-1] @ bin_state[ly,:-1])
        for ly in range(Ly - 1):
            diag += + 0.5 * V * (bin_state[ly,:] @ bin_state[ly+1,:])
        if Ly == 2:
            diag += + 0.5 * V * (bin_state[ly,:] @ bin_state[ly+1,:])
            
    state_weight.append([n, diag])
    
    for yy in range(Ly):
        for xx in range(Lx - int(not PBC)):
            indx1, indx2 = (yy,xx), (yy,np.mod(xx+1,Lx))
            if bin_state[indx1] != bin_state[indx2]:
                new_state = bit_flip_2D(bin_state, indx1, indx2)
                m = int("".join(str(i) for i in new_state.reshape(-1)[::-1] ), 2)
                st,nd = sorted( (indx1[1]+Lx*indx1[0], indx2[1]+Lx*indx2[0]) )               
                sign_factor = sum(bin_state.reshape(-1)[st+1:nd])
                if m not in np.transpose(state_weight)[0]:              
                    state_weight.append( [m, -J*(-1)**sign_factor] )

    for yy in range(Ly - int(not PBC)):
        for xx in range(Lx):
            indx1, indx2 = (yy,xx), (np.mod(yy+1,Ly),xx)
            if bin_state[indx1] != bin_state[indx2]:
                new_state = bit_flip_2D(bin_state, indx1, indx2)
                m = int("".join(str(i) for i in new_state.reshape(-1)[::-1] ), 2)
                st,nd = sorted( (indx1[1]+Lx*indx1[0], indx2[1]+Lx*indx2[0]) )                
                sign_factor = sum(bin_state.reshape(-1)[st+1:nd])
                if m not in np.transpose(state_weight)[0]:             
                    state_weight.append( [m, -J*(-1)**sign_factor] )

    return state_weight


def build_hamiltonian_2D(dims:tuple, *physical_parameter, **kwargs):
    
    Lx, Ly = dims
    basis_2D = basis_set_2D( Lx*Ly )
    HAM = np.zeros( (np.size(basis_2D), np.size(basis_2D)) )
    for ind1, n in enumerate(basis_2D):

        out_state_weight = acting_ham_2D(n,dims,*physical_parameter,**kwargs)
        for m in out_state_weight:
            ind2 = basis_2D.index(int(m[0]))
            HAM[ind2, ind1] += m[1]
                    
    return HAM


def particle_count_2D(Lx, Ly, *physical_parameter, K = 2, **kwargs):
    L = Lx*Ly
    N = L//2

    energies, vectors = sp.linalg.eigh( build_hamiltonian_2D( (Lx,Ly), *physical_parameter, **kwargs), subset_by_index=[0,K-1]) 
    
    particle_stat = np.zeros((K,L))
    for w, n in enumerate(basis_set_N(N,L)):
        for k in range(K):
            particle_stat[k,:] += (np.conjugate(vectors[w,k])*vectors[w,k]) * np.array([int(i) for i in np.binary_repr(n, width=L)])
    
    return particle_stat, energies




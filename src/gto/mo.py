import jax
import numpy as np
import jax.numpy as jnp

Ry = 2
delta = 1e-8
coeff_sto3g = jnp.array([[3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],
                        [0.16885540, 0.44463454]])
coeff_sto6g = jnp.array([[35.52322122, 0.00916359628],
                        [6.513143725, 0.04936149294],
                        [1.822142904, 0.16853830490],
                        [0.625955266, 0.37056279970],
                        [0.243076747, 0.41649152980],
                        [0.100112428, 0.13033408410]])

def make_hf(basis='sto3g'):
    n = xp.shape[0]
    dim_mat = 1 * n
    
    # coefficients of the basis
    if basis == 'sto3g':
        alpha = coeff_sto3g[:, 0]  # (4,)
        coeff = coeff_sto3g[:, 1:2].T  # (2, 4)
    else if basis == 'sto6g':
        alpha = coeff_sto6g[:, 0]  # (4,)
        coeff = coeff_sto6g[:, 1:2].T  # (2, 4)
    
    # intermediate varaibles
    sum_alpha = alpha[:, None] + alpha[None, :]  # (4, 4)
    pro_alpha = jnp.einsum('i,j->ij', alpha, alpha)  # (4, 4)
    alpha2 = pro_alpha / sum_alpha  # (4, 4)
     
    # erf function
    def f0(x):
        x += delta
        return jax.lax.erf(x)/x

    def hf(xp):
        # overlap
        Rmn = jnp.sum(jnp.square(xp[:, None, :] - xp[None, :, :]), axis=2) # (n, n)
        _ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmn->mpinqj', coeff, coeff, jnp.power(pro_alpha, 0.75)/jnp.power(sum_alpha, 1.5), 
                jnp.exp(-jnp.einsum('ij,mn->ijmn', alpha2, Rmn)))
        ovlp = jnp.reshape(jnp.einsum('mpinqj->mpnq', _ovlp), (dim_mat, dim_mat))

        # kinetic
        K = jnp.reshape(jnp.einsum('mpinqj,ij,ijmn->mpnq', _ovlp, alpha2, 3-2*jnp.einsum('ij,mn->ijmn', alpha2, Rmn)), (dim_mat, dim_mat))
        
        # potential
        rminj = (xp[:,None,None,None,:]*alpha[None,:,None,None,None]+xp[None,None,:,None,:]*alpha[None,None,None,:,None])/sum_alpha[None, :, None, :, None]
        x = jnp.einsum('ij,vminj->vminj', jnp.sqrt(sum_alpha), jnp.linalg.norm(xp[:,None,None,None,None,:]-rminj[None,...], axis=5))
        V = -jnp.reshape(jnp.einsum('mpinqj,ij,vminj->mpnq', _ovlp, jnp.sqrt(sum_alpha), f0(x)), (dim_mat, dim_mat))

        # core Hamiltonian
        hcore = K + V

        # diagonalization
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, np.diag(w**(-0.5)))
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), hcore, v)
        w1, _ = jnp.linalg.eigh(f1)
        e = 2 * jnp.sum(w1[0:n//2])

        return Ry * e # this is without vpp
    
    return hf

if __name__ == "__main__":
    from pyscf import gto, scf
    
    d = 1.4
    xp = jnp.array([[0,0,0],[d,0,0]])

    hf = make_hf()
    print(hf(xp))

    mol = gto.Mole()
    mol.unit = 'B'
    mol.atom = [['H', (0,0,0)], ['H', (d,0,0)]]
    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 1
    mf.get_veff = lambda *args: np.zeros(mf.get_hcore().shape)
    mf.kernel()
    print(Ry*(mf.e_tot-mf.energy_nuc()))

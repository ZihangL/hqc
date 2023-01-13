import numpy as np
import jax.numpy as jnp

Ry = 2
coeff_sto3g = jnp.array([[8.3744350009, -0.0283380461, 0.0000000000],
                          [1.8058681460, -0.1333810052, 0.0000000000],
                          [0.4852528328, -0.3995676063, 0.0000000000],
                          [0.1658236932, -0.5531027541, 1.0000000000]])
coeff_gthdzv = jnp.array([[8.3744350009, -0.0283380461, 0.0000000000],
                          [1.8058681460, -0.1333810052, 0.0000000000],
                          [0.4852528328, -0.3995676063, 0.0000000000],
                          [0.1658236932, -0.5531027541, 1.0000000000]])

def make_hf(basis=gth_dzv):
    n = xp.shape[0]
    dim_mat = 2 * n
    
    # coefficients of the basis
    alpha = coeff_gthdzv[:, 0]  # (4,)
    coeff = coeff_gthdzv[:, 1:3].T  # (2, 4)
    
    # intermediate varaibles
    sum_alpha = alpha[:, None] + alpha[None, :]  # (4, 4)
    pro_alpha = jnp.einsum('i,j->ij', alpha, alpha)  # (4, 4)
    alpha2 = pro_alpha / sum_alpha  # (4, 4)
     
    # erf function
    def f0(x):
        return jnp.erf(x)/x

    def hf(xp):
        # overlap
        Rmn = jnp.sum(jnp.square(xp[:, None, :] - xp[None, :, :]), axis=2) # (n, n)
        _ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmn->mpinqj', coeff, coeff, 
                jnp.power(pro_alpha, 0.75)/jnp.power(sum_alpha, 1.5), 
                jnp.exp(-jnp.einsum('ij,mn->ijmn', alpha2, Rmn)))
        ovlp = jnp.reshape(jnp.einsum('mpinqj->mpnq', ovlp), (dim_mat, dim_mat))

        # kinetic
        K = jnp.reshape(jnp.einsum('mpinqj,ij,ijmn->mpnq', _ovlp, 
                alpha2, 3-2*jnp.einsum('ij,mn->ijmn', alpha2, Rmn)), (dim_mat, dim_mat))
        
        # potential
        rminj = (xp[:,None,None,None,:]*alpha[None,:,None,None,None]+xp[None,None,:,None,:]*alpha[None,None,None,:,None])/sum_alpha[None, :, None, :, None]
        x = jnp.einsum('ij,vminj->vminj', jnp.sqrt(sum_alpha), jnp.linalg.norm(xp[:,None,None,None,None,:]-rminj[None,...], axis=5)
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

if name == "main":
    from pyscf import scf
    from zerovee import zerovee
    
    n, dim = 4, 3
    L, d = 10.0, 1.4
    center = np.array([L/2, L/2, L/2])
    offset = np.array([[d/2, 0., 0.],
                    [-d/2, 0., 0.]])
    xp = center + offset

    pyscfhf = zerovee(L, xp)
    pyscfovlp = pyscfhf.kmf.getovlp()[0].real
    pyscfK = scf.hf.gett(pyscfhf.cell, kpt=pyscfhf.kpts)[0].real
    pyscfVep = scf.hf.getpp(pyscfhf.cell, kpt=pyscfhf.kpts[0]).real
    pyscfE = pyscfhf.E()

    hf = hydrogen(L, xp)
    hfovlp = hf.overlap()
    hfK = hf.kinetic()
    hfVep = hf.potential()
    E = hf.kernel()+pyscfhf.Vpp()



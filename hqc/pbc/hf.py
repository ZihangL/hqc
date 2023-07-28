import jax
import jax.numpy as jnp
from pyscf.pbc import gto, scf

Ry = 2

def make_hf(n, L, nG, kpt, tol=1e-6, max_cycle=10):
    """
        Make PBC Hartree Fock function.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            L: float, side length of unit cell, unit: Bohr.
            nG: number plane wave bases with G points in one positive half axis.
            kpt: array of shape (dim,), kpoint in first Brillouin zone.
            tol: the tolerance for convergence.
            max_cycle: the maximum number of iterations.

        OUTPUT:
            hf: hartree fock function.
    """
    invvec = 2*jnp.pi/L
    omega = L**3 # volume of unit cell
    omega_inv = 4*jnp.pi/omega

    n_grid=2*nG+1
    n_basis = n_grid**3 # number of bases
    grid = jnp.arange(n_grid)-jnp.ones(n_grid, dtype=jnp.int_)*nG # (n_grid, ), nx=ny=nz=n_grid
    mesh = jnp.array(jnp.meshgrid(*( [grid]*3 ))).transpose(1,2,3,0) # (nx, ny, nz, 3)
    n2g = mesh.reshape(-1, 3) # (nx*ny*nz, 3)
    n2G = n2g*invvec # (nx*ny*nz, 3)
    # g2n = jnp.arange(n_basis).reshape(n_grid, n_grid, n_grid).transpose(1,0,2) # (nx, ny, nz)

    T = 0.5*jnp.diag(jnp.sum(jnp.square(kpt+n2G), axis=1)) # kinetic matrix
    VG = omega_inv/jnp.sum(jnp.square(n2G[:,None,:]-n2G[None,:,:]), axis=2)
    VG = VG.at[jnp.diag_indices_from(VG)].set(0) # potential matrix

    delta_rspq = jnp.zeros((n_basis,n_basis,n_basis,n_basis))
    for r in range(n_grid):
        for s in range(n_grid):
            if r == s:
                continue
            else:
                for p in range(n_grid):
                    for q in range(n_grid):
                        if (n2g[r]-n2g[s] == n2g[p]-n2g[q]).all():
                            delta_rspq = delta_rspq.at[r,s,p,q].set(1)
    eris = jnp.einsum('rs,rspq->prsq', VG, delta_rspq) # 2 orbital density integrals.
    
    def density_matrix(mo_coeff):
        """
            density matrix for closed shell system. (Hermitian)
        """
        dm = 2*jnp.einsum('im,jm->ij', mo_coeff[:,:n//2], mo_coeff.conjugate()[:,:n//2])
        return dm

    def hartree(dm):
        """
            Hartree matrix.
            Args:
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                hartree matrix: array of shape (n_ao, n_ao)
        """
        J = jnp.einsum('rs,prsq->pq', dm, eris)
        return J
        
    def exchange(dm):
        """
            Exchange matrix.
            Args:
                eris: array of shape (n_ao, n_ao, n_ao, n_ao), two-electron repulsion integrals.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                exchange matrix: array of shape (n_ao, n_ao)
        """
        K = jnp.einsum('rs,pqsr->pq', dm, eris)
        return K

    def hf(xp):
        # potential
        SI = jnp.sum(jnp.exp(-1j*(n2G[:,None,:]-n2G[None,:,:]).dot(xp.T)), axis=2) # (n_basis, n_basis)
        V = -VG*SI
        
        # core Hamiltonian
        Hcore = T + V

        # initialization
        _, mo_coeff = jnp.linalg.eigh(Hcore) # (n_ao, n_mo)
        dm = density_matrix(mo_coeff)

        # scf loop
        def body_fun(carry):
            _, E, dm = carry
            
            # Hartree & Exchange
            J = hartree(dm)
            K = exchange(dm)

            # Fock matrix
            F = Hcore + J - 0.5 * K

            # diagonalization
            _, mo_coeff = jnp.linalg.eigh(F) # (n_ao, n_mo)
            dm = density_matrix(mo_coeff)

            # energy
            E_new = 0.5*jnp.einsum('pq,qp', F+Hcore, dm) * Ry
            return (E.real, E_new.real, dm)
        
        def cond_fun(carry):
            return abs(carry[1] - carry[0]) > tol
            
        _, E, dm = jax.lax.while_loop(cond_fun, body_fun, (0., 1., dm))

        return E

    return hf

def pyscf_hf(L, xp, basis, kpt):

    """
        hartree fock without Vee pyscf benchmark.

        OUTPUT:
            energy without Vpp, unit: Ry
    """
    n = xp.shape[0]
    cell = np.eye(3)
    gtocell = gto.Cell()
    gtocell.unit = 'B'
    gtocell.atom = []
    for i in range(n):
        gtocell.atom.append(['H', tuple(xp[i])])
    gtocell.spin = 0
    gtocell.basis = basis
    gtocell.a = cell*L
    gtocell.build()

    kpts = [kpt.tolist()]
    # kpts = gtocell.make_kpts([1,1,1],scaled_center=[0,0,0])
    kmf = scf.khf.KRHF(gtocell, kpts=kpts)
    kmf.verbose = 0
    kmf.kernel()
    
    return Ry*(kmf.e_tot - kmf.energy_nuc())

if __name__=='__main__':
    import time
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    rs = 1.4
    n, dim = 4, 3
    L = (n*4./3*jnp.pi)**(1./3)*rs
    print("L:", L)
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    kpt = jnp.array([0,0,0])
    # kpt = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)

    nG = 2
    t0 = time.time()
    hf = make_hf(n, L, nG, kpt)
    t1 = time.time()
    E = hf(x)
    t2 = time.time()
    print("E:", E)
    print("make time:", t1-t0)
    print("run time:", t2-t1)

    # pyscf benchmark
    basis = 'gth-szv'
    E_pyscf = pyscf_hf(L, x, basis, kpt)
    t3 = time.time()
    print("pyscf E:", E_pyscf)
    print("pyscf time:", t3-t2)

    import resource
    print(f"{1e-3 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}MB")
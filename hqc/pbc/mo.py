import jax
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto, scf

from hqc.pbc.ao import gen_lattice, make_ao

unknown = 0.22578495
Ry = 2
const = (2 / jnp.pi)**0.75
coeff_sto3g = jnp.array([[3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],   
                        [0.16885540, 0.44463454]])
coeff_gthszv = jnp.array([[8.3744350009, -0.0283380461],
                        [1.8058681460, -0.1333810052],
                        [0.4852528328, -0.3995676063],
                        [0.1658236932, -0.5531027541]])
coeff_gthdzv = jnp.array([[8.3744350009, -0.0283380461, 0.0000000000],
                        [1.8058681460, -0.1333810052, 0.0000000000],
                        [0.4852528328, -0.3995676063, 0.0000000000],
                        [0.1658236932, -0.5531027541, 1.0000000000]])

def make_hf(n, L, basis, tol=1e-6, max_cycle=50):
    """
        Make PBC Hartree Fock function.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            L: float, side length of unit cell, unit: Bohr.
            basis: gto basis name, eg:'gth-szv'.
            tol: the tolerance for convergence.
            max_cycle: the maximum number of iterations.

        OUTPUT:
            hf: hartree fock function.
    """

    cell = jnp.eye(3)
    n_grid = round(L / 0.12) # same with pyscf

    lattice = gen_lattice(cell, L)
    ao = make_ao(lattice, basis)

    # coefficients of the basis
    if basis == 'gth-szv':
        n_ao = n
        alpha = coeff_gthszv[:, 0]  # (4,)
        coeff = coeff_gthszv[:, 1:2].T  # (1, 4)
    elif basis == 'sto-3g':
        n_ao = n
        alpha = coeff_sto3g[:, 0]  # (4,)
        coeff = coeff_sto3g[:, 1:2].T  # (1, 4)
    elif basis == 'gth-dzv':
        n_ao = 2*n
        alpha = coeff_gthdzv[:, 0]  # (4,)
        coeff = coeff_gthdzv[:, 1:3].T  # (2, 4)

    # intermediate variables
    sum_alpha = alpha[:, None] + alpha[None, :]  # (4, 4)
    pro_alpha = jnp.einsum('i,j->ij', alpha, alpha)  # (4, 4)
    alpha2 = pro_alpha / sum_alpha  # (4, 4)

    def make_mesh():
        # Rmesh
        grid = jnp.arange(n_grid) # (n_grid, ), nx = ny = nz = n_grid
        mesh = jnp.array(jnp.meshgrid(*( [grid]*3 ))).transpose(1,2,3,0) # (nx, ny, nz, 3)
        Rmesh = mesh.dot(cell.T)*L/n_grid # (nx, ny, nz, 3)
        
        # Gmesh
        Gmesh = mesh.dot(jnp.linalg.inv(cell))*2*jnp.pi/L # (nx, ny, nz, 3), range: (0, n_grid*2*pi/L)
        shift = jnp.append(jnp.zeros(n_grid-n_grid//2), jnp.ones(n_grid//2))  # (n_grid, ), (0000...1111...)
        shift3 = jnp.array(jnp.meshgrid(*( [shift]*3 ))).transpose(1,2,3,0) # (nx, ny, nz, 3)
        Gshift = shift3.dot(jnp.linalg.inv(cell))*n_grid*2*jnp.pi/L # (nx, ny, nz, 3)
        Gmesh -= Gshift # (nx, ny, nz, 3), range: (-n_grid*pi/L, n_grid*pi/L)

        # Coulomb potential on reciprocal space
        Gnorm2 = jnp.sum(jnp.square(Gmesh), axis=3)
        VG = 4 * jnp.pi / jnp.linalg.det(cell) / L ** 3 / Gnorm2 # (nx, ny, nz)
        VG = VG.at[0,0,0].set(0)
        return Rmesh, Gmesh, VG
    
    Rmesh, Gmesh, VG = make_mesh()
    
    def vep_int(xp, phi):
        """ 
            Vep matrix.
        """
        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)
        vlocR = n_grid**3 * jnp.fft.ifftn(vlocG).reshape(-1) # (nx*ny*nz,)
        vep = jnp.einsum('xm,x,xn->mn', phi.conjugate(), vlocR, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        return vep
    
    def density_int(phi, phir, phis):
        """
            2 orbital density integrals.
            To save RAM, we use integral index: 'p(rs)q', where 'rs' need to be mapped
        """
        rhoR = (phir*phis.conjugate()).reshape(n_grid,n_grid,n_grid) # (nx,ny,nz)
        rhoG = jnp.fft.fftn(rhoR)*jnp.linalg.det(cell)*(L/n_grid)**3 # (nx,ny,nz)
        VR = n_grid**3*jnp.fft.ifftn(VG*rhoG).reshape(-1) # (nx*ny*nz)
        eris0 = jnp.einsum('xp,x,xq->pq', phi.conjugate(), rhoG[0,0,0,None], phi)/n_grid**3*4*jnp.pi*L**2*jnp.linalg.det(cell)**(2/3)*unknown # (n_ao, n_ao)
        eris = jnp.einsum('xp,x,xq->pq', phi.conjugate(), VR, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        return jnp.stack((eris, eris0))
    
    def density_matrix(mo_coeff):
        """
            density matrix for closed shell system. (Hermitian)
        """
        dm = 2*jnp.einsum('im,jm->ij', mo_coeff[:,:n//2], mo_coeff.conjugate()[:,:n//2])
        return dm
    
    def hartree(eris, dm):
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

    def exchange(eris, dm):
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

    def hf(xp, kpt, use_remat=False):
        """
            PBC Hartree Fock without vee.
            INPUT:
                xp: array of shape (n, dim), position of protons.
                kpt: array of shape (dim,), kpoint in first Brillouin zone.
            OUTPUT:
                energy without vpp, unit: Rydberg.
        """
        assert xp.shape[0] == n

        # overlap
        Rmnc = jnp.sum(jnp.square(xp[:, None, None, :] - xp[None, :, None, :] + lattice[None, None, :, :]), axis=3)
        _ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmnl,l->mpinqjl', coeff, coeff, jnp.power(pro_alpha, 0.75)/jnp.power(sum_alpha, 1.5), 
            jnp.exp(-jnp.einsum('ij,mnl->ijmnl', alpha2, Rmnc)), jnp.exp(-1j*kpt.dot(lattice.T)))
        ovlp = jnp.reshape(jnp.einsum('mpinqjl->mpnq', _ovlp), (n_ao, n_ao))

        # kinetic
        T = jnp.reshape(jnp.einsum('mpinqjl,ij,ijmnl->mpnq', _ovlp, alpha2, 3-2*jnp.einsum('ij,mnc->ijmnc', alpha2, Rmnc)), (n_ao, n_ao))

        # potential
        phi = jax.lax.map(lambda xe: ao(xp, xe, kpt), Rmesh.reshape(-1, 3)) # (nx*ny*nz, n_ao)
        #phi = jax.vmap(ao, (None, 0), 0)(xp, Rmesh.reshape(-1, 3))
        if use_remat:
            V = jax.remat(vep_int)(xp, phi)
        else:
            V = vep_int(xp, phi)

        # core Hamiltonian
        Hcore = T + V

        # Hartree & Exchange integral initialization
        density_int1 = jax.vmap(density_int, (None, 1, None), 2)
        if use_remat:
            carry = jax.lax.map(lambda phis: jax.remat(density_int1)(phi, phi, phis), phi.transpose(1,0))
        else:
            carry = jax.lax.map(lambda phis: density_int1(phi, phi, phis), phi.transpose(1,0))
        eris = carry[:,0].transpose(1,2,0,3)
        eris0 = carry[:,1].transpose(1,2,0,3)

        # intialize molecular orbital
        mo_coeff = jnp.zeros((n_ao, n_ao))
        dm = density_matrix(mo_coeff) + 0j

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # scf loop
        def body_fun(carry):
            _, E, dm = carry
            
            # Hartree & Exchange
            J = hartree(eris, dm)
            K = exchange(eris+eris0, dm)

            # Fock matrix
            F = Hcore + J - 0.5 * K

            # diagonalization
            f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
            _, c1 = jnp.linalg.eigh(f1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm = density_matrix(mo_coeff)

            # energy
            E_new = 0.5*jnp.einsum('pq,qp', F+Hcore, dm) * Ry
            return (E.real, E_new.real, dm)
        
        def cond_fun(carry):
            return abs(carry[1] - carry[0]) > tol
            
        _, E, dm = jax.lax.while_loop(cond_fun, body_fun, (0., 1., dm))

        return E # this is without vpp, unit: Ry
    
    return hf

def pyscf_hf(L, xp, basis, kpt):

    """
        hartree fock without Vee pyscf benchmark.

        OUTPUT:
            energy without Vpp, unit: Ry
    """
    Ry = 2
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

    # ovlp = kmf.get_ovlp()
    # Hcore = kmf.get_hcore()
    # c2 = kmf.mo_coeff[0]
    # dm = kmf.make_rdm1()
    # J, K = kmf.get_jk()
    # F = kmf.get_fock()
    # print("pyscf overlap:\n", ovlp)
    # print("pyscf mo_coeff:\n", c2)
    # print("pyscf densigy matrix:\n", dm)
    # print("pyscf J:\n", J)
    # print("bands pyscf:", kmf.get_bands(kpts)[0][0])
    # print("pyscf K:\n", K)
    # print("pyscf F:\n", F)
    
    return Ry*(kmf.e_tot - kmf.energy_nuc())


if __name__=='__main__':
    import time
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    rs = 1.4
    n, dim = 4, 3
    basis = 'sto-3g'
    L = (n*4./3*jnp.pi)**(1./3)*rs
    print("L:", L)
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    # basis = 'sto-3g'
    # n, dim = 2, 3
    # L, d = 10.0, 1.4
    # center = np.array([L/2, L/2, L/2])
    # offset = np.array([[d/2, 0., 0.],
    #                 [-d/2, 0., 0.]])
    # x = center + offset

    kpt = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)
    # kpt = jnp.array([0,0,0])

    t0 = time.time()
    hf = make_hf(n, L, basis)
    t1 = time.time()
    print("make time:", t1-t0)
    
    E = hf(x, kpt)
    t2 = time.time()
    print("E:", E)
    print("time:", t2-t1)

    # batch test
    # batch = 32
    # x = jax.random.uniform(key, (batch, n, dim), minval=0., maxval=L)
    # kpt = jax.random.uniform(key, (batch, dim), minval=-jnp.pi/L, maxval=jnp.pi/L)
    # E = jax.vmap(hf, (0, 0), 0)(x, kpt)
    # t2 = time.time()
    # print("E:", E)
    # print("time:", t2-t1)

    E_pyscf = pyscf_hf(L, x, basis, kpt)
    t3 = time.time()
    print("pyscf E:", E_pyscf)
    print("pyscf time:", t3-t2)

    # x = jnp.concatenate([x, x]).reshape(2, n, dim)
    # print (jax.vmap(hf, (0, None), 0)(x, kpt))

    import resource
    print(f"{1e-3 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}MB")

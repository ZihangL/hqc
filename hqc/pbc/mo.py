import jax
import jax.numpy as jnp
from jax import vmap, grad
import numpy as np
from pyscf.pbc import gto, dft
from hqc.pbc.ao import gen_lattice, make_ao
import jax_xc

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

def make_dft(n, L, basis, tol=1e-6, max_cycle=10):
    """
        Make PBC Restricted Kohn-Sham for periodic systems with k-point sampling.
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
    n_grid3 = n_grid**3
    Omega = jnp.linalg.det(cell)*L**3

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
        VG = 4 * jnp.pi/Omega/Gnorm2 # (nx, ny, nz)
        VG = VG.at[0,0,0].set(0)
        return Rmesh, Gmesh, VG
    
    Rmesh, Gmesh, VG = make_mesh()
    
    def vep_int(xp, phi):
        """ 
            Vep matrix.
        """
        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)
        vlocR = n_grid3 * jnp.fft.ifftn(vlocG).reshape(-1) # (nx*ny*nz,)
        vep = jnp.einsum('xm,x,xn->mn', phi.conjugate(), vlocR, phi)*Omega/n_grid3 # (n_ao, n_ao)
        return vep
    
    def density_int(phi):
        """
            2 orbital density integrals.
            To save RAM, we use integral index: 'p(rs)q', where 'rs' need to be mapped
            Args:
                phi: array of shape (nx*ny*nz,, n_ao), wave function.
            Returns:
                eris: array of shape (n_ao, n_ao)
        """
        rhoR = jnp.einsum('xm,xn->xmn', phi, phi.conjugate()).reshape(n_grid,n_grid,n_grid,n_ao,n_ao) # (nx,ny,nz,n_ao,n_ao)
        rhoG = jnp.fft.fftn(rhoR, axes=(0,1,2))*jnp.linalg.det(cell)*(L/n_grid)**3 # (nx,ny,nz,n_ao,n_ao)
        eris = jnp.einsum('x,xrs,xqp->prsq', VG[1:,0,0], rhoG[1:,0,0], jnp.flip(rhoG[1:,0,0],0)) \
             + jnp.einsum('y,yrs,yqp->prsq', VG[0,1:,0], rhoG[0,1:,0], jnp.flip(rhoG[0,1:,0],0)) \
             + jnp.einsum('z,zrs,zqp->prsq', VG[0,0,1:], rhoG[0,0,1:], jnp.flip(rhoG[0,0,1:],0)) \
             + jnp.einsum('yz,yzrs,yzqp->prsq', VG[0,1:,1:], rhoG[0,1:,1:], jnp.flip(rhoG[0,1:,1:],(0,1))) \
             + jnp.einsum('xz,xzrs,xzqp->prsq', VG[1:,0,1:], rhoG[1:,0,1:], jnp.flip(rhoG[1:,0,1:],(0,1))) \
             + jnp.einsum('xy,xyrs,xyqp->prsq', VG[1:,1:,0], rhoG[1:,1:,0], jnp.flip(rhoG[1:,1:,0],(0,1))) \
             + jnp.einsum('xyz,xyzrs,xyzqp->prsq', VG[1:,1:,1:], rhoG[1:,1:,1:], jnp.flip(rhoG[1:,1:,1:],(0,1,2)))
        return eris

    def density_matrix(mo_coeff):
        """
            density matrix for closed shell system. (Hermitian)
            Args:
                mo_coeff: array of shape (n_ao, n_mo), molecular coefficients.
            Returns:
                dm: array of shape (n_ao, n_ao), density matrix.
        """
        dm = 2*jnp.einsum('im,jm->ij', mo_coeff[:,:n//2], mo_coeff.conjugate()[:,:n//2]).real
        return dm
    
    def density(xp, dm, r, kpt):
        """
            density at r.
            Args:
                xp: array of shape (n, dim), position of protons.
                dm: array of shape (n_ao, n_ao), density matrix.
                r: array of shape (dim,)
            Returns:
                density at r (dtype: float)
        """
        phi = ao(xp, r, kpt) # (n_ao,)
        dens = jnp.einsum('rs,r,s', dm, phi, phi.conjugate())
        return jnp.float64(dens.real)
    
    def e_xc(xp, dm, kpt):
        """
            Return Exc
            Args:
                xp: array of shape (n, dim), position of protons.
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                e_xc (dtype: float)
        """        
        density_r = lambda r: density(xp, dm, r, kpt)
        lda_x = jax_xc.lda_x(polarized=False)
        e_xc_r = lambda R: lda_x(density_r, R) * density_r(R)
        e_xc_R = vmap(e_xc_r)(Rmesh.reshape(-1, 3)) # (nx*ny*nz,)
        e_xc= jnp.sum(e_xc_R)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        return e_xc

    def v_xc(xp, dm, r, kpt):   
        """
            Return Vxc at r.
            Args:
                xp: array of shape (n, dim), position of protons.
                dm: array of shape (n_ao, n_ao), density matrix.
                r: array of shape (dim,)
            Returns:
                V_xc (dtype: float)
        """
        density_r = lambda r: density(xp, dm, r, kpt)
        lda_x = jax_xc.lda_x(polarized=False)
        lda_x_r = lambda R: lda_x(density_r, R)
        V_xc = lda_x_r(r)+density_r(r)*jnp.dot(grad(lda_x_r)(r),1/grad(density_r)(r))/3
        return V_xc

    def dft_xc(xp, dm, phi, kpt):
        """
            Return the DFT exchange correlation matrix.
            Args:
                xp: array of shape (n, dim), position of protons.
                dm: array of shape (n_ao, n_ao), density matrix.
                phi: array of shape (nx*ny*nz, n_ao), wave function.
            Returns:
                xc: array of shape (n_ao, n_ao), exchange correlation matrix.
        """
        v_xc_r = lambda r: v_xc(xp, dm, r, kpt)
        v_xc_R = vmap(v_xc_r)(Rmesh.reshape(-1, 3)) # (nx*ny*nz,)
        xc = jnp.einsum('xp,x,xq->pq', phi.conjugate(), v_xc_R, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        return xc

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

    def geneigensolver(F, v):
        """
            Return the eigenstate of the generalized eigenvalue problem FC=eSC.
        """
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
        _, c1 = jnp.linalg.eigh(f1)
        c = jnp.dot(v, c1) # (n_ao, n_mo)
        return c

    def dft(xp, kpt, use_remat=False):
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
        # phi = vmap(ao, (None, 0), 0)(xp, Rmesh.reshape(-1, 3))
        if use_remat:
            V = jax.remat(vep_int)(xp, phi)
        else:
            V = vep_int(xp, phi)

        # core Hamiltonian
        Hcore = T + V

        # Hartree & Exchange integral initialization
        eris = density_int(phi)

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # intialize molecular orbital
        mo_coeff = geneigensolver(Hcore, v)
        dm = density_matrix(mo_coeff)
            
        # scf loop
        def body_fun(carry):
            _, E, dm = carry
            
            # Hartree & Exchange-Correlation
            J = hartree(eris, dm)
            xc = dft_xc(xp, dm, phi, kpt)

            # Fock matrix
            H = Hcore + J + xc

            # molecular orbitals and density matrix
            mo_coeff = geneigensolver(H, v) # (n_ao, n_mo)
            dm = density_matrix(mo_coeff)

            # energy
            E_new = (jnp.einsum('pq,qp', Hcore+0.5*J, dm) + e_xc(xp, dm, kpt)) * Ry
            return (E.real, E_new.real, dm)
        
        def cond_fun(carry):
            return abs(carry[1] - carry[0]) > tol
            
        _, E, dm = jax.lax.while_loop(cond_fun, body_fun, (0., 1., dm))

        return E # this is without vpp, unit: Ry
    
    return dft

def pyscf_dft(L, xp, basis, kpt):
    """
        dft pyscf benchmark.

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
    kmf = dft.krks.KRKS(gtocell, kpts=kpts)
    kmf.xc = 'lda,'
    kmf = kmf.density_fit()
    kmf = kmf.newton()
    kmf.verbose = 0
    kmf.kernel()

    return Ry*(kmf.e_tot - kmf.energy_nuc())

if __name__=='__main__':
    import time
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    rs = 1.4
    n, dim = 4, 3
    basis = 'gth-szv'
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

    # kpt = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)
    kpt = jnp.array([0,0,0])

    t0 = time.time()
    krks = make_dft(n, L, basis)
    t1 = time.time()
    print("make time:", t1-t0)
    
    E = krks(x, kpt)
    t2 = time.time()
    print("E:", E)
    print("time:", t2-t1)

    # # batch test
    # batch = 32
    # x = jax.random.uniform(key, (batch, n, dim), minval=0., maxval=L)
    # kpt = jax.random.uniform(key, (batch, dim), minval=-jnp.pi/L, maxval=jnp.pi/L)
    # E = vmap(hf, (0, 0), 0)(x, kpt)
    # t2 = time.time()
    # print("E:", E)
    # print("time:", t2-t1)

    E_pyscf = pyscf_dft(L, x, basis, kpt)
    t3 = time.time()
    print("pyscf E:", E_pyscf)
    print("pyscf time:", t3-t2)

    # x = jnp.concatenate([x, x]).reshape(2, n, dim)
    # print (vmap(hf, (0, None), 0)(x, kpt))

    import resource
    print(f"{1e-3 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}MB")
  

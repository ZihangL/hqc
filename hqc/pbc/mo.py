import jax
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto, scf
import pyscf.gto
import pyscf.scf
from hqc.pbc.ao import gen_lattice, make_ao

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
max_cycle = 15

def make_hf(n, L, basis):
    """
        Make PBC Hartree Fock function.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            L: float, side length of unit cell, unit: Bohr.
            basis: gto basis name, eg:'gth-szv'.

        OUTPUT:
            hf: hartree fock function.
    """

    cell = jnp.eye(3)
    n_grid = round(L / 0.12) # same with pyscf

    lattice = gen_lattice(cell, L)
    ao = make_ao(lattice, basis)

    # coefficients of the basis
    if basis == 'gth-szv':
        dim_mat = n
        alpha = coeff_gthszv[:, 0]  # (4,)
        coeff = coeff_gthszv[:, 1:2].T  # (1, 4)
    elif basis == 'sto-3g':
        dim_mat = n
        alpha = coeff_sto3g[:, 0]  # (4,)
        coeff = coeff_sto3g[:, 1:2].T  # (1, 4)
    elif basis == 'gth-dzv':
        dim_mat = 2*n
        alpha = coeff_gthdzv[:, 0]  # (4,)
        coeff = coeff_gthdzv[:, 1:3].T  # (2, 4)

    # intermediate variables
    sum_alpha = alpha[:, None] + alpha[None, :]  # (4, 4)
    pro_alpha = jnp.einsum('i,j->ij', alpha, alpha)  # (4, 4)
    alpha2 = pro_alpha / sum_alpha  # (4, 4)

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
    
    def vep_int(xp, phi):
        """ 
            Vep matrix.
        """
        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        vlocG = -SI * VG # (nx, ny, nz)
        vlocR = n_grid**3 * jnp.fft.ifftn(vlocG).reshape(-1) # (nx*ny*nz,)
        vep = jnp.einsum('xm,x,xn->mn', phi.conjugate(), vlocR, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        return vep
    
    def density_int(phi):
        """
            2 orbital density integrals.
        """
        rhoR = jnp.einsum('xm,xn->xmn', phi, phi.conjugate()).reshape(n_grid,n_grid,n_grid,dim_mat,dim_mat) # (nx,ny,nz,n_ao,n_ao)
        rhoG = jnp.fft.fftn(rhoR, axes=(0, 1, 2))*jnp.linalg.det(cell)*(L/n_grid)**3 # (nx,ny,nz,n_ao,n_ao)
        return rhoG
    
    def density_matrix(mo_coeff):
        """
            density matrix for closed shell system. (Hermitian)
        """
        dm = 2*jnp.einsum('im,jm->ij', mo_coeff[:,:n//2], mo_coeff.conjugate()[:,:n//2])
        return dm
    
    def hartree_int(phi, rhoG, dm):
        """
            Hartree matrix.
            Args:
                phi: array of shape (nx*ny*nz, n_ao), atomic orbitals
                rhoG: array of shape (nx, ny, nz, n_ao, n_ao)
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                hartree matrix: array of shape (n_ao, n_ao)
        """
        rhog = jnp.einsum('mn,xyzmn->xyz', dm, rhoG) # (nx,ny,nz)
        VH = n_grid**3*jnp.fft.ifftn(VG*rhog).reshape(-1) # (nx*ny*nz,)
        J = jnp.einsum('xm,x,xn->mn', phi.conjugate(), VH, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        return J

    def exchange_int(phi, rhoG, dm):
        """
            Exchange matrix.
            Args:
                phi: array of shape (nx*ny*nz, n_ao), atomic orbitals
                rhoG: array of shape (nx, ny, nz, n_ao, n_ao)
                dm: array of shape (n_ao, n_ao), density matrix.
            Returns:
                exchange matrix: array of shape (n_ao, n_ao)
        """
        VX = n_grid**3*jnp.fft.ifftn(jnp.einsum('xyz,xyzmn->xyzmn',VG,rhoG), axes=(0,1,2)).reshape(-1,dim_mat,dim_mat) # (nx*ny*nz, n_ao, n_ao)
        K = jnp.einsum('rs,xp,xqs,xr->pq', dm, phi.conjugate(), VX, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
        K0 = jnp.einsum('rs,xp,xqs,xr->pq', dm, phi.conjugate(), rhoG[0,0,0,None,:,:], phi)/n_grid**3*4*jnp.pi*L**2*jnp.linalg.det(cell)**(2/3)
        return K+0.22578495*K0, K0

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
        ovlp = jnp.reshape(jnp.einsum('mpinqjl->mpnq', _ovlp), (dim_mat, dim_mat))

        # kinetic
        T = jnp.reshape(jnp.einsum('mpinqjl,ij,ijmnl->mpnq', _ovlp, alpha2, 3-2*jnp.einsum('ij,mnc->ijmnc', alpha2, Rmnc)), (dim_mat, dim_mat))

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
        rhoG = density_int(phi)

        # intialize molecular orbital
        mo_coeff = jnp.zeros((dim_mat, dim_mat))
        dm = density_matrix(mo_coeff)

        # diagonalization of overlap
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))

        # scf
        for cycle in range(max_cycle):

            # Hartree & Exchange
            J = hartree_int(phi, rhoG, dm)
            K, jks = exchange_int(phi, rhoG, dm)

            # Fock matrix
            F = Hcore + J - 0.5 * K

            # diagonalization
            f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
            _, c1 = jnp.linalg.eigh(f1)

            # molecular orbitals and density matrix
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            dm = density_matrix(mo_coeff)

            # energy
            E = 0.5*jnp.einsum('pq,qp', F+Hcore, dm)
            print("E:", E.real * Ry)

        bands = 0.5*jnp.einsum('pq,qa,pa->a', F+Hcore, mo_coeff, mo_coeff.conjugate())
        print("bands:", bands)
        # quick test
        # print("overlap:\n", ovlp)
        # print("mo_coeff:\n", mo_coeff)
        print("density matrix:\n", dm)
        # print("J:\n", J)
        print("K:\n", K)
        # print("F:\n", F)

        return E.real * Ry, K, jks # this is without vpp
    
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
    Hcore = kmf.get_hcore()
    c2 = kmf.mo_coeff[0]
    dm = kmf.make_rdm1()
    J, K = kmf.get_jk()
    F = kmf.get_fock()
    # print("pyscf overlap:\n", ovlp)
    # print("pyscf mo_coeff:\n", c2)
    print("pyscf densigy matrix:\n", dm)
    # print("pyscf J:\n", J)
    print("bands pyscf:", kmf.get_bands(kpts)[0][0])
    print("pyscf K:\n", K)
    # print("pyscf F:\n", F)
    
    return Ry*(kmf.e_tot - kmf.energy_nuc()), K


if __name__=='__main__':
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    rs = 1.4
    n, dim = 4, 3
    basis = 'gth-szv'
    L = (n*4./3*jnp.pi)**(1./3)*rs
    print("L:", L)
    key = jax.random.PRNGKey(43)
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

    hf = make_hf(n, L, basis)
    E, K, jks = hf(x, kpt)
    E_pyscf, K_pyscf = pyscf_hf(L, x, basis, kpt)
    print(jnp.mean((K_pyscf[0]-K)/jks))
    print("E:\n", E)
    print("pyscf E:\n", E_pyscf)

    # x = jnp.concatenate([x, x]).reshape(2, n, dim)
    # print (jax.vmap(hf, (0, None), 0)(x, kpt))

    import resource
    print(f"{1e-3 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}MB")

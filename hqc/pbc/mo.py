import jax
import jax.numpy as jnp
from hqc.pbc.ao import gen_lattice, make_ao

Ry = 2
const = (2 / jnp.pi)**0.75
coeff_gthszv = jnp.array([[8.3744350009, -0.0283380461],
                        [1.8058681460, -0.1333810052],
                        [0.4852528328, -0.3995676063],
                        [0.1658236932, -0.5531027541]])
coeff_gthdzv = jnp.array([[8.3744350009, -0.0283380461, 0.0000000000],
                        [1.8058681460, -0.1333810052, 0.0000000000],
                        [0.4852528328, -0.3995676063, 0.0000000000],
                        [0.1658236932, -0.5531027541, 1.0000000000]])

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
        vlocR = n_grid**3 * jnp.fft.ifftn(vlocG).reshape(-1) # (nx*ny*nz)
        return jnp.einsum('xm,x,xn->mn', phi.conjugate(), vlocR, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)
    
    def hartree_int(xp, kpt, phi, mo_coeff):
        """
            Hartree matrix.
        """
        dm = density_matrix(mo_coeff)
        rhoR = jax.vmap(density, (None, None, None, 0), 0)(dm, kpt, xp, Rmesh.reshape(-1,3)) # (nx*ny*nz)
        rhoG = (L/n_grid)**3*jnp.fft.fftn(rhoR.reshape(n_grid, n_grid, n_grid))
        VH = n_grid**3*jnp.fft.ifftn(VG*rhoG).reshape(-1) # (nx*ny*nz)
        return jnp.einsum('xm,x,xn->mn', phi.conjugate(), VH, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)

    def hf(xp, kpt, use_remat=False):
        """
            PBC Hartree Fock without vee.
            INPUT:
                xp: array of shape (n, dim), position of protons.
                kpt: array of shape (dim,), kpoint in first Brillouin zone.

            OUTPUT:
                energy, unit: Rydberg.
        """
        assert xp.shape[0] == n

        # overlap
        Rmnc = jnp.sum(jnp.square(xp[:, None, None, :] - xp[None, :, None, :] + lattice[None, None, :, :]), axis=3)
        _ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmnl,l->mpinqjl', coeff, coeff, jnp.power(pro_alpha, 0.75)/jnp.power(sum_alpha, 1.5), 
            jnp.exp(-jnp.einsum('ij,mnl->ijmnl', alpha2, Rmnc)), jnp.exp(-1j*kpt.dot(lattice.T)))
        ovlp = jnp.reshape(jnp.einsum('mpinqjl->mpnq', _ovlp), (dim_mat, dim_mat))

        # kinetic
        K = jnp.reshape(jnp.einsum('mpinqjl,ij,ijmnl->mpnq', _ovlp, alpha2, 3-2*jnp.einsum('ij,mnc->ijmnc', alpha2, Rmnc)), (dim_mat, dim_mat))

        # potential
        phi = jax.lax.map(lambda xe: ao(xp, xe, kpt), Rmesh.reshape(-1, 3)) # (nx*ny*nz, )
        #phi = jax.vmap(ao, (None, 0), 0)(xp, Rmesh.reshape(-1, 3))
        if use_remat:
            V = jax.remat(vep_int)(xp, phi)
        else:
            V = vep_int(xp, phi)

        # core Hamiltonian
        hcore = K + V

        # intialize molecular orbital
        mo_coeff = jnp.zeros((dim_mat, dim_mat))

        # scf
        for cycle in range(30):

            # Hartree term
            J = hartree_int(xp, kpt, phi, mo_coeff)

            # Hamiltonian
            h = hcore + J

            # diagonalization
            w, u = jnp.linalg.eigh(ovlp)
            v = jnp.dot(u, jnp.diag(w**(-0.5)))
            f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), h, v)
            w1, c1 = jnp.linalg.eigh(f1)
            mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
            E = jnp.sum(w1[0:n//2]) + jnp.einsum('pq,pm,qm', h, mo_coeff.conjugate(), mo_coeff)
            print("cycle:", cycle, "E:", E * Ry)

        return E * Ry # this is without vpp
    
    def density_matrix(mo_coeff):
        """
            density matrix for closed shell system.
        """
        dm = 2*jnp.einsum('im,jm->ij', mo_coeff[:,:n//2], mo_coeff.conjugate()[:,:n//2])
        return dm
    
    def density(dm, kpt, xp, r):
        """ 
            Returns density of electrons rho(r)
        Args:
            dm: array of shape (n_ao, n_ao), density matrix.
            kpt: array of shape (dim,), kpoint in first Brillouin zone.
            xp: array of shape (n, dim), position of protons.
            r: array of shape (3,)
        Returns:
            rho
        """
        ao_value = ao(xp, r, kpt) # (n_ao,)
        return jnp.einsum('i,j,ij', ao_value, ao_value.conjugate(), dm)

    return hf

if __name__=='__main__':
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    n = 4
    dim = 3
    rs = 1.4
    L = (n*4./3*jnp.pi)**(1./3)*rs

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    kpt = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)

    hf = make_hf(n, L , 'gth-szv')
    E = jax.jit(hf)(x, kpt)
    print(E)

    # x = jnp.concatenate([x, x]).reshape(2, n, dim)
    # print (jax.vmap(hf, (0, None), 0)(x, kpt))

    import resource
    print(f"{1e-3 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}MB")

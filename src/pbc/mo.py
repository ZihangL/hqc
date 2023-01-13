import jax
import numpy as np
import jax.numpy as jnp
from hyqc.pbc.ao import gen_lattice, make_ao

Ry = 2
n_grid = 30
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
    cell = jnp.eye(3)
    kpts = jnp.array([[0,0,0]])

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

    def vep_int(xp):
        phi = jax.lax.map(lambda xe: ao(xp, xe), Rmesh.reshape(-1, 3)) # (nx*ny*nz, )
        #phi = jax.vmap(ao, (None, 0), 0)(xp, Rmesh.reshape(-1, 3))
        
        SI = jnp.sum(jnp.exp(-1j*Gmesh.dot(xp.T)), axis=3) # (nx, ny, nz)
        Gnorm = jnp.linalg.norm(Gmesh, axis=3) # (nx, ny, nz)
        VG = 4 * jnp.pi / jnp.linalg.det(cell) / L ** 3 / jnp.square(Gnorm) # (nx, ny, nz)
        VG = VG.at[0,0,0].set(0)
        vlocG = -SI * VG # (nx, ny, nz)

        vlocR = n_grid**3 * jnp.fft.ifftn(vlocG) # (nx, ny, nz)
        vlocR = jnp.reshape(vlocR, -1) # (nx*ny*nz)
        return jnp.einsum('xm,x,xn->mn', phi.conjugate(), vlocR, phi)*jnp.linalg.det(cell)*(L/n_grid)**3 # (n_ao, n_ao)

    def hf(xp, use_remat=False):
        assert xp.shape[0] == n

        # overlap
        Rmnc = jnp.sum(jnp.square(xp[:, None, None, :] - xp[None, :, None, :] + lattice[None, None, :, :]), axis=3)
        _ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmnc,kc->kmpinqjc', coeff, coeff, jnp.power(pro_alpha, 0.75)/jnp.power(sum_alpha, 1.5), 
            jnp.exp(-jnp.einsum('ij,mnc->ijmnc', alpha2, Rmnc)), jnp.exp(1j*kpts.dot(lattice.T)))
        ovlp = jnp.reshape(jnp.einsum('kmpinqjc->kmpnq', _ovlp), (dim_mat, dim_mat))

        # kinetic
        K = jnp.reshape(jnp.einsum('kmpinqjc,ij,ijmnc->kmpnq', _ovlp, alpha2, 3-2*jnp.einsum('ij,mnc->ijmnc', alpha2, Rmnc)), (dim_mat, dim_mat))

        # potential
        if use_remat:
            V = jax.remat(vep_int)(xp)
        else:
            V = vep_int(xp)

        # core Hamiltonian
        hcore = K + V

        # diagonalization
        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, jnp.diag(w**(-0.5)))
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), hcore, v)
        w1, _ = jnp.linalg.eigh(f1)
        E = 2*jnp.sum(w1[0:n//2])
        
        return E * Ry # this is without vpp 

    return hf

if __name__=='__main__':
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    n = 16
    dim = 3
    rs = 1.4
    L = (n*4./3*jnp.pi)**(1./3)*rs

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    
    hf = make_hf(n, L , 'gth-szv')
    f = jax.jit(jax.grad(hf))(x)
    print (f)

    x = jnp.concatenate([x, x]).reshape(2, n, dim)
    print (jax.vmap(hf)(x))

    import resource
    print(f"{1e-3 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}MB")

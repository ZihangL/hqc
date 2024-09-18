import jax
import jax.numpy as jnp
import numpy as np

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

def gen_lattice(cell, L, rcut=18):
    """
        Return lattice T within the cutoff radius in real space.

        INPUT:
            cell: (dim, dim)
            L: float
            cell * L is the basic vector of unit cell.

        OUTPUT:
            lattice: (n_lattice, 3), unit: Bohr.
    """
    tmax = rcut//(min(jnp.linalg.norm(cell, axis=-1))*L)
    nt = np.arange(-tmax, tmax+1)
    nis = np.meshgrid(*( [nt]*3 ))
    lattice = np.array([ni.flatten() for ni in nis]).T.dot(cell.T)*L
    lattice2 = (lattice**2).sum(axis=-1)
    lattice = lattice[lattice2<=rcut**2] # (n_lattice, 3)
    return lattice

def make_ao(lattice, basis):
    """
        Make PBC gto orbitals function.
        INPUT:
            basis: basis name, eg:'gth-szv'.
        OUTPUT:
            eval_pbc_gto: PBC gto orbitals function.
    """
    if basis == 'gth-szv':
        coeff = coeff_gthszv
    elif basis == 'sto-3g':
        coeff = coeff_sto3g
    elif basis == 'gth-dzv':
        coeff = coeff_gthdzv

    @jax.remat
    def eval_pbc_gto(xp, xe, kpt):
        """
            PBC gto orbitals.
            INPUT:
                xp: array of shape (n, dim), position of protons in unit cell.
                xe: array of shape (dim,), position one electron in unit cell.
                kpt: array of shape (dim,), kpoint in first Brillouin zone.
            OUTPUT:
                pbc_gto: PBC gto orbitals at xe, shape:(n_ao,)
        """
        r = jnp.sum(jnp.square(xe[None, None, :] - xp[:, None, :] - lattice[None, :, :]), axis=2) # (n_p, n_lattice)
        exp_ikT = jnp.exp(1j*jnp.dot(kpt, lattice.T)) # (n_lattice, )
        pbc_gto = const * jnp.einsum('ib,i,ipl,l->pb', coeff[:, 1:], jnp.power(coeff[:, 0], 0.75), \
                jnp.exp(-jnp.einsum('i,pl->ipl', coeff[:, 0], r)), exp_ikT).reshape(-1)  # (n_p, n_lattice, n_basis)
        return pbc_gto

    return eval_pbc_gto

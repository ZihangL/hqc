import jax
import jax.numpy as jnp
import numpy as np
import gtos

def load_gto(basis='sto-3g'):
    """
        Load GTO coefficients from gto.py
        INPUT:
            basis
        OUTPUT
            (const, coeff_s, coeff_p) if the basis has p orbitals
            (const, coeff_s) else
    """
    basis = basis.replace("-", "")
    const = getattr(gtos, "const")
    if getattr(gtos, basis, "NOEXIST") == "NOEXIST":
        raise ValueError("Basis " + basis + " not found!")
    elif basis[-1]=='p':
        idx = basis.find('v')
        coeff_s = getattr(gtos, basis[:idx+1])
        coeff_p = getattr(gtos, basis)
        return const, coeff_s, coeff_p
    else:
        coeff_s = getattr(gtos, basis)
        return const, coeff_s

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

def make_ao(lattice, basis='sto-3g'):
    """
        Make PBC gto orbitals function.
        INPUT:
            basis: basis name, eg:'gth-szv'.
        OUTPUT:
            eval_pbc_gto: PBC gto orbitals function.
    """
    
    if basis[-1]=='p':
        const, coeff, coeff_p = load_gto(basis)
    else:
        const, coeff = load_gto(basis)

    @jax.remat
    def eval_pbc_gto_s(xp, xe):
        """
            PBC gto orbitals.
            INPUT:
                xp: array of shape (n, dim), position of protons in unit cell.
                xe: array of shape (dim,), position one electron in unit cell.
            OUTPUT:
                pbc_gto: PBC gto orbitals at xe, shape:(n_ao,)
        """
        r = jnp.sum(jnp.square(xe[None, None, :] - xp[:, None, :] - lattice[None, :, :]), axis=2) # (n_p, n_lattice)
        pbc_gto = const * jnp.einsum('ib,i,ipl->pb', coeff[:, 1:], jnp.power(coeff[:, 0], 0.75), \
                jnp.exp(-jnp.einsum('i,pl->ipl', coeff[:, 0], r))).reshape(-1)  # (n_ao, )
        return pbc_gto
    
    @jax.remat
    def eval_pbc_gto_sp(xp, xe):
        """
            Evaluates s and p orbitals at a batch of electron coordinates.
        Args:
            xp: array of shape (n, dim), position of protons in unit cell.
            xe: array of shape (dim,), position one electron in unit cell.
        Returns:
            pbc_gto: PBC gto orbitals at xe, shape:(n_ao,)
            n_ao = n_p * (n_basis)
            n_basis = n_basis_s + n_basis_p
        """
        unknown2 = 1.16384316
        n_p = xp.shape[0]
        r_vec = xe[None, None, :] - xp[:, None, :] - lattice[None, :, :]  # (n_p, n_lattice, dim) 
        r_square = jnp.sum(jnp.square(r_vec), axis=2)  # (n_p, n_lattice)
        pbc_gto_s = const * jnp.einsum('ib,i,ipl->pb', coeff[:, 1:], jnp.power(coeff[:, 0], 0.75), \
                jnp.exp(-jnp.einsum('i,pl->ipl', coeff[:, 0], r_square)))  # (n_p, n_basis_s)
        pbc_gto_p = jnp.einsum('ij,i,ipl,pld->pjd', coeff_p[:, 1:], jnp.power(coeff_p[:, 0], 1.25), \
                jnp.exp(-jnp.einsum('i,pl->ipl', coeff_p[:, 0], r_square)), \
                jnp.sqrt(1.5)*r_vec).reshape(n_p, -1) * unknown2 # (n_p, n_basis_p)
        pbc_gto = jnp.append(pbc_gto_s, pbc_gto_p, axis=1).reshape(-1) # (n_ao,)
        return pbc_gto

    return eval_pbc_gto_sp if basis[-1]=='p' else eval_pbc_gto_s

def make_hf_orbitals(n, L, rs, basis='sto-3g'):
    cell = jnp.eye(3)
    lattice = gen_lattice(cell, L*rs)
    gto = make_ao(lattice, basis)

    def hf_orbitals(xp, xe, mo_coeff, state_idx):
        """
            logpsi, which is generally complex.
            Args:
                xp: (n, 3)
                xe: (n, 3)
                mo_coeff: (n_ao, n_mo)
                state_idx: (nx,) the first half are spin up, the latter half are spin down.
            Returns:
                log_slater_determinant
        """  
        assert xp.shape[0] == n
        xp *= rs
        xe *= rs
        ao_val = jax.lax.map(lambda xe: gto(xp, xe), xe) # (n, n_ao)
        ao_val_up = ao_val[:n//2] # (n_up, n_ao)
        ao_val_dn = ao_val[n//2:] # (n_up, n_ao)
        slater_up = jnp.einsum('ij,jk->ik', ao_val_up, mo_coeff[:, state_idx[:n//2]]) # (n_up, n_up)
        slater_dn = jnp.einsum('ij,jk->ik', ao_val_dn, mo_coeff[:, state_idx[n//2:]]) # (n_dn, n_dn)
        return slater_up, slater_dn

    return hf_orbitals

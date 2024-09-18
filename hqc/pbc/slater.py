import jax
import jax.numpy as jnp

from hqc.pbc.ao import make_pbc_gto

def make_lcao_orbitals(n, L, rs, basis='sto-3g'):
    if 'aug' in basis.rsplit('-'):
        rcut = 24
    else:
        rcut = 18
    gto = make_pbc_gto(basis, L*rs, rcut=rcut)

    def lcao_orbitals(xp, xe, mo_coeff, state_idx):
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
        ao_val = jax.lax.map(lambda xe: gto(xp*rs, xe*rs), xe) # (n, n_ao)
        ao_val_up = ao_val[:n//2] # (n_up, n_ao)
        ao_val_dn = ao_val[n//2:] # (n_up, n_ao)
        slater_up = jnp.einsum('ij,jk->ik', ao_val_up, mo_coeff[:, state_idx[:n//2]]) # (n_up, n_up)
        slater_dn = jnp.einsum('ij,jk->ik', ao_val_dn, mo_coeff[:, state_idx[n//2:]]) # (n_dn, n_dn)
        return slater_up, slater_dn

    return lcao_orbitals
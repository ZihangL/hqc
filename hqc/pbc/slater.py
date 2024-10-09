import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from hqc.pbc.gto import make_pbc_gto

def make_slater(n: int, L: float, rs: float, basis: str, 
                rcut: float = 24, groundstate: bool = True) -> Callable:
    """
        Make a slater determinant function.
        Args:
            n: int, number of electrons.
            L: float, box size.
            rs: float, Wigner-Seitz radius.
            basis: str, basis set, default is 'sto-3g'.
            rcut: float, cutoff radius.
            groundstate: bool, whether to return the groundstate slater.
    """
    assert n % 2 == 0
    gto = make_pbc_gto(basis, L*rs, rcut=rcut)
    
    def slater(xp: jnp.ndarray, xe: jnp.ndarray, mo_coeff: jnp.ndarray, 
               state_idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
            logpsi, which is generally complex.
            Args:
                xp: (n, 3) in units of rs.
                xe: (n, 3) in units of rs.
                mo_coeff: (n_ao, n_mo)
                state_idx: (nx,) the first half are spin up, the latter half are spin down.
                    For groundstate of n=8 system, state_idx = [0, 1, 2, 3, 0, 1, 2, 3]
            Returns:
                slater up: (n_up, n_up)
                slater dn: (n_dn, n_dn)
                slater matrix of spin up and spin down electrons.
        """  
        assert xp.shape[0] == n
        ao_val = jax.lax.map(lambda xe: gto(xp*rs, xe*rs), xe) # (n, n_ao)
        ao_val_up = ao_val[:n//2] # (n_up, n_ao)
        ao_val_dn = ao_val[n//2:] # (n_up, n_ao)
        slater_up = jnp.einsum('ij,jk->ik', ao_val_up, mo_coeff[:, state_idx[:n//2]]) # (n_up, n_up)
        slater_dn = jnp.einsum('ij,jk->ik', ao_val_dn, mo_coeff[:, state_idx[n//2:]]) # (n_dn, n_dn)
        return slater_up, slater_dn

    if groundstate:
        groundstate_idx = jnp.concatenate([jnp.arange(n//2), jnp.arange(n//2)])
        return lambda xp, xe, mo_coeff: slater(xp, xe, mo_coeff, groundstate_idx)
    else:
        return slater

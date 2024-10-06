import jax
import numpy as np
import jax.numpy as jnp

def fixed_point(v_ovlp, Hcore, dm_init,
                hartree_fn, exchange_correlation_fn, density_matrix_fn, energy_fn, 
                tol=1e-7, max_cycle=100):
    """
        Fixed point iteration for SCF.
        Input:
            v_ovlp: array of shape (n_ao, n_ao), orthonormal matrix of overlap matrix S
                V^{\dagger}SV = I
            Hcore: array of shape (n_ao, n_ao), core Hamiltonian matrix
                Hcore = T + V
            dm_init: array of shape (n_ao, n_ao), initial guess of density matrix
            hartree_fn: function to compute coulomb repulsion between electrons (Hartree term)
                J = hartree_fn(dm), where dm and J has shape (n_ao, n_ao)
            exchange_correlation_fn: function to compute exchange-correlation energy
                Vxc = exchange_correlation_fn(dm), where dm and Vxc has shape (n_ao, n_ao)
                for Hartree-Fock, Vxc = -0.5*K, for DFT, Vxc = Vxc
                F = Hcore + J + Vxc
            density_matrix_fn: function to compute density matrix
                dm = density_matrix_fn(mo_coeff, w1), where mo_coeff has shape (n_ao, n_mo)
                and w1 has shape (n_mo,), dm has shape (n_ao, n_ao)
            energy_fn: function to compute energy
                energy = energy_fn(dm, J, Vxc)
            tol: float, tolerance
            max_iter: int, maximum number of iterations
        Output:

    """
    mo_coeff_init = jnp.empty_like(dm_init)
    w_init = jnp.empty_like(dm_init[0], dtype=jnp.float64)

    def body_fun(carry):
        _, E, _, dm, _, loop = carry

        # hartree and exchange-correlation term
        J = hartree_fn(dm)
        Vxc = exchange_correlation_fn(dm)

        # energy
        E_new = energy_fn(dm, Hcore, J, Vxc)

        # Fock matrix
        F = Hcore + J + Vxc

        # diagonalization
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), F, v)
        w1, c1 = jnp.linalg.eigh(f1)

        # molecular orbitals and density matrix
        mo_coeff = jnp.dot(v, c1) # (n_ao, n_mo)
        dm = density_matrix_fn(mo_coeff, w1) # (n_ao, n_ao)

        # ======================= debug =======================
        # jax.debug.print("fp loop: {x}", x=loop)
        # jax.debug.print("E:{x}, E_new:{y}", x=E, y=E_new)
        # =====================================================

        return E, E_new, mo_coeff, dm, w1, loop+1
    
    def cond_fun(carry):
        return (abs(carry[1] - carry[0]) > tol) * (carry[5] < max_cycle)
        
    _, E, mo_coeff, dm, w1, loop = jax.lax.while_loop(cond_fun, body_fun, (0., 1., mo_coeff_init, dm_init, w_init, 0))
    converged = not loop==max_cycle

    return mo_coeff, w1, E, converged

def diis():
    """
        DIIS for SCF.
    """
    pass



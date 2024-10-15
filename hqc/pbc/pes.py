import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

from hqc.pbc.lcao import make_lcao
from hqc.pbc.potential import potential_energy_pp

def make_pes(n: int, L: float, rs: float, basis: str,
             rcut: float = 24, 
             tol: float = 1e-7, 
             max_cycle: int = 100, 
             grid_length: float = 0.12,
             diis: bool = True, 
             diis_space: int = 8, 
             diis_start_cycle: int = 1, 
             diis_damp: float = 0.,
             use_jit: bool = True,
             dft: bool = False, 
             xc: str = 'lda,vwn',
             smearing: bool = False,
             smearing_method: str = 'fermi',
             smearing_sigma: float = 0.,
             search_method: str = 'newton', 
             search_cycle: int = 100, 
             search_tol: float= 1e-7,
             gamma: bool = True,
             Gmax: int = 15,
             kappa: float = 10) -> Callable:
    """
        Make Potential Energy Surface (PES) function for a periodic box.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            L: float, side length of unit cell, unit: rs.
            rs: float, unit: Bohr
            basis: gto basis name, eg:'gth-szv'.
            tol: the tolerance for convergence.
            max_cycle: the maximum number of iterations.
            grid_length: the grid length for real space grid, unit: Bohr.
            diis: if True, use DIIS.
            diis_space: the number of vectors in DIIS space.
            diis_start_cycle: the cycle to start DIIS.
            diis_damp: the damping factor for DIIS.
            use_jit: if True, use jit.
            dft: if True, use DFT, if False, use HF.
            xc: exchange-correlation functional.
            smearing: if True, use smearing.
            smearing_method: 'fermi' or 'gauss'.
            smearing_sigma: smearing width, unit: Hartree.
            search_method: 'bisect' or 'newton'.
            search_cycle: the maximum number of iterations for search.
            search_tol: the tolerance for searching mu.
            gamma: bool, if True, return pes(xp) for gamma point only, 
                         else, return pes(xp, kpt) for a single k-point.
            Gmax: int, the cutoff of G-vectors.
            kappa: float, the screening parameter.
        OUTPUT:
            pes: pes function.
                Inputs: xp: (n, dim) proton coordinates.
                        kpt: (3,) k-point coordinates, if gamma=False.
                Outputs: pes: float, total energy e = k+vep+vee+vpp, unit: Ry.
    """

    lcao = make_lcao(n, L, rs, basis, rcut=rcut, tol=tol, max_cycle=max_cycle, 
                     grid_length=grid_length, diis=diis, diis_space=diis_space, 
                     diis_start_cycle=diis_start_cycle, diis_damp=diis_damp,
                     use_jit=use_jit, dft=dft, xc=xc, smearing=smearing, 
                     smearing_method=smearing_method, smearing_sigma=smearing_sigma,
                     search_method=search_method, search_cycle=search_cycle, 
                     search_tol=search_tol, gamma=gamma)

    pes_gamma = lambda xp: lcao(xp)[2] + potential_energy_pp(xp, L, rs, kappa=kappa, Gmax=Gmax)

    pes_kpt = lambda xp, kpt: lcao(xp, kpt)[2] + potential_energy_pp(xp, L, rs, kappa=kappa, Gmax=Gmax)

    if gamma:
        pes = pes_gamma
    else:
        pes = pes_kpt

    if use_jit:
        return jax.jit(pes)
    else:
        return pes
    

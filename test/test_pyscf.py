import jax
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto, dft, scf
from typing import Tuple

from config import *

from hqc.basis.parse import load_as_str

# Global test variables
n, dim = 4, 3
rs = 1.5
L = (4/3*jnp.pi*n)**(1/3)
basis = 'gth-dzv'
xc = 'lda,vwn'
rcut = 24
grid_length = 0.12
smearing_sigma = 0.1
key = jax.random.PRNGKey(42)
gamma = False
smearing = True

key_p, key_kpt = jax.random.split(key)
xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)
if gamma:
    kpoint = jnp.array([0., 0., 0.])
else:
    kpoint = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

def pyscf_hf(n: int, 
             L: float, 
             rs: float, 
             xp: jnp.ndarray, 
             kpt: jnp.ndarray, 
             basis: str, 
             smearing: bool = False,
             smearing_method: str = 'fermi', 
             smearing_sigma: float = 0.1, 
             hf0: bool = False, 
             silent: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
               np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
               float, float, float, float, float, float]:
    """
        Pyscf Hartree Fock solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        xp: array of shape (n, dim), position of protons in rs unit.
        sigma: smearing width, unit: Hartree.
        kpt: k-point, array of shape (3,).
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        smearing: if True, use smearing (finite temperature Hartree Fock or thermal Hartree Fock).
        smearing_method: Fermi-Dirac smearing method, eg:'fermi', 'gaussian'.
        smearing_sigma: smearing width, unit: Hartree.
        hf0: if True, do Hartree Fock scf without Vee.
        silent: if True, suppress output.
    OUTPUT:
        pyscf_ovlp: overlap matrix, real or complex array of shape (n_ao, n_ao).
        pyscf_hcore: core Hamiltonian matrix, real or complex array of shape (n_ao, n_ao).
        pyscf_veff: effective potential matrix, real or complex array of shape (n_ao, n_ao).
        pyscf_j: Coulomb matrix, real or complex array of shape (n_ao, n_ao).
        pyscf_k: exchange matrix, real or complex array of shape (n_ao, n_ao).
        pyscf_fock: Fock matrix, real or complex array of shape (n_ao, n_ao).
        mo_coeff: molecular orbitals coefficients, complex array of shape (n_ao, n_mo).
        dm: density matrix, real or complex array of shape (n_ao, n_ao).
        bands: energy bands of corresponding molecular orbitals, 
            ranking of energy from low to high. array of shape (n_mo,) unit: Rydberg.
        occ: occupation number of molecular orbitals. array of shape (n_mo,).
        Eelec: electronic energy, unit: Rydberg.
        Ecore: core energy, unit: Rydberg.
        Vee: electron-electron interaction energy, unit: Rydberg.
        Vpp: proton-proton interaction energy, unit: Rydberg.
        Se: entropy.
        Etot: total energy, unit: Rydberg.
    """
    Ry = 2
    xp *= rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H':gto.parse(load_as_str('H', basis), optimize=True)}
    cell.symmetry = False
    cell.build()

    kpt = [kpt.tolist()]
    kmf = scf.hf.RHF(cell, kpt=kpt)
    # kmf.diis = False
    kmf.init_guess = '1e'
    if hf0:
        kmf.max_cycle = 1
        kmf.get_veff = lambda *args, **kwargs: np.zeros(kmf.get_hcore().shape)
    if smearing:
        kmf = scf.addons.smearing_(kmf, sigma=sigma, method=smearing_method)
    if silent:
        kmf.verbose = 0
    else:
        kmf.verbose = 4
    kmf.kernel()

    if not silent:
        
        print(f"{YELLOW}============ PYSCF DIR ============{RESET}")
        print(f"{GREEN}dir(kmf):{RESET}", dir(kmf))
        
        print(f"{YELLOW}============ PYSCF matrices ============{RESET}")
        # overlap matrix
        pyscf_ovlp = kmf.get_ovlp()
        print(f"{GREEN}pyscf overlap.shape:{RESET}", pyscf_ovlp.shape)
        # print(f"{GREEN}pyscf overlap:{RESET}\n", pyscf_ovlp)

        pyscf_hcore = kmf.get_hcore()
        print(f"{GREEN}pyscf hcore.shape:{RESET}", pyscf_hcore.shape)
        # print(f"{GREEN}pyscf hcore:{RESET}\n", pyscf_hcore)
        
        pyscf_veff = kmf.get_veff()
        print(f"{GREEN}pyscf veff.shape:{RESET}", pyscf_veff.shape)
        # print(f"{GREEN}pyscf veff:{RESET}\n", pyscf_veff)

        pyscf_j = kmf.get_j()
        print(f"{GREEN}pyscf j.shape:{RESET}", pyscf_j.shape)
        # print(f"{GREEN}pyscf j:{RESET}\n", pyscf_j)

        pyscf_fock = kmf.get_fock()
        pyscf_k = 2 * (pyscf_hcore + pyscf_j - pyscf_fock)
        print(f"{GREEN}pyscf k.shape:{RESET}", pyscf_k.shape)
        # print(f"{GREEN}pyscf k:{RESET}\n", pyscf_k)

        pyscf_fock = kmf.get_fock()
        print(f"{GREEN}pyscf fock.shape:{RESET}", pyscf_fock.shape)
        # print(f"{GREEN}pyscf fock:{RESET}\n", pyscf_fock)

        assert np.allclose(pyscf_fock, pyscf_hcore + pyscf_j - 0.5 * pyscf_k)

        print(f"{YELLOW}============ PYSCF solution ============{RESET}")
        # molecular coefficients
        mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
        print(f"{GREEN}mo_coeff.shape:{RESET}", mo_coeff.shape)
        # print("mo_coeff:\n", mo_coeff)

        # density matrix
        dm = kmf.make_rdm1() # (n_ao, n_ao)
        print(f"{GREEN}dm.shape:{RESET}", dm.shape)
        # print("dm:\n", dm)

        # energy bands
        bands = kmf.get_bands(kpts_band=kpt, kpt=kpt)[0][0]
        print(f"{GREEN}bands.shape:{RESET}", bands.shape)
        print(f"{GREEN}bands (Hartree):{RESET}", bands)
        print(f"{GREEN}bands (Rydberg):{RESET}", bands * Ry)

        # occupation number
        occ = kmf.get_occ(mo_energy=bands)
        print(f"{GREEN}occ.shape:{RESET}", occ.shape)
        print(f"{GREEN}occ:{RESET}", occ)

        print(f"{YELLOW}============ PYSCF energy ============{RESET}")
        # total energy
        Eelec = kmf.e_tot - kmf.energy_nuc()
        print(f"{GREEN}Eelec = K + Vep + Vee (Hartree):{RESET}", Eelec)
        print(f"{GREEN}Eelec = K + Vep + Vee (Rydberg):{RESET}", Eelec * Ry)

        Ecore = np.einsum('pq,qp', dm, pyscf_hcore).real
        print(f"{GREEN}K + Vep (Hartree):{RESET}", Ecore)
        print(f"{GREEN}K + Vep (Rydberg):{RESET}", Ecore * Ry)

        Vee = kmf.energy_elec()[1]
        print(f"{GREEN}Vee (Hartree):{RESET}", Vee)
        print(f"{GREEN}Vee (Rydberg):{RESET}", Vee * Ry)

        assert np.allclose(Vee, Eelec - Ecore)

        Vpp = kmf.energy_nuc()
        print(f"{GREEN}Vpp (Hartree):{RESET}", Vpp)
        print(f"{GREEN}Vpp (Rydberg):{RESET}", Vpp * Ry)

        Se = kmf.entropy
        print(f"{GREEN}Se:{RESET}", Se)

        Etot = kmf.e_tot
        print(f"{GREEN}Etot = K + Vep + Vee + Vpp (Hartree):{RESET}", Etot)
        print(f"{GREEN}Etot = K + Vep + Vee + Vpp (Rydberg):{RESET}", Etot * Ry)
        assert np.allclose(Etot, Eelec + Vpp)

    return pyscf_ovlp, pyscf_hcore, pyscf_veff, pyscf_j, pyscf_k, pyscf_fock, \
           mo_coeff, dm, bands*Ry, occ, \
           Eelec*Ry, Ecore*Ry, Vee*Ry, Vpp*Ry, Se, Etot*Ry

def pyscf_dft(n, L, rs, sigma, xp, basis, kpt, xc='lda,vwn', smearing=False, smearing_method='fermi'):
    """
        Pyscf DFT solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        kpt: k-point, array of shape (3,).
        xc: exchange correlation functional, eg:'lda', 'pbe', 'b3lyp'.
        smearing: if True, use Fermi-Dirac smearing
            (finite temperature Hartree Fock or thermal Hartree Fock).
    OUTPUT:
        mo_coeff: molecular orbitals coefficients, complex array of shape (n_ao, n_mo).
        bands: energy bands of corresponding molecular orbitals, 
            ranking of energy from low to high. array of shape (n_mo,).
    """
    Ry = 2
    xp *= rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H':gto.parse(load_as_str('H', basis), optimize=True)}
    cell.build()

    kpt = [kpt.tolist()]
    kmf = dft.RKS(cell, kpt=kpt)
    if smearing:
        kmf = scf.addons.smearing_(kmf, sigma=sigma, method=smearing_method)
    kmf.xc = xc
    # kmf.diis = False
    kmf.verbose = 0
    kmf.kernel()
    mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
    bands = kmf.get_bands(kpts_band=kpt, kpt=np.array(kpt))[0][0]
    E = kmf.e_tot - kmf.energy_nuc()

    # print("dir(kmf):", dir(kmf))
    # pyscf_ovlp = kmf.get_ovlp(kpt=kpt)[0]
    # print("pyscf overlap.shape:\n", pyscf_ovlp.shape)
    # print("pyscf overlap:\n", pyscf_ovlp)
    # pyscf_hcore = kmf.get_hcore(kpt=kpt)[0]
    # print("pyscf hcore.shape:\n", pyscf_hcore.shape)
    # print("pyscf hcore:\n", pyscf_hcore)
    # pyscf_veff = kmf.get_veff(kpt=kpt)
    # print("pyscf veff.shape:\n", pyscf_veff.shape)
    # print("pyscf veff:\n", pyscf_veff)
    # pyscf_dm = kmf.make_rdm1(kpt=kpt)
    # print("pyscf dm.shape:\n", pyscf_dm.shape)
    # print("pyscf dm:\n", pyscf_dm)
    # pyscf_j= kmf.get_j(kpt=kpt)
    # print("pyscf j.shape:\n", pyscf_j.shape)
    # print("pyscf j:\n", pyscf_j)
    # pyscf_fock = kmf.get_fock()
    # print("pyscf fock.shape:\n", pyscf_fock.shape)
    # print("pyscf fock:\n", pyscf_fock)
    # pyscf_k = 2 * (pyscf_hcore + pyscf_j - pyscf_fock)
    # print("pyscf k.shape:\n", pyscf_k.shape)
    # print("pyscf k:\n", pyscf_k)

if __name__ == "__main__":
    pyscf_hf(n, L, rs, xp, kpoint, basis, smearing=smearing, sigma=smearing_sigma, silent=False)
    # pyscf_dft(n, L, rs, smearing_sigma, xp, basis, kpoint, xc=xc, smearing=smearing)
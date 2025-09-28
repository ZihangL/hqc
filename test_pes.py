import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, scf, dft

from hqc.pbc.pes import make_pes
from hqc.basis.parse import load_as_str

jax.config.update("jax_enable_x64", True)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def pyscf_solver(n: int, 
                 L: float, 
                 rs: float, 
                 xp: jnp.ndarray, 
                 kpt: jnp.ndarray, 
                 basis: str, 
                 ifdft: bool = False,
                 xc: str = 'lda,vwn',
                 smearing: bool = False,
                 smearing_method: str = 'fermi', 
                 smearing_sigma: float = 0.1, 
                 hf0: bool = False, 
                 pyscf_verbose: int = 0,
                 silent: bool = True
            ) -> dict:
    """
        Pyscf Hartree Fock / DFT solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        xp: array of shape (n, dim), position of protons in rs unit.
        sigma: smearing width, unit: Hartree.
        kpt: k-point, array of shape (3,).
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        ifdft: if True, use DFT solver, otherwise use Hartree Fock solver.
        xc: exchange correlation functional, eg:'lda,vwn', only for .
        smearing: if True, use smearing (finite temperature Hartree Fock or thermal Hartree Fock).
        smearing_method: Fermi-Dirac smearing method, eg:'fermi', 'gaussian'.
        smearing_sigma: smearing width, unit: Hartree.
        hf0: if True, do Hartree Fock scf without Vee.
        pyscf_verbose: verbose level of pyscf.
        silent: if True, suppress output.
    OUTPUT:
        data: a dict including:
            ovlp: overlap matrix, real or complex array of shape (n_ao, n_ao).
            hcore: core Hamiltonian matrix, real or complex array of shape (n_ao, n_ao).
            veff: effective potential matrix, real or complex array of shape (n_ao, n_ao).
            j: Coulomb matrix, real or complex array of shape (n_ao, n_ao).
            k: exchange matrix, real or complex array of shape (n_ao, n_ao).
            fock: Fock matrix, real or complex array of shape (n_ao, n_ao).
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
    if ifdft:
        kmf = dft.RKS(cell, kpt=kpt)
        kmf.xc = xc
    else:
        kmf = scf.hf.RHF(cell, kpt=kpt)
    # kmf.diis = False
    kmf.init_guess = '1e'
    if hf0:
        kmf.max_cycle = 1
        kmf.get_veff = lambda *args, **kwargs: np.zeros(kmf.get_hcore().shape)
    if smearing:
        kmf = scf.addons.smearing_(kmf, sigma=smearing_sigma, method=smearing_method)
    kmf.verbose = pyscf_verbose
    kmf.kernel()

    pyscf_ovlp = kmf.get_ovlp()
    pyscf_hcore = kmf.get_hcore()
    pyscf_veff = kmf.get_veff()
    pyscf_j = kmf.get_j()
    pyscf_fock = kmf.get_fock()
    pyscf_k = 2 * (pyscf_hcore + pyscf_j - pyscf_fock)
    mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
    dm = kmf.make_rdm1() # (n_ao, n_ao)
    bands = kmf.get_bands(kpts_band=kpt, kpt=np.array(kpt))[0][0]
    occ = kmf.get_occ(mo_energy=bands)
    Eelec = kmf.e_tot - kmf.energy_nuc()
    Ecore = np.einsum('pq,qp', dm, pyscf_hcore).real
    Vee = kmf.energy_elec()[1]
    Vpp = kmf.energy_nuc()
    if smearing:
        Se = kmf.entropy
    else:
        Se = 0
    Etot = kmf.e_tot
    data = {"ovlp": pyscf_ovlp,
            "hcore": pyscf_hcore,
            "veff": pyscf_veff,
            "j": pyscf_j,
            "k": pyscf_k,
            "fock": pyscf_fock,
            "mo_coeff": mo_coeff,
            "dm": dm,
            "bands": bands*Ry,
            "occ": occ,
            "Eelec": Eelec*Ry,
            "Ecore": Ecore*Ry,
            "Vee": Vee*Ry,
            "Vpp": Vpp*Ry,
            "Se": Se,
            "Etot": Etot*Ry}
    assert np.allclose(data["fock"], data["hcore"] + data["j"] - 0.5 * data["k"])
    assert np.allclose(data["Vee"], data["Eelec"] - data["Ecore"])
    assert np.allclose(data["Etot"], data["Eelec"] + data["Vpp"])
    if not silent:
        print(f"{YELLOW}============ PYSCF DIR ============{RESET}")
        print(f"{BLUE}dir(kmf):{RESET}", dir(kmf))

        print(f"{YELLOW}============ PYSCF matrices ============{RESET}")        
        print(f"{BLUE}pyscf overlap.shape:{RESET}", data["ovlp"].shape)
        # print(f"{BLUE}pyscf overlap:{RESET}\n", data["ovlp"])
        print(f"{BLUE}pyscf hcore.shape:{RESET}", data["hcore"].shape)
        # print(f"{BLUE}pyscf hcore:{RESET}\n", data["hcore"])
        print(f"{BLUE}pyscf veff.shape:{RESET}", data["veff"].shape)
        # print(f"{BLUE}pyscf veff:{RESET}\n", data["veff"])
        print(f"{BLUE}pyscf j.shape:{RESET}", data["j"].shape)
        # print(f"{BLUE}pyscf j:{RESET}\n", data["j"])
        print(f"{BLUE}pyscf k.shape:{RESET}", data["k"].shape)
        # print(f"{BLUE}pyscf k:{RESET}\n", data["k"])
        print(f"{BLUE}pyscf fock.shape:{RESET}", data["fock"].shape)
        # print(f"{BLUE}pyscf fock:{RESET}\n", data["fock"])

        print(f"{YELLOW}============ PYSCF solution ============{RESET}")
        print(f"{BLUE}mo_coeff.shape:{RESET}", data["mo_coeff"].shape)
        # print("mo_coeff:\n", data["mo_coeff"])
        print(f"{BLUE}dm.shape:{RESET}", data["dm"].shape)
        # print("dm:\n", data["dm"])
        print(f"{BLUE}bands.shape:{RESET}", data["bands"].shape)
        # print(f"{BLUE}bands (Hartree):{RESET}", data["bands"])
        print(f"{BLUE}bands (Rydberg):{RESET}", data["bands"])
        print(f"{BLUE}occ.shape:{RESET}", data["occ"].shape)
        print(f"{BLUE}occ:{RESET}", data["occ"])

        print(f"{YELLOW}============ PYSCF energy ============{RESET}")
        # print(f"{BLUE}Eelec = K + Vep + Vee (Hartree):{RESET}", data["Eelec"])
        print(f"{BLUE}Eelec = K + Vep + Vee (Rydberg):{RESET}", data["Eelec"])
        # print(f"{BLUE}K + Vep (Hartree):{RESET}", data["Ecore"])
        print(f"{BLUE}K + Vep (Rydberg):{RESET}", data["Ecore"])
        # print(f"{BLUE}Vee (Hartree):{RESET}", data["Vee"])
        print(f"{BLUE}Vee (Rydberg):{RESET}", data["Vee"])
        # print(f"{BLUE}Vpp (Hartree):{RESET}", data["Vpp"])
        print(f"{BLUE}Vpp (Rydberg):{RESET}", data["Vpp"])
        print(f"{BLUE}Se:{RESET}", data["Se"])
        # print(f"{BLUE}Etot = K + Vep + Vee + Vpp (Hartree):{RESET}", data["Etot"])
        print(f"{BLUE}Etot = K + Vep + Vee + Vpp (Rydberg):{RESET}", data["Etot"])

    return data
    

# Global test variables
n, dim = 14, 3
rs = 1.86
L = (4/3*jnp.pi*n)**(1/3)
basis = 'gth-dzv'
xc = 'lda,vwn'
rcut = 24
grid_length = 0.5
T = 10000
reciprocal_beta = T/157888.088922572 # temperature in unit of Ry
smearing_sigma = reciprocal_beta/2 # temperature in Hartree unit
# smearing_sigma = 0.1
key = jax.random.PRNGKey(42)
key_p, key_kpt = jax.random.split(key)
xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)
kpt = jnp.array([0.25, 0.25, 0.25])
# kpt = jnp.array([0, 0, 0])
kpt = kpt * (2*jnp.pi/L/rs)
# kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

dft = False
smearing = True
gamma = False

pes = make_pes(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing, smearing_sigma=smearing_sigma, gamma=gamma, mode='dev')

for i in range(100):
    key_p, key_kpt = jax.random.split(key_p)
    
    if i == 4: # nan error
        
        xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)

        E, Ki, Vep, Vee, Vpp, Se, converged = pes(xp, kpt)

        pyscf_data = pyscf_solver(n, L, rs, xp, kpt, basis, ifdft=dft, xc=xc, smearing=smearing,
                                      smearing_method='fermi', smearing_sigma=smearing_sigma)

        print(f"{YELLOW}i:{RESET}", i)
        if not jnp.isnan(E):
            
            print(f"{BLUE}converged:{RESET}", converged)
            # assert converged
            # print(f"{GREEN}solver converged{RESET}")

            print(f"{BLUE}Etot:{RESET}", E)
            print(f"{BLUE}Etot_pyscf:{RESET}", pyscf_data['Etot'])
            # assert np.allclose(E, pyscf_data['Etot'], atol=1e-3)
            # print(f"{GREEN}same E{RESET}")

            print(f"{BLUE}kinetic:{RESET}", Ki)
            # print(f"{BLUE}Ecore_pyscf:{RESET}", pyscf_data["Ecore"])
            # assert np.allclose(Ki+Vep, pyscf_data["Ecore"], atol=1e-3)
            # print(f"{GREEN}same Ecore{RESET}")

            print(f"{BLUE}Vep:{RESET}", Vep)

            print(f"{BLUE}Vee:{RESET}", Vee)
            # print(f"{BLUE}Vee_pyscf:{RESET}", pyscf_data["Vee"])
            # assert np.allclose(Vee, pyscf_data["Vee"], atol=1e-3)
            # print(f"{GREEN}same Vee{RESET}")

            print(f"{BLUE}Vpp:{RESET}", Vpp)
            # print(f"{BLUE}Vpp_pyscf:{RESET}", pyscf_data["Vpp"])
            # assert np.allclose(Vpp, pyscf_data["Vpp"], atol=1e-3)
            # print(f"{GREEN}same Vpp{RESET}")

            print(f"{BLUE}Se:{RESET}", Se)
            print(f"{BLUE}Se_pyscf:{RESET}", pyscf_data["Se"])
            # assert np.allclose(Se, pyscf_data["Se"], atol=1e-3)
            # print(f"{GREEN}same Se{RESET}")

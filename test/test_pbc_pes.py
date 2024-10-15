import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, scf, dft
jax.config.update("jax_enable_x64", True)

from hqc.pbc.pes import make_pes
from hqc.basis.parse import load_as_str

# Global test variables
n, dim = 4, 3
rs = 1.5
L = (4/3*jnp.pi*n)**(1/3)
basis_set = ['gth-dzv', 'gth-dzvp']
xc = 'lda,vwn'
rcut = 24
grid_length = 0.12
smearing_sigma = 0.1
key = jax.random.PRNGKey(42)
key_p, key_kpt = jax.random.split(key)
xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)
kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

def pyscf_hf_etot(n, L, rs, sigma, xp, basis, kpt, hf0=False, smearing=False, smearing_method='fermi'):
    """
        Pyscf Hartree Fock solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        kpt: k-point, array of shape (3,).
        hf0: if True, do Hartree Fock scf without Vpp.
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
    kmf.verbose = 0
    kmf.kernel()
    mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
    bands = kmf.get_bands(kpts_band=kpt, kpt=kpt)[0][0]
    
    return kmf.e_tot * Ry

def pyscf_dft_etot(n, L, rs, sigma, xp, basis, kpt, xc='lda,vwn', smearing=False, smearing_method='fermi'):
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

    return kmf.e_tot * Ry

def pes_test(dft, smearing, gamma):
 
    print("\n============= test info =============")
    if dft and gamma:
        print("test: dft gamma")
    elif dft and not gamma:
        print("test: dft kpt")
    elif not dft and gamma:
        print("test: hf gamma")
    else:
        print("test: hf kpt")
    if gamma:
        kpoint = jnp.array([0., 0., 0.])
    else:
        kpoint = kpt
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("kpt:", kpoint)
    print("basis_set:", basis_set)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("smearing:", smearing)
    if smearing:
        print("smearing sigma:", smearing_sigma)
    print("xp:\n", xp)

    for basis in basis_set:
        print("\n-----", basis, "-----")

        # PBC energy test
        if dft:
            E_pyscf = pyscf_dft_etot(n, L, rs, smearing_sigma, xp, basis, kpoint, xc=xc, smearing=smearing)
        else:
            E_pyscf = pyscf_hf_etot(n, L, rs, smearing_sigma, xp, basis, kpoint, smearing=smearing)

        pes = make_pes(n, L, rs, basis, grid_length=grid_length, dft=dft, 
                        smearing=smearing, smearing_sigma=smearing_sigma, gamma=gamma)
        if gamma:
            E = pes(xp)
        else:
            E = pes(xp, kpoint)

        print("E:", E)
        print("E_pyscf:", E_pyscf)
        assert np.allclose(E, E_pyscf, atol=1e-3)
        print("same E")

def test_pes_hf_gamma():
    dft = False
    gamma = True
    smearing = False
    pes_test(dft, smearing, gamma)

def test_pes_hf_kpt():
    dft = False
    gamma = False
    smearing = False
    pes_test(dft, smearing, gamma)

def test_pes_dft_gamma():
    dft = True
    gamma = True
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_dft_kpt():
    dft = True
    gamma = False
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_hf_gamma_smearing():
    dft = False
    gamma = True
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_hf_kpt_smearing():
    dft = False
    gamma = False
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_dft_gamma_smearing():
    dft = True
    gamma = True
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_dft_kpt_smearing():
    dft = True
    gamma = False
    smearing = True
    pes_test(dft, smearing, gamma)

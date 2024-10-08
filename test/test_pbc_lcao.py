import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, dft, scf
from hqc.pbc.lcao import make_lcao
from hqc.basis.parse import load_as_str
jax.config.update("jax_enable_x64", True)


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
kpt = jax.random.uniform(key_kpt, (3,))


def pyscf_hf(n, L, rs, sigma, xp, basis, kpt, hf0=False, smearing=False, smearing_method='fermi'):
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

    return mo_coeff, bands * Ry, E * Ry


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
    
    return mo_coeff, bands * Ry, E * Ry


def lcao_test(dft, diis, smearing, gamma):
 
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
    print("DIIS:", diis)
    print("smearing:", smearing)
    print("smearing sigma:", smearing_sigma)
    print("xp:\n", xp)

    for basis in basis_set:
        print("\n-----", basis, "-----")

        # PBC energy test
        if dft:
            mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_dft(n, L, rs, smearing_sigma, xp, basis, kpoint, xc=xc, smearing=smearing)
        else:
            mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_hf(n, L, rs, smearing_sigma, xp, basis, kpoint, smearing=smearing)

        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, diis=diis, dft=dft, 
                         smearing=smearing, smearing_sigma=smearing_sigma, gamma=gamma)
        if gamma:
            mo_coeff, bands, E = lcao(xp)
        else:
            mo_coeff, bands, E = lcao(xp, kpoint)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]).conjugate())
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]).conjugate())
        print("mo_coeff:\n", mo_coeff)
        print("mo_coeff_pyscf:\n", mo_coeff_pyscf)
        assert np.allclose(mo_coeff, mo_coeff_pyscf, atol=1e-2)
        print("same mo_coeff")

        print("bands:\n", bands)
        print("bands_pyscf:\n", bands_pyscf)
        assert np.allclose(bands, bands_pyscf, atol=1e-3)
        print("same bands")

        print("E:", E)
        print("E_pyscf:", E_pyscf)
        assert np.allclose(E, E_pyscf, atol=1e-3)
        print("same E")

        # vmap test
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, 3)
        if gamma:  
            jax.vmap(lcao, 0, (0, 0, 0))(xp2)
        else:
            jax.vmap(lcao, (0, None), (0, 0, 0))(xp2, kpoint)

def test_hf_gamma_diis():
    dft = False
    gamma = True
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_kpt_diis():
    dft = False
    gamma = False
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_gamma_diis():
    dft = True
    gamma = True
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_kpt_diis():
    dft = True
    gamma = False
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_gamma_fp():
    dft = False
    gamma = True
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_kpt_fp():
    dft = False
    gamma = False
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_gamma_fp():
    dft = True
    gamma = True
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_kpt_fp():
    dft = True
    gamma = False
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_gamma_diis_smearing():
    dft = False
    gamma = True
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

def test_hf_kpt_diis_smearing():
    dft = False
    gamma = False
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

def test_dft_gamma_diis_smearing():
    dft = True
    gamma = True
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

def test_dft_kpt_diis_smearing():
    dft = True
    gamma = False
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

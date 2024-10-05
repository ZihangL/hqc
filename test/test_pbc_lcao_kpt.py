
import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, dft, scf
from hqc.pbc.lcao import make_lcao
from hqc.basis.parse import load_as_str
jax.config.update("jax_enable_x64", True)

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


def pyscf_dft(n, L, rs, sigma, xp, basis, kpt, xc='lda,', smearing=False, smearing_method='fermi'):
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
    kmf = scf.hf.RHF(cell, kpt=kpt)
    kmf = dft.RKS(cell)
    if smearing:
        kmf = scf.addons.smearing_(kmf, sigma=sigma, method=smearing_method)
    kmf.xc = xc
    # kmf.diis = False
    kmf.verbose = 0
    kmf.kernel()
    mo_coeff = kmf.mo_coeff  # (n_ao, n_mo)
    bands = kmf.get_bands(kpts_band=cell.make_kpts([1,1,1]))[0][0]
    E = kmf.e_tot - kmf.energy_nuc()

    # print("pyscf e_elec (Ha):", kmf.e_tot-kmf.energy_nuc())
    # print("pyscf e_elec (Ha):", kmf.energy_elec())
    # print("pyscf e_nuc (Ha):", kmf.energy_nuc())
    
    return mo_coeff, bands * Ry, E * Ry

def test_hf():
    n, dim = 4, 3
    rs = 1.5
    basis_set = ['gth-szv', 'gth-dzv', 'gth-dzvp']
    basis_set = ['gth-dzvp']
    rcut = 24
    grid_length = 0.12
    dft = False
    smearing = True
    sigma = 0.05
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (n, dim), minval=0., maxval=L)
    kpt = jax.random.uniform(key_kpt, (3,))
    
    print(print("\n============= begin test ============="))
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis_set:", basis_set)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("hf:", not dft)
    print("smearing:", smearing)
    print("xp:\n", xp)
    print("kpt:\n", kpt)

    for basis in basis_set:
        print("\n==========", basis, "==========")

        # PBC energy test
        mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_hf(n, L, rs, sigma, xp, basis, kpt, smearing=smearing)
        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing, smearing_sigma=sigma, gamma=False)
        mo_coeff, bands, E = lcao(xp, kpt)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]).conjugate())
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]).conjugate())
        # print("mo_coeff:\n", mo_coeff)
        # print("mo_coeff_pyscf:\n", mo_coeff_pyscf)
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
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
        jax.vmap(lcao, (0, None), (0, 0, 0))(xp2, kpt)
        

def test_dft():
    n, dim = 4, 3
    rs = 1.5
    basis_set = ['gth-szv', 'gth-dzv', 'gth-dzvp']
    rcut = 24
    grid_length = 0.12
    dft = True
    xc = "lda,vwn"
    smearing = False
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (n, dim), minval=0., maxval=L)
    kpt = jax.random.uniform(key_kpt, (3,))
    
    print(print("\n============= begin test ============="))
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis_set:", basis_set)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("dft:", dft)
    print("smearing:", smearing)
    print("xp:\n", xp)
    print("kpt:\n", kpt)

    for basis in basis_set:
        print("\n==========", basis, "==========")

        # PBC energy test
        mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_dft(n, L, rs, 0, xp, basis, kpt, xc=xc, smearing=smearing)
        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, xc=xc, smearing=smearing, gamma=False)
        mo_coeff, bands, E = lcao(xp, kpt)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]))
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]))
        print("mo_coeff:\n", mo_coeff)
        print("mo_coeff_pyscf:\n", mo_coeff_pyscf)
        assert np.allclose(mo_coeff, mo_coeff_pyscf, atol=1e-3)
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
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
        jax.vmap(lcao, 0, (0, 0, 0))(xp2)

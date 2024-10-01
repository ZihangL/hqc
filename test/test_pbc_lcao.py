import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, dft, scf
from hqc.pbc.lcao import make_lcao
from hqc.basis.parse import load_as_str
jax.config.update("jax_enable_x64", True)

def pyscf_hf(n, L, rs, sigma, xp, basis='sto-3g', hf0=False, smearing=False, smearing_method='fermi'):
    """
        Pyscf Hartree Fock solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
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

    kmf = scf.hf.RHF(cell)
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
    bands = kmf.get_bands(kpts_band=cell.make_kpts([1,1,1]))[0][0]
    E = kmf.e_tot - kmf.energy_nuc()

    # print("pyscf overlap:\n", kmf.get_ovlp())
    # print("pyscf kinetic:\n", kmf.get_ovlp)
    # print("pyscf potential:\n", kmf.get_vnuc())
    #print("pyscf Hcore:\n", kmf.get_hcore())
    print("pyscf energy per atom:", kmf.e_tot/n)
    print("pyscf converged", kmf.converged)

    return mo_coeff, bands * Ry, E * Ry


def pyscf_dft(n, L, rs, sigma, xp, basis='sto-3g', xc='lda,', smearing=False, smearing_method='fermi'):
    """
        Pyscf DFT solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
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
    rcut = 24
    grid_length = 0.12
    dft = False
    smearing = False
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    
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

    for basis in basis_set:
        print("\n==========", basis, "==========")

        # PBC energy test
        mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_hf(n, L, rs, 0, xp, basis, smearing=smearing)
        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing)
        mo_coeff, bands, E = lcao(xp)

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
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    
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

    for basis in basis_set:
        print("\n==========", basis, "==========")

        # PBC energy test
        mo_coeff_pyscf, bands_pyscf, E_pyscf = pyscf_dft(n, L, rs, 0, xp, basis, xc=xc, smearing=smearing)
        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, xc=xc, smearing=smearing)
        mo_coeff, bands, E = lcao(xp)

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

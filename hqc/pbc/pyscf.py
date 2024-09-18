import numpy as np
from pyscf.pbc import dft, gto, scf
from hqc.basis.parse import load_as_str

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

    # print("pyscf overlap:\n", kmf.get_ovlp())
    # print("pyscf kinetic:\n", kmf.get_ovlp)
    # print("pyscf potential:\n", kmf.get_vnuc())
    # print("pyscf Hcore:\n", kmf.get_hcore())
    print("pyscf e_tot:", (kmf.e_tot - kmf.energy_nuc())*Ry)

    return mo_coeff[:,::-1]+0j, bands[::-1] * Ry 

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

    # print("pyscf e_elec (Ha):", kmf.e_tot-kmf.energy_nuc())
    # print("pyscf e_elec (Ha):", kmf.energy_elec())
    # print("pyscf e_nuc (Ha):", kmf.energy_nuc())
    
    return mo_coeff[:,::-1]+0j, bands[::-1] * Ry 

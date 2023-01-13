from config import *
from pyscf import gto, scf
from hyqc.gto.ao import make_hf

def test_gto_mo():
    Ry = 2
    d = 1.4
    xp = jnp.array([[0,0,0],[d,0,0]])

    hf = make_hf()
    # print(hf(xp))

    mol = gto.Mole()
    mol.unit = 'B'
    mol.atom = [['H', (0,0,0)], ['H', (d,0,0)]]
    mol.basis = 'sto3g'
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 1
    mf.get_veff = lambda *args: np.zeros(mf.get_hcore().shape)
    mf.kernel()
    # print(Ry*(mf.e_tot-mf.energy_nuc()))

    assert np.allclose(hf(xp), Ry*(mf.e_tot-mf.energy_nuc()))
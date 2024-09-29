import jax
import jax.numpy as jnp
import numpy as np
import itertools
from hqc.pbc.lcao import make_lcao

from test_pbc_lcao import pyscf_dft, pyscf_hf
jax.config.update("jax_enable_x64", True)

def make_atoms(
    ncopy = [2, 2, 2],
):
  """make atom pyscf style coords
  """
  lattice = (4/3*np.pi*2)**(1/3) # lattice constant for bcc lattice, length unit rs*a0
  atom_strs = []
  for ii,jj,kk in itertools.product(range(ncopy[0]), range(ncopy[1]), range(ncopy[2])):
    xx = ii * lattice
    yy = jj * lattice
    zz = kk * lattice
    atom_strs.append([xx, yy, zz])
    atom_strs.append([xx+0.5*lattice, yy+0.5*lattice, zz+0.5*lattice])
  return np.array(atom_strs)

def test_bcc_solid_hf():
    n, dim = 16, 3
    rs = 1.31
    basis_set = ['gth-szv'] #, 'gth-dzv', 'gth-dzvp']
    rcut = 24
    grid_length = 0.12
    dft = False
    smearing = False
    sigma = 0.002
    L = (4/3*jnp.pi*n)**(1/3)

    xp = make_atoms([2, 2, 2])
    print (xp.shape)
    
    print("\n============= begin test =============")
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
        mo_coeff_pyscf, bands_pyscf = pyscf_hf(n, L, rs, sigma, xp, basis, smearing=smearing)
        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing, smearing_sigma=sigma)
        mo_coeff, bands = lcao(xp)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]))
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]))
        print (np.abs(mo_coeff - mo_coeff_pyscf).max())
        print (np.abs(bands - bands_pyscf).max())

test_bcc_solid_hf()

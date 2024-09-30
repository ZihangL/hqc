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
    dim = 3
    rs = 1.31
    basis_set = ['gth-szv'] # , 'gth-dzv', 'gth-dzvp']
    rcut = 24
    grid_length = 0.12
    dft = False
    xc = "lda,vwn"
    smearing = False
    sigma = 0.0 # smearing parameter 
    perturbation = 0.0 # perturbation strength for atom position
    max_cycle = 50

    xp = make_atoms([2, 2, 2]) # bcc crystal
    n = xp.shape[0]
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp += jax.random.normal(key, (n, dim)) * perturbation
    xp = xp - L * jnp.floor(xp/L)

    print("\n============= begin test =============")
    print("n:", n)
    print("rs:", rs)
    print("L:", L)
    print("basis_set:", basis_set)
    print("rcut:", rcut)
    print("grid_length:", grid_length)
    print("hf:", not dft)
    print("smearing:", smearing)
    print("perturbation:\n", perturbation)
    print("xp:\n", xp)

    for basis in basis_set:
        print("\n==========", basis, "==========")
        if dft: 
            mo_coeff_pyscf, bands_pyscf = pyscf_dft(n, L, rs, sigma, xp, basis, xc=xc, smearing=smearing)
        else:
            mo_coeff_pyscf, bands_pyscf = pyscf_hf(n, L, rs, sigma, xp, basis, smearing=smearing)

        lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, smearing=smearing, smearing_sigma=sigma, max_cycle = max_cycle)
        mo_coeff, bands = lcao(xp)

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]))
        mo_coeff_pyscf = mo_coeff_pyscf @ jnp.diag(jnp.sign(mo_coeff_pyscf[0]))

        print ("coef diff", np.abs(mo_coeff - mo_coeff_pyscf).max())
        print ("band diff", np.abs(bands - bands_pyscf).max())

        print("bands:\n", bands)
        print("bands_pyscf:\n", bands_pyscf)

        dm = mo_coeff[:, :n//2] @ (mo_coeff[:, :n//2].T)
        dm_pyscf = mo_coeff_pyscf[:, :n//2] @ (mo_coeff_pyscf[:, :n//2].T)

        print ("dm diff", np.abs(dm - dm_pyscf).max())
        
        np.set_printoptions(suppress=True)
        print("dm:\n", dm)
        print("dm_pyscf:\n", dm_pyscf)
        print("diff:\n", dm - dm_pyscf)

test_bcc_solid_hf()

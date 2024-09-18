# Hydrogen-QC

Quantum chemistry calculations in Hydrogen system.

Returns Hartree fock energy of PBC or isolated Hydrogen system without Vpp, unit: Rydberg.

Branch 'falcon' is for the fastest and cheapest Hartree Fock and DFT algorithm of PBC hydrogen system.

## Install

clone and cd to this directory.

use "pip install -e ." to install hydrogenqc.
```bash
pip install -e .
```

use "pip uninstall hqc" to uninstall.
```bash
pip unintall hqc
```

## Import

use "import hqc" directly to import this package anywhere.
```python
import hqc
```

## Example
Compare the DFT results with Pyscf

```python
from hqc.pbc.lcao import make_lcao
from hqc.pbc.pyscf import pyscf_dft
import jax
import jax.numpy as jnp

rs = 1.25
n, dim = 8, 3
basis = 'gth-dzv'
T = 10000 # K
beta = 157888.088922572/T # inverse temperature in unit of 1/Ry
sigma = 1/beta/2 # temperature in Hartree unit

L = (4/3*jnp.pi*n)**(1/3)
key = jax.random.PRNGKey(42)
xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

lcao = make_lcao(n, L, rs, basis, dft=True, smearing=True, smearing_sigma=sigma)
mo_coeff, bands = lcao(xp)
print("================= solver =================")
# print("mo_coeff:\n", mo_coeff)
print("bands:\n", bands)

mo_coeff_dft, energy_dft = pyscf_dft(n, L, rs, sigma, xp, basis, xc='lda,vwn', smearing=True, smearing_method='fermi')
print("================= pyscf =================")
# print("mo_coeff:\n", mo_coeff)
print("bands:\n", bands)                                          
```
or use "python main.py" to run a specific kind of test.
```bash
python main.py
```

## Requirements

        jax

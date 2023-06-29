# Hydrogen-QC

Quantum chemistry calculations in Hydrogen system.

Returns Hartree fock energy of PBC or isolated Hydrogen system without Vpp, unit: Rydberg.

## Install

clone and cd to this directory.

use "pip install -e ." to install hydrogenqc.

use "pip uninstall hqc" to uninstall.

## Import

use "import hqc" directly to import this package anywhere.

## Example

```python
from hqc.pbc.mo import make_hf
import jax
import jax.numpy as jnp

rs = 1.25
n, dim = 16, 3
L = (4/3*jnp.pi*n)**(1/3)*rs
key = jax.random.PRNGKey(42)
xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
kpt = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)

hf = make_hf(n, L, basis)
E = hf(xp, kpt)
print("E:", E)
```

## Requirements

        jax

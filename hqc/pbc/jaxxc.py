import jax
import jax.numpy as jnp
import jax_xc
from jax.config import config   
config.update("jax_enable_x64", True)

def rho(r):
    """Electron number density. We take gaussian as an example.

    A function that takes a real coordinate, and returns a scalar
    indicating the number density of electron at coordinate r.

    Args:
    r: a 3D coordinate.
    Returns:
    rho: If it is unpolarized, it is a scalar.
        If it is polarized, it is a array of shape (2,).
    """
    return jnp.prod(jax.scipy.stats.norm.pdf(r, loc=0, scale=1))

# create a density functional
gga_xc_pbe = jax_xc.gga_x_pbe(polarized=False)

# a grid point in 3D
r = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)

# pass rho and r to the functional to compute epsilon_xc (energy density) at r.
# corresponding to the 'zk' in libxc
print(rho(r))
epsilon_xc_r = gga_xc_pbe(rho, r)
print(epsilon_xc_r)
print(epsilon_xc_r.dtype)

import jax
import jax.numpy as jnp

from hqc.pbc.lcao import make_lcao
from hqc.pbc.slater import make_slater

n = 8
rs = 2.0
L = (4/3*jnp.pi*n)**(1/3)
basis = "gth-dzv"
grid_length = 0.12
dft = True
xc = "lda,vwn"
smearing = True
smearing_method = "fermi"
smearing_temperature = 10000

beta = 157888.088922572/smearing_temperature # inverse temperature in unit of 1/Ry
smearing_sigma = 1/beta/2 # temperature in Hartree unit
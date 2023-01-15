import jax
import jax.numpy as jnp
import numpy as np

const = (2 / jnp.pi)**0.75
coeff_sto3g = jnp.array([[3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],
                        [0.16885540, 0.44463454]])
coeff_sto6g = jnp.array([[35.52322122, 0.00916359628],
                        [6.513143725, 0.04936149294],
                        [1.822142904, 0.16853830490],
                        [0.625955266, 0.37056279970],
                        [0.243076747, 0.41649152980],
                        [0.100112428, 0.13033408410]])

def make_ao(basis):
   
    if basis == 'sto3g':
        coeff = coeff_sto3g
    elif basis == 'sto6g':
        coeff = coeff_sto6g
     
    @jax.remat 
    def eval_gto(xp, xe):  
        r = jnp.sum(jnp.square(xe[None, :] - xp), axis=1) # (n_p,)
        gto = const * jnp.einsum('i,i,i...->...', coeff[:, 1], jnp.power(coeff[:, 0], 0.75), \
                jnp.exp(-jnp.einsum('i,...->i...', coeff[:, 0], r))).reshape(-1)  # (n_p,)
        return val

    return eval_gto

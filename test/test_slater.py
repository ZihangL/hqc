import jax
import time
import functools
import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import Sequence, Optional
jax.config.update("jax_enable_x64", True)

from hqc.pbc.lcao import make_lcao
from hqc.pbc.slater import make_slater

def kpoints(dim, Gmax):
    """
        Compute all the integer k-mesh indices (n_1, ..., n_dim) in spatial
    dimention `dim`, whose length do not exceed `Gmax`.
    """
    n = np.arange(-Gmax, Gmax+1)
    nis = np.meshgrid(*( [n]*dim ))
    G = np.array([ni.flatten() for ni in nis]).T
    G2 = (G**2).sum(axis=-1)
    G = G[(G2<=Gmax**2) * (G2>0)]
    return jnp.array(G)

def Madelung(dim, kappa, G):
    """
        The Madelung constant of a simple cubic lattice of lattice constant L=1
    in spatial dimension `dim`, namely the electrostatic potential experienced by
    the unit charge at a lattice site.
    """
    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = jax.scipy.special.erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    return g_k.sum() + g_0 - 2*kappa/jnp.sqrt(jnp.pi)

def psi(rij, kappa, G, forloop=True):
    """
        The electron coordinate-dependent part 1/2 sum_{i}sum_{j neq i} psi(r_i, r_j)
    of the electrostatic energy (per cell) for a periodic system of lattice constant L=1.
        NOTE: to account for the Madelung part `Vconst` returned by the function
    `Madelung`, add the term 0.5*n*Vconst.
    """
    dim = rij.shape[0]

    # Only the nearest neighbor is taken into account in the present implementation of real-space summation.
    dij = jnp.linalg.norm(rij, axis=-1)
    V_shortrange = (jax.scipy.special.erfc(kappa * dij) / dij)

    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = jax.scipy.special.erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    if forloop:
        def _body_fun(i, val):
            return val + g_k[i] * jnp.cos(2*jnp.pi * jnp.sum(G[i]*rij))  
        V_longrange = jax.lax.fori_loop(0, G.shape[0], _body_fun, 0.0) + g_0
    
    else:
        V_longrange = ( g_k * jnp.cos(2*jnp.pi * jnp.sum(G*rij, axis=-1)) ).sum() \
                    + g_0 
     
    potential = V_shortrange + V_longrange
    return potential

@partial(jax.vmap, in_axes=(0, None, None, None, None), out_axes=0)
def potential_energy(x, kappa, G, L, rs):
    """
        Potential energy for a periodic box of size L, only the nontrivial
    coordinate-dependent part. Unit: Ry/rs^2.
        To account for the Madelung part `Vconst` returned by the function `Madelung`,
    add the term n*rs/L*Vconst. See also the docstring for function `psi`.

    INPUTS: 
        x: (n, dim) proton + electron coordinates
    """

    n, dim = x.shape

    x -= L * jnp.floor(x/L)
    i, j = jnp.triu_indices(n, k=1)
    rij = ( (x[:, None, :] - x)[i, j] )/L
    rij -= jnp.rint(rij)
    
    Z = jnp.concatenate([jnp.ones(n//2), -jnp.ones(n//2)])

    #Zij = (Z[:, None] * Z)[i,j]
    # return 2*rs/L * jnp.sum( Zij * jax.vmap(psi, (0, None, None), 0)(rij, kappa, G) )

    total_charge = (Z[:, None]+Z )[i, j]

    v = jax.vmap(psi, (0, None, None), 0)(rij, kappa, G)

    v_pp = jnp.sum(jnp.where(total_charge==2, v, jnp.zeros_like(v)))
    v_ep = -jnp.sum(jnp.where(total_charge==0, v, jnp.zeros_like(v)))
    v_ee = jnp.sum(jnp.where(total_charge==-2, v, jnp.zeros_like(v)))

    return 2*rs/L*v_pp , 2*rs/L * v_ep , 2*rs/L*v_ee

@partial(jax.vmap, in_axes=(None, 0, None, None, None, None), out_axes=(0, 0, 0, 0, 0))
def observables(xp, xe, mo_coeff, n, rs, logpsi_grad_laplacian):
    L = (4/3*jnp.pi*n)**(1/3)
    
    Gmax = 15
    kappa = 10
    G = kpoints(3, Gmax)
    Vconst = n * rs/L * Madelung(3, kappa, G)

    vpp, vep, vee = potential_energy(jnp.array([jnp.concatenate([xp, xe], axis=0)]), kappa, G, L, rs)
    vpp += Vconst
    vee += Vconst

    grad, laplacian = logpsi_grad_laplacian(xe, xp, mo_coeff)
    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

    Eloc = kinetic + vep + vee

    return Eloc.real, kinetic.real, vpp, vep, vee

@partial(jax.jit, static_argnums=(0, 3))
def mcmc(logp_fn, x, key, mc_steps, mc_width):
    """
        MCMC sampling of x from logp_fn.
        x has shape (batch, n, dim).
    """
    def step(i, state):
        x, logp, key, num_accepts = state
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        x_proposal = x + mc_width * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[:, None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        num_accepts += accept.sum()

        return x_new, logp_new, key, num_accepts

    logp_init = logp_fn(x)
    x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x, logp_init, key, 0.))
    batchsize = x.shape[0]
    accept_rate = num_accepts / (mc_steps * batchsize) 
    return x, accept_rate

@partial(jax.jit, static_argnums=0)
def mcmc_ebes(logp_fn, x_init, key, mc_steps, mc_width,
              logp_init=None):
    """
        Markov Chain Monte Carlo (MCMC) sampling algorithm
        with electron-by-electron sampling (EBES).

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (..., n, dim).
        x_init: initial value of x, with shape (..., n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

        logp_init: initial logp (...,)

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """

    def single_step(ii, state):
        x, logp, key, accept_rate = state
        key, key_proposal, key_accept = jax.random.split(key, 3)

        batchshape = x.shape[:-2]
        dim = x.shape[-1]
        x_move = jax.random.normal(key_proposal, (*batchshape, dim))
        x_proposal = x.at[..., ii, :].add(mc_width * x_move)
        logp_proposal = logp_fn(x_proposal)  # batchshape

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio
        accept_rate += accept.mean()

        x_new = jnp.where(accept[..., None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)

        return x_new, logp_new, key, accept_rate

    def step(i, state):
        x, logp, key, accept_rate = state

        n = x.shape[-2]
        x_new, logp_new, key, accept_rate = jax.lax.fori_loop(0, n, single_step,
                                                                           (x, logp, key, accept_rate))
        return x_new, logp_new, key, accept_rate 

    if logp_init is None:
        logp_init = logp_fn(x_init)

    x, logp, key, accept_rate = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
    n = x.shape[-2]
    accept_rate /= mc_steps * n
    return x, accept_rate

def sample_x_mcmc(key, xp, xe, logpsi2, mo_coeff, mc_steps, mc_width, L):
    """
        Sample electron coordinates from the ground state wavefunction.
    """
    key, key_mcmc = jax.random.split(key)
    logpsi2_mcmc_novmap = lambda x: logpsi2(x, xp, mo_coeff)
    logpsi2_mcmc = jax.vmap(logpsi2_mcmc_novmap)
    xe, acc = mcmc_ebes(logpsi2_mcmc, xe, key_mcmc, mc_steps, mc_width)
    xe -= L * jnp.floor(xe/L)
    return key, xe, acc

def hf_wfn_mcmc(n, rs, xp, L, logpsi2, logpsi_grad_laplacian, mo_coeff, batchsize, basis, grid_length, mc_steps, mc_width):
    key = jax.random.PRNGKey(42)
    xe = jax.random.uniform(key, (batchsize, n, 3), minval=0., maxval=L)

    for _ in range(5):
        key, xe, acc = sample_x_mcmc(key, xp, xe, logpsi2, mo_coeff, mc_steps, mc_width, L)
        e, k, vpp, vep, vee = observables(xp, xe, mo_coeff, n, rs, logpsi_grad_laplacian)

        e_mean = e.mean()/rs**2/n 
        e_err = e.std()/jnp.sqrt(batchsize)/rs**2/n

        k_mean = k.mean()/rs**2/n 
        k_err = k.std()/jnp.sqrt(batchsize)/rs**2/n 

        vep_mean = vep.mean()/rs**2/n 
        vep_err = vep.std()/jnp.sqrt(batchsize)/rs**2/n 

        vee_mean = vee.mean()/rs**2/n 
        vee_err = vee.std()/jnp.sqrt(batchsize)/rs**2/n 

        vpp_mean = vpp.mean()/rs**2/n
        vpp_err = vpp.std()/jnp.sqrt(batchsize)/rs**2/n 

        print ("e, k, vep, vee, vpp, acc", 
                      e_mean, "+/-", e_err, 
                      k_mean, "+/-", k_err, 
                      vep_mean, "+/-", vep_err, 
                      vee_mean, "+/-", vee_err, 
                      vpp_mean, "+/-", vpp_err, 
                      acc)

def logdet_matmul(xs: Sequence[jnp.ndarray],
                  logw: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Combines determinants and takes dot product with weights in log-domain.
    We use the log-sum-exp trick to reduce numerical instabilities.
    Args:
        xs: FermiNet orbitals in each determinant. Either of length 1 with shape
        (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
        (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
        determinants are factorised into block-diagonals for each spin channel).
        w: weight of each determinant. If none, a uniform weight is assumed.
    Returns:
        sum_i exp(logw_i) D_i in the log domain, where logw_i is the log-weight of D_i, the i-th
        determinant (or product of the i-th determinant in each spin channel, if
        full_det is not used).
    """
    # Special case to avoid taking log(0) if any matrix is of size 1x1.
    # We can avoid this by not going into the log domain and skipping the
    # log-sum-exp trick.
    det1 = functools.reduce(
        lambda a, b: a * b,
        [x.reshape(-1) for x in xs if x.shape[-1] == 1],
        1
    )

    # Compute the logdet for all matrices larger than 1x1
    sign_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
        (1, 0)
    )

    if logw is not None:
        logdet = logw + logdet

    # log-sum-exp trick
    maxlogdet = jnp.max(logdet)
    det = sign_in * det1 * jnp.exp(logdet - maxlogdet)
    result = jnp.sum(det)

    sign_out = result/jnp.abs(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return sign_out, log_out

def make_logpsi_hf(hf_orbitals):
    """
        Make the logpsi electron wavefunction.
        hf_orbitals is a static function generated by hf.
        kpt is fixed at gamma point.
    """

    def logpsi(x, s, mo_coeff):
        """
            Generic function that computes ln Psi(x) given momenta `k` and proton position
        `s`, a set of electron coordinates `x`
        INPUT:
            x: (n, dim)     
            s: (n, dim)
            mo_coeff: coefficient of hf orbitals on atomic orbitals (n_ao, n_mo)
        OUTPUT:
            a single complex number ln Psi(x), given in the form of a 2-tuple (real, imag).
        """
        D_up, D_dn =  hf_orbitals(s, x, mo_coeff)
        phase, logabsdet = logdet_matmul([D_up[None, :, :], D_dn[None, :, :]])
        log_phi = logabsdet + jnp.log(phase)
        return jnp.stack([log_phi.real, log_phi.imag])

    return logpsi

def make_logpsi2(logpsi):
    
    def logpsi2(x, s, mo_coeff):
        """
            logp = logpsi + logpsi* = 2 Re logpsi
        Input:
            x: (n, dim)
            s: (n, dim)
            mo_coeff: (n_ao, n_mo)
        Output:
            logp: float
        """
        return 2 * logpsi(x, s, mo_coeff)[0]
    
    return logpsi2

def make_logpsi_grad_laplacian(logpsi):

    def logpsi_grad_laplacian(x, s, mo_coeff):
        """
            Computes the gradient and laplacian of logpsi w.r.t. electron coordinates x.
        The final result is in complex form.

        Relevant dimensions: (after vmapped)

        INPUT:
            x: (n, dim)  s: (n, dim)
        OUTPUT:
            grad: (n, dim)   laplacian: float
        """

        grad = jax.jacrev(logpsi)(x, s, mo_coeff)
        grad = grad[0] + 1j * grad[1]

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        grad_logpsi = jax.jacrev(lambda x: logpsi(x.reshape(n, dim), s, mo_coeff))

        def _laplacian(x):
            def body_fun(i, val):
                _, tangent = jax.jvp(grad_logpsi, (x,), (eye[i],))
                return val + tangent[0, i] + 1j * tangent[1, i]
            eye = jnp.eye(x.shape[0])
            laplacian = jax.lax.fori_loop(0, x.shape[0], body_fun, 0.+0.j)
            return laplacian

        laplacian = _laplacian(x_flatten)

        return grad, laplacian

    return logpsi_grad_laplacian

def test_slater_hf(xp, rs, basis, rcut, grid_length, smearing, sigma, max_cycle):
    n = xp.shape[0]
    batchsize = 1024
    mc_steps = 100
    mc_width = 0.02

    L = (4/3*jnp.pi*n)**(1/3)
   
    print("------- system information -------")
    print(f"n: {n}")
    print(f"rs: {rs} (Bohr)")
    print(f"hf basis: {basis}")
    print(f"L: {L}")
    print(f"xp: \n{xp}")
    print(f"mc_steps: {mc_steps}")
    print(f"mc_width: {mc_width}")

    print("------- HF and MC results -------")

    lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=False, smearing=smearing, smearing_sigma=sigma, max_cycle = max_cycle)
    mo_coeff, bands, e = lcao(xp)

    print("e_hf (k+vep+vee in Ry):", e)

    hf_orbitals = make_slater(n, L, rs, basis=basis, groundstate=True)

    logpsi = make_logpsi_hf(hf_orbitals)
    logpsi2 = make_logpsi2(logpsi)
    force_fn_e = jax.grad(logpsi2)
    logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi)

    hf_wfn_mcmc(n, rs, xp, L, logpsi2, logpsi_grad_laplacian, mo_coeff, batchsize, basis, grid_length, mc_steps, mc_width)

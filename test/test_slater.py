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

from tools.vmc import sample_x_mcmc
from tools.observables import observables
from tools.logpsi import make_logpsi_hf, make_logpsi2, make_logpsi_grad_laplacian


def hf_wfn_mcmc(n, rs, xp, L, logpsi2, logpsi_grad_laplacian, mo_coeff, batchsize, mc_steps, mc_width):
    key = jax.random.PRNGKey(42)
    xe = jax.random.uniform(key, (batchsize, n, 3), minval=0., maxval=L)

    for ii in range(10):
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

        print ("steps, e, k, vep, vee, vpp, acc", 
                      ii, 
                      e_mean, "+/-", e_err, 
                      k_mean, "+/-", k_err, 
                      vep_mean, "+/-", vep_err, 
                      vee_mean, "+/-", vee_err, 
                      vpp_mean, "+/-", vpp_err, 
                      acc)

def test_slater_hf(xp, rs, basis, rcut, grid_length, smearing, sigma, max_cycle):
    n = xp.shape[0]
    batchsize = 256
    mc_steps = 300
    mc_width = 0.04

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

    print("e_hf per atom (k+vep+vee in Ry):", e/n)

    hf_orbitals = make_slater(n, L, rs, basis=basis, rcut=rcut, groundstate=True)

    logpsi = make_logpsi_hf(hf_orbitals)
    logpsi2 = make_logpsi2(logpsi)
    logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi)

    hf_wfn_mcmc(n, rs, xp, L, logpsi2, logpsi_grad_laplacian, mo_coeff, batchsize, mc_steps, mc_width)

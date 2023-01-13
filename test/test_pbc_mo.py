from config import *
from pyscf.pbc import gto, scf
from hyqc.pbc.mo import make_hf

def zerovee(L, xp, basis):

    """
        hartree fock without Vee pyscf benchmark.

        OUTPUT:
            energy without Vpp, unit: Ry
    """
    Ry = 2
    n = xp.shape[0]
    cell = np.eye(3)
    n_alpha = n_beta = int(n/2)
    gtocell = gto.Cell()
    gtocell.unit = 'B'
    gtocell.atom = []
    for i in range(n):
        gtocell.atom.append(['H', tuple(xp[i])])
    gtocell.spin = 0
    gtocell.basis = basis
    gtocell.a = cell*L
    gtocell.build()

    kpts = gtocell.make_kpts([1,1,1],scaled_center=[0,0,0])
    kmf = scf.khf.KRHF(gtocell, kpts=kpts)
    kmf.verbose = 0
    kmf.max_cycle = 1
    kmf.get_veff = lambda *args: np.zeros(kmf.get_hcore().shape)
    kmf.kernel()

    ovlp = kmf.get_ovlp()
    K = scf.hf.get_t(kmf.cell, kpt=kmf.kpts)[0]
    V = scf.hf.get_pp(kmf.cell, kpt=kmf.kpts[0])
    Hcore = scf.hf.get_hcore(kmf.cell, kpt=kmf.kpts)[0]
    
    c2 = kmf.mo_coeff[0]
    return Ry*(kmf.e_tot - kmf.energy_nuc())

def test_pbc_mo():
    rtol = 1e-4
    basis = "gth-szv"
    n, dim = 4, 3
    rs = 1.25
    cell = np.eye(dim)
    L = (4/3*jnp.pi*n)**(1/3)*rs
    key = jax.random.PRNGKey(42)
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    # energy test
    hf = make_hf(n, L, basis)
    hy_E = hf(xp)
    pyscf_E = zerovee(L, xp, basis)

    print("E:\n", hy_E, "\npyscf E:\n", pyscf_E)
    assert np.allclose(hy_E, pyscf_E, rtol=rtol)

    # jit, vmap, grad test
    jax.jit(jax.grad(hf))(xp)
    xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
    jax.vmap(hf)(xp2)


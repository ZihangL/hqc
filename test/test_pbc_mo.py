from config import *
from pyscf.pbc import gto, scf
from hqc.pbc.mo import make_hf

def pyscf_hf(L, xp, basis, kpt):

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

    kpts = [kpt.tolist()]
    # kpts = gtocell.make_kpts([1,1,1],scaled_center=[0,0,0])
    kmf = scf.khf.KRHF(gtocell, kpts=kpts)
    kmf.verbose = 0
    kmf.kernel()

    # ovlp = kmf.get_ovlp()
    # T = scf.hf.get_t(kmf.cell, kpt=kmf.kpts)[0]
    # V = scf.hf.get_pp(kmf.cell, kpt=kmf.kpts[0])
    # Hcore = scf.hf.get_hcore(kmf.cell, kpt=kmf.kpts)[0]
    # c2 = kmf.mo_coeff[0]
    # dm = kmf.make_rdm1()
    # print("pyscf overlap:\n", ovlp)
    # print("pyscf T:\n", T)
    # print("pyscf V:\n", V)
    # print("pyscf densigy matrix:\n", dm)

    return Ry*(kmf.e_tot - kmf.energy_nuc())

def test_pbc_mo():
    rs = 1.25
    n, dim = 4, 3
    k0 = jnp.array([0,0,0])
    L = (4/3*jnp.pi*n)**(1/3)*rs
    key = jax.random.PRNGKey(42)
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    key = jax.random.PRNGKey(43)
    twist = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)

    basis_set = ['gth-szv', 'gth-dzv']
    for basis in basis_set:
        print("basis:", basis)

        # PBC energy test
        hf = make_hf(n, L, basis)
        E = hf(xp, k0)
        E_pyscf = pyscf_hf(L, xp, basis, k0)
        print("\nE:\n", E, "\npyscf E:\n", E_pyscf)
        assert np.allclose(E, E_pyscf, rtol=1e-4)

        # TBC energy test
        E_twist = hf(xp, twist)
        E_pyscf_twist = pyscf_hf(L, xp, basis, twist)
        print("E:\n", E_twist, "\npyscf E:\n", E_pyscf_twist)
        assert np.allclose(E_twist, E_pyscf_twist, rtol=1e-4)

        # jit, vmap, grad test
        # jax.jit(jax.grad(hf))(xp, k0)
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
        jax.vmap(hf, (0, None), 0)(xp2, k0)
        kpts = jnp.concatenate([k0, k0]).reshape(2, dim)
        jax.vmap(hf, (None, 0), 0)(xp, kpts)


from config import *
from pyscf.pbc import gto
from hqc.pbc.ao import gen_lattice, make_ao

def pyscf_eval_ao(L, xp, xe, basis, kpt):
    p = xp.shape[0]
    cell = gto.Cell()
    cell.unit = 'B'
    for ip in range(p):
        cell.atom.append(['H', tuple(xp[ip])])
    cell.spin = 0
    cell.basis = basis
    cell.a = np.eye(3) * L
    cell.build()
    kpts = [kpt.tolist()]
    ao_value = cell.pbc_eval_ao("GTOval_sph", xe, kpts=kpts)[0]
    return ao_value

def test_pbc_ao():
	n, dim = 14, 3
	rs = 1.25
	cell = np.eye(3)
	k0 = jnp.array([0,0,0])
	L = (4/3*jnp.pi*n)**(1/3)*rs
	key = jax.random.PRNGKey(42)
	xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
	key = jax.random.PRNGKey(43)
	xe = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
	key = jax.random.PRNGKey(44)
	twist = jax.random.uniform(key, (dim,), minval=-jnp.pi/L, maxval=jnp.pi/L)

	basis_set = ['gth-szv', 'gth-dzv']
	for basis in basis_set:
	
		# pbc pyscf benchmark
		pyscf_ao = pyscf_eval_ao(L, xp, xe, basis, k0)
		lattice = gen_lattice(cell, L, rcut=30)
		eval_ao = jax.vmap(make_ao(lattice, basis), (None, 0, None), 0)
		ao = eval_ao(xp, xe, k0)
		assert np.allclose(pyscf_ao, ao)

		# pbc test
		image = np.random.randint(-2, 3, size=(n, dim)).dot(cell.T)*L
		pbc_ao = eval_ao(xp, xe+image, k0)
		assert np.allclose(ao, pbc_ao, rtol=1e-3)

		# twist pyscf benchmark
		pyscf_ao_twist = pyscf_eval_ao(L, xp, xe, basis, twist)
		ao_twist = eval_ao(xp, xe, twist)
		assert np.allclose(pyscf_ao_twist, ao_twist)

		# jit, vmap, test
		jax.jit(eval_ao)(xp, xe, k0)
		xe2 = jnp.concatenate([xe, xe]).reshape(2, n, dim)
		jax.vmap(eval_ao, (None, 0, None), 0)(xp, xe2, k0)
		xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
		jax.vmap(eval_ao, (0, None, None), 0)(xp2, xe, k0)
		kpts = jnp.concatenate([k0, k0]).reshape(2, dim)
		jax.vmap(eval_ao, (None, None, 0), 0)(xp, xe, kpts)

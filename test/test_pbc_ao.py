from config import *
from pyscf.pbc import gto
from hqc.pbc.ao import gen_lattice, make_ao

def pyscf_eval_ao(L, xp, xe, basis):
    p = xp.shape[0]
    cell = gto.Cell()
    cell.unit = 'B'
    for ip in range(p):
        cell.atom.append(['H', tuple(xp[ip])])
    cell.spin = 0
    cell.basis = basis
    cell.a = np.eye(3) * L
    cell.build()
    kpts = cell.make_kpts([1, 1, 1], scaled_center=[0., 0., 0.])
    ao_value = cell.pbc_eval_ao("GTOval_sph", xe, kpts=kpts)[0]
    return ao_value

def test_pbc_ao():
	basis = 'gth-szv'
	n, dim = 14, 3
	rs = 1.25
	cell = np.eye(3)
	L = (4/3*jnp.pi*n)**(1/3)*rs
	key = jax.random.PRNGKey(42)
	xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
	key = jax.random.PRNGKey(43)
	xe = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
	
	# pyscf benchmark
	pyscf_ao = pyscf_eval_ao(L, xp, xe, basis)
	lattice = gen_lattice(cell, L, rcut=30)
	eval_ao = jax.vmap(make_ao(lattice, basis), (None, 0), 0)
	ao = eval_ao(xp, xe)
	assert np.allclose(pyscf_ao, ao)

	# pbc test
	image = np.random.randint(-2, 3, size=(n, dim)).dot(cell.T)*L
	pbc_ao = eval_ao(xp, xe+image)
	print("pbc ao:\n", pbc_ao)
	assert np.allclose(ao, pbc_ao, rtol=1e-3)

	# jit, vmap, test
	jax.jit(eval_ao)(xp, xe)
	xe2 = jnp.concatenate([xe, xe]).reshape(2, n, dim)
	jax.vmap(eval_ao, (None, 0), 0)(xp, xe2)
	xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
	jax.vmap(eval_ao, (0, None), 1)(xp2, xe)

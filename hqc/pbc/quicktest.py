import jax.numpy as jnp

def density_matrix(mo_coeff):
    """
        density matrix for closed shell system.
    """
    n = mo_coeff.shape[0]
    dm = 2*jnp.einsum('im,jm->ij', mo_coeff[:,:n//2], mo_coeff.conjugate()[:,:n//2])
    return dm

def energy(FH, dm):
    return 0.5*jnp.einsum('pq,qp', FH, dm)

mo1 = jnp.array([[-0.02244608+0.87268981j,  3.35738351+0.00583435j],
                [-0.27049585+0.83001403j, -3.16629417-1.11653004j]])
mo2 = jnp.array([[ 0.87272749+0.02092916j,  3.35642351-0.08049149j],
                [ 0.83048293+0.26905274j, -3.19395518-1.03475021j]])

dm1 = density_matrix(mo1)
dm2 = density_matrix(mo2)

FH1 = jnp.array([[0.4354686 -8.30816283e-10j, 0.22163022-8.54360418e-02j],
                [0.22163022+8.54360417e-02j, 0.4354686 -2.94185774e-09j]])
FH2 = jnp.array([[ 0.10793691+5.58417193e-17j, -0.08744799+2.29506533e-02j],
                [-0.08744799-2.29506533e-02j,  0.10793691-3.77315210e-16j]])

e1 = energy(FH1, dm1)
e2 = energy(FH2, dm2)

print(e1*2)
print(e2*2)
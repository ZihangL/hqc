from typing import Union
import jax.numpy as jnp
import numpy as np

Ry = 2

class hydrogen:

    def __init__(self, length: float, position_nuc: Union[np.ndarray,jnp.ndarray], rcut:float, gcut:float) -> None:
        """
        Args:
            length: length of hydrogen cell (unit: Bohr).
            position_nuc: (n_nuc, 3) array, positions of hydrogen nuclei.
        """
        self.length = length
        self.R = position_nuc
        self.n_H = position_nuc.shape[0]
        self.dim_mat = 2 * self.n_H
        self.coeff_gthdzv = jnp.array([[8.3744350009, -0.0283380461, 0.0000000000],
                                       [1.8058681460, -0.1333810052, 0.0000000000],
                                       [0.4852528328, -0.3995676063, 0.0000000000],
                                       [0.1658236932, -0.5531027541, 1.0000000000]])
        
        # cutoff in real space
        self.rcut = rcut # 14.486425944846197
        gcut = self.rcut / self.length
        grange = jnp.arange(-self.rcut // self.length, self.rcut // self.length + 1)
        x_grid, y_grid, z_grid = jnp.meshgrid(grange, grange, grange) 
        disk = (x_grid**2 + y_grid**2 + z_grid**2) <= gcut**2
        x, y, z = jnp.where(disk == True)
        self.cell = jnp.array([x_grid[x, y, z], y_grid[x, y, z], z_grid[x, y, z]]).T  # (n_cell, 3)
        self.cell = self.length * self.cell

        # cutoff in reciprocal space
        self.gcut = gcut
        self.glength = 2 * jnp.pi / self.length
        gcut = self.gcut / self.glength
        grange = jnp.arange(-self.gcut // self.glength, self.gcut // self.glength + 1)
        x_grid, y_grid, z_grid = jnp.meshgrid(grange, grange, grange)
        disk = (x_grid**2 + y_grid**2 + z_grid**2) <= gcut**2
        x, y, z = jnp.where(disk == True)
        self.gcell = jnp.array([x_grid[x, y, z], y_grid[x, y, z], z_grid[x, y, z]]).T  # (n_cell, 3)
        index = self.gcell.shape[0]//2
        self.gcell = jnp.delete(self.gcell[:], index, 0) # remove G = 0
        self.gcell = self.glength * self.gcell

        self.alpha = self.coeff_gthdzv[:, 0]  # (4,)
        self.__sum_alpha = self.alpha[:, None] + self.alpha[None, :]  # (4, 4)
        self.__pro_alpha = jnp.einsum('i,j->ij', self.alpha, self.alpha)  # (4, 4)
        self.__alpha2 = self.__pro_alpha / self.__sum_alpha  # (4, 4)
        self.__coeff = self.coeff_gthdzv[:, 1:3].T  # (2, 4)
        self.__Rnc = self.R[:, None, :] + self.cell[None, :, :]
        self.__Rn_Rnc = self.__Rnc[:, None, :, :] - self.R[None, :, None, :]
        self.__Rminjc = (self.R[:, None, None, None, None, :]*self.alpha[None, :, None, None, None, None] \
            +self.__Rnc[None, None, :, None, :, :]*self.alpha[None, None, None, :, None, None]) \
            /self.__sum_alpha[None, :, None, :, None, None]
        self.__G = jnp.linalg.norm(self.gcell, axis=1)
        self.__VG = 4 * jnp.pi / length ** 3 / jnp.square(self.__G)
        self.__exp1 = jnp.exp(-1j*jnp.einsum('gx,kminjcx->kgminjc', self.gcell, 
            self.R[:, None, None, None, None, None, :]-self.__Rminjc[None, :, :, :, :, :, :])) # index GNminjc
        self.__exp2 = jnp.exp(-jnp.square(self.__G)[:, None, None]/4/self.__sum_alpha[None, :, :])

    def overlap(self) -> jnp.ndarray:
        """
            Returns orbital ovarlap matrix of shape (n_ao, n_ao)
        """
        self.__ovlp = 2**1.5*jnp.einsum('pi,qj,ij,ijmnc->mpicnqj', self.__coeff, self.__coeff, 
                jnp.power(self.__pro_alpha, 0.75)/jnp.power(self.__sum_alpha, 1.5), 
                jnp.exp(-jnp.einsum('ij,mnc->ijmnc', self.__alpha2, 
                jnp.square(jnp.linalg.norm(self.__Rn_Rnc, axis=3)))))
        self.ovlp = jnp.reshape(jnp.einsum('mpicnqj->mpnq', self.__ovlp), (self.dim_mat, self.dim_mat))
        return self.ovlp

    def kinetic(self) -> jnp.ndarray:
        """
            Returns kinetic matrix of shape (n_ao, n_ao)
        """
        self.K = jnp.reshape(jnp.einsum('mpicnqj,ij,ijmnc->mpnq', self.__ovlp, 
                self.__alpha2, 3-2*jnp.einsum('ij,mnc->ijmnc', self.__alpha2, 
                jnp.square(jnp.linalg.norm(self.__Rn_Rnc, axis=3)))), (self.dim_mat, self.dim_mat))
        return self.K

    def potential(self) -> jnp.ndarray:
        """
            Returns potential matrix of shape (n_ao, n_ao)
        """
        self.V = jnp.reshape(jnp.einsum('g,mpicnqj,kgminjc,gij->mpnq', self.__VG, 
                self.__ovlp, self.__exp1, self.__exp2), (self.dim_mat, self.dim_mat))
        return -self.V.real

    def hcore(self) -> jnp.ndarray:
        """
            Returns core of hamiltonian matrix of shape (n_ao, n_ao)
        """
        return self.kinetic() + self.potential()

    def kernel(self) -> jnp.ndarray:
        """
            Returns coefficients of molecular orbitals of shape (n_ao, n_mo)
        """
        ovlp = self.overlap()
        hcore = self.hcore()

        w, u = jnp.linalg.eigh(ovlp)
        v = jnp.dot(u, np.diag(w**(-0.5)))
        f1 = jnp.einsum('pq,qr,rs->ps', v.T.conjugate(), hcore, v)
        w1, c1 = jnp.linalg.eigh(f1)
        c2 = jnp.dot(v, c1)
        idx = w1.argsort()
        w1 = w1[idx]
        c2 = c2[:, idx]
        w = w1[0:self.n_H//2]
        c = c2[:, 0:self.n_H//2]
        e = 2 * jnp.sum(w) # just for non-interacting system
        # e = jnp.sum(w) + jnp.einsum('pq,pe,qe', hcore, c.conjugate(), c)
        return Ry * e

if __name__ == "__main__":
    from pyscf.pbc import scf
    from zerovee import zerovee
    
    n, dim = 2, 3
    L, d = 10.0, 1.4
    center = np.array([L/2, L/2, L/2])
    offset = np.array([[d/2, 0., 0.],
                    [-d/2, 0., 0.]])
    xp = center + offset

    pyscf_hf = zerovee(L, xp)
    pyscf_ovlp = pyscf_hf.kmf.get_ovlp()[0].real
    pyscf_K = scf.hf.get_t(pyscf_hf.cell, kpt=pyscf_hf.kpts)[0].real
    pyscf_Vep = scf.hf.get_pp(pyscf_hf.cell, kpt=pyscf_hf.kpts[0]).real
    pyscf_E = pyscf_hf.E()

    hf = hydrogen(L, xp)
    hf_ovlp = hf.overlap()
    hf_K = hf.kinetic()
    hf_Vep = hf.potential()
    E = hf.kernel()+pyscf_hf.Vpp()



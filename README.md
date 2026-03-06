# HQC - Hydrogen Quantum Chemistry

[![PyPI version](https://img.shields.io/pypi/v/hqc.svg)](https://pypi.org/project/hqc/)
[![Python versions](https://img.shields.io/pypi/pyversions/hqc.svg)](https://pypi.org/project/hqc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Quantum chemistry calculations for Hydrogen systems using JAX. HQC provides GPU-accelerated Hartree-Fock and DFT calculations for periodic and isolated Hydrogen systems.

## Features

- **GPU-Accelerated**: Built on JAX for high-performance computing on GPUs
- **Periodic Boundary Conditions**: Full support for PBC calculations using GPW (Gaussian and Plane Waves) method
- **Multiple Methods**: Hartree-Fock and DFT (LDA, GGA) implementations
- **K-point Support**: K-point sampling for periodic systems
- **Flexible Basis Sets**: CP2K-format basis sets (GTH pseudopotentials)
- **Temperature Effects**: Smearing methods for finite-temperature calculations
- **Automatic Differentiation**: JAX-based automatic differentiation for forces and gradients

## Installation

### From PyPI

```bash
pip install hqc
```

### From Source

```bash
git clone https://code.itp.ac.cn/lzh/hydrogen-qc.git
cd hydrogen-qc
pip install -e .
```

### Requirements

- Python >= 3.9
- JAX >= 0.4.0
- JAXlib >= 0.4.0
- NumPy >= 1.20.0
- PySCF >= 2.0.0 (for testing and comparison)

## Quick Start

### Periodic Systems (PBC)

#### Basic DFT Calculation

```python
import jax
import jax.numpy as jnp
from hqc.pbc.lcao import make_lcao

# System parameters
n = 8  # number of electrons
rs = 1.25  # Wigner-Seitz radius
L = (4/3*jnp.pi*n)**(1/3) * rs  # box size
basis = 'gth-dzv'

# Generate random positions
key = jax.random.PRNGKey(42)
xp = jax.random.uniform(key, (n, 3), minval=0., maxval=L)

# Create LCAO solver
lcao = make_lcao(n, L, rs, basis, dft=True)

# Run calculation
mo_coeff, bands = lcao(xp)
print("Band energies:", bands)
```

#### Finite Temperature Calculation

```python
from hqc.pbc.lcao import make_lcao

T = 10000  # Temperature in Kelvin
beta = 157888.088922572/T  # inverse temperature (1/Ry)
sigma = 1/beta/2  # smearing parameter (Hartree)

lcao = make_lcao(
    n, L, rs, basis,
    dft=True,
    smearing=True,
    smearing_sigma=sigma
)

mo_coeff, bands = lcao(xp)
```

#### Potential Energy Surface

```python
from hqc.pbc.pes import make_pes

# Create PES calculator
pes = make_pes(n, L, rs, basis, dft=True)

# Calculate energy
energy = pes(xp)
print(f"Total energy: {energy} Ry")
```

### Isolated Molecules (GTO)

#### Hartree-Fock Calculation

```python
import jax.numpy as jnp
from hqc.gto.solver import make_solver

# H2 molecule
atom_charges = jnp.array([1.0, 1.0])
n_electrons = 2

# Create solver (preprocessing: basis loading, occupation numbers, etc.)
hf = make_solver(atom_charges, n_electrons, basis='gth-szv', use_jit=True)

# Calculate for different geometries (efficient repeated calls)
for distance in [1.0, 1.2, 1.4, 1.6, 1.8]:
    positions = jnp.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    result = hf(positions)
    print(f"d={distance:.2f} Bohr: E={result['energy']:.6f} Ha")
```

#### Geometry Optimization Example

```python
from hqc.gto.solver import make_solver
import jax

# Setup
atom_charges = jnp.array([1.0, 1.0])
n_electrons = 2
hf = make_solver(atom_charges, n_electrons, basis='gth-szv', use_jit=True)

# Energy function for optimization
def energy_fn(positions):
    result = hf(positions)
    return result['energy']

# Compute gradient
grad_fn = jax.grad(energy_fn)

# Initial geometry
positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

# Get energy and gradient
energy = energy_fn(positions)
gradient = grad_fn(positions)
print(f"Energy: {energy:.6f} Ha")
print(f"Gradient:\n{gradient}")
```

## Modules

### Periodic Systems (hqc.pbc)

- **hqc.pbc.gto**: Gaussian-type orbital evaluation for periodic systems
- **hqc.pbc.lcao**: Hartree-Fock and DFT solvers (LCAO method)
- **hqc.pbc.pes**: Potential energy surface calculations
- **hqc.pbc.overlap**: Basis set overlap integrals
- **hqc.pbc.slater**: Slater determinant for LCAO orbitals
- **hqc.pbc.solver**: Low-level solver with detailed output (entropy, energy components)
- **hqc.pbc.potential**: Electron-electron and electron-ion potentials

### Isolated Molecules (hqc.gto)

- **hqc.gto.solver**: High-level Hartree-Fock solver interface
- **hqc.gto.integral**: Vectorized Gaussian integral evaluation
- **hqc.gto.scf**: Self-consistent field iteration with DIIS
- **hqc.gto.gto**: Atomic orbital evaluation for wavefunction reconstruction
- **hqc.gto.boys**: Boys function for nuclear attraction integrals

## Basis Sets

HQC uses CP2K-format basis sets with GTH pseudopotentials. Basis files are located in `hqc/basis/` and are automatically discovered by name.

Available basis sets include:
- `gth-szv`, `gth-dzv`, `gth-tzv`
- `gth-dzvp`, `gth-tzvp`, `gth-qzv3p`
- And many more in `hqc/basis/gth-raw/`

Example basis file format (`gth-dzv.dat`):
```
#BASIS SET
H DZV-GTH
  1
  1  0  0  4  2
        8.3744350009  -0.0283380461   0.0000000000
        1.8058681460  -0.1333810052   0.0000000000
        0.4852528328  -0.3995676063   0.0000000000
        0.1658236932  -0.5531027541   1.0000000000
#
```

## Testing

Run the test suite with pytest:

```bash
pip install pytest
pytest test/
```

Compare with PySCF results:

```bash
python -m pytest test/test_pbc_solver.py -v
```

## Development

### Code Formatting

Install development tools:

```bash
pip install black ruff
```

Format code:

```bash
make format  # runs black
make lint    # runs ruff
```

Or manually:

```bash
black .
ruff check --fix .
```

## Documentation

For detailed algorithm documentation, see [doc/solver_algorithm.md](doc/solver_algorithm.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use HQC in your research, please cite:

```bibtex
@software{hqc2025,
  author = {Li, Zihang},
  title = {HQC: Hydrogen Quantum Chemistry with JAX},
  year = {2025},
  url = {https://code.itp.ac.cn/lzh/hydrogen-qc}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.


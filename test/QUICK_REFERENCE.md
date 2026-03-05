# Quick Reference: Improved Test Suite

## Test File Structure

Each test file now follows this pattern:

```python
"""Module docstring explaining what is tested."""

import statements...

@dataclass
class TestConfig:
    """Configuration parameters with defaults."""
    param1: type = default_value
    param2: type = default_value

    @property
    def computed_property(self):
        """Computed values based on parameters."""
        return calculation

class TestClassName:
    """Main test class."""

    @pytest.fixture(scope="class")
    def config(self):
        """Configuration fixture."""
        return TestConfig()

    @pytest.fixture(scope="class")
    def data(self, config):
        """Test data fixture."""
        return generate_data(config)

    def _helper_method(self, ...):
        """Private helper methods start with _"""
        pass

    def test_something(self, config, data):
        """Public test methods."""
        pass

# Legacy functions for backward compatibility
def test_something():
    """Original test function name."""
    test = TestClassName()
    # Call new implementation
```

## Running Tests

### Run All PBC Tests
```bash
pytest test/test_pbc_*.py -v
```

### Run Specific Test File
```bash
pytest test/test_pbc_solver.py -v
```

### Run Specific Test Class
```bash
pytest test/test_pbc_solver.py::TestPBCSolver -v
```

### Run Specific Test Method
```bash
pytest test/test_pbc_solver.py::TestPBCSolver::test_hf_gamma_diis -v
```

### Run Legacy Function
```bash
pytest test/test_pbc_solver.py::test_hf_gamma_diis -v
```

### Run with Output
```bash
pytest test/test_pbc_solver.py -v -s  # -s shows print statements
```

### Run with Coverage
```bash
pytest test/test_pbc_*.py --cov=hqc.pbc --cov-report=html
```

## Key Improvements

### 1. Configuration Management
**Before:**
```python
n = 4
rs = 1.5
L = (4/3*jnp.pi*n)**(1/3)
# ... scattered throughout code
```

**After:**
```python
@dataclass
class TestConfig:
    n: int = 4
    rs: float = 1.5

    @property
    def L(self) -> float:
        return (4/3*jnp.pi*self.n)**(1/3)

config = TestConfig()
# Easy to modify: config = TestConfig(n=8, rs=2.0)
```

### 2. Test Organization
**Before:**
```python
def test_hf_gamma_diis():
    dft = False
    gamma = True
    diis = True
    smearing = False
    # ... 50 lines of test code
```

**After:**
```python
class TestPBCSolver:
    def test_hf_gamma_diis(self, config, positions):
        params = SolverParams(dft=False, gamma=True, diis=True, smearing=False)
        self._run_test(config, params, *positions)

    def _run_test(self, config, params, xp, kpt):
        # Reusable test logic
```

### 3. Assertions
**Before:**
```python
assert np.allclose(mo_coeff, mo_coeff_pyscf, atol=1e-2)
print("same mo_coeff")
```

**After:**
```python
def _assert_close(self, name, hqc_val, pyscf_val, atol):
    diff = jnp.max(jnp.abs(hqc_val - pyscf_val))
    print(f"{name} max diff: {diff:.2e}")
    assert np.allclose(hqc_val, pyscf_val, atol=atol), \
        f"{name} mismatch: max diff = {diff:.2e}, atol = {atol}"
    print(f"✓ {name} matches PySCF")

self._assert_close("MO coefficients", mo_coeff, mo_coeff_pyscf, config.atol_mo)
```

## Test Output Format

### Before
```
test info
n: 4
rs: 1.5
...
max diff between mo_coeff and pyscf_mo_coeff: 0.001
same mo_coeff
```

### After
```
============= Test: hf_gamma_diis =============
Method: HF
K-point: Gamma
Acceleration: DIIS
System: n=4, rs=1.5, L=2.5589
...

===== Basis: gth-dzv =====
✓ Solver converged
MO coefficients max diff: 1.43e-04
✓ MO coefficients matches PySCF
...
```

## Adding New Tests

### 1. Add to Configuration
```python
@dataclass
class TestConfig:
    # Add new parameter
    new_param: float = 1.0
```

### 2. Create Test Method
```python
def test_new_feature(self, config, positions):
    """Test new feature."""
    # Test implementation
    pass
```

### 3. Add Legacy Function (Optional)
```python
def test_new_feature():
    """Legacy wrapper."""
    test = TestClassName()
    config = TestConfig()
    # ... call new implementation
```

## Common Patterns

### Parametrized Tests
```python
@pytest.mark.parametrize("params", [
    SolverParams(dft=False, gamma=True, diis=True, smearing=False),
    SolverParams(dft=False, gamma=True, diis=False, smearing=False),
])
def test_hf_gamma(self, config, positions, params):
    self._run_test(config, params, *positions)
```

### Skip Known Issues
```python
@pytest.mark.skip(reason="PySCF k-point tests cause segmentation fault")
def test_hf_kpt_diis(self, config, positions):
    pass
```

### Fixtures with Scope
```python
@pytest.fixture(scope="class")  # Shared across all tests in class
def config(self):
    return TestConfig()

@pytest.fixture(scope="function")  # New for each test
def temp_data(self):
    return generate_temp_data()
```

## Benefits Summary

✓ **Readability**: Clear structure, good documentation
✓ **Maintainability**: Easy to modify and extend
✓ **Reusability**: Helper methods eliminate duplication
✓ **Debugging**: Detailed output helps identify issues
✓ **Professional**: Follows Python/pytest best practices
✓ **Backward Compatible**: Legacy functions still work

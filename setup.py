"""Setup for pip package."""
import unittest
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'jax',
    'jaxlib',
    'numpy',
    'pyscf'
]

def hydrogenqc_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('hydrogenqc/tests', pattern='*_test.py')
    return test_suite

setup(
    name='hydrogenqc',
    version='0.0.1',
    description='Quantum chemistry calculations in Hydrogen system.',
    url='https://code.itp.ac.cn/lzh/hydrogen-qc',
    author='lzh',
    author_email='lzh@iphy.ac.cn',
    # Contained modules and scripts.
    scripts=['bin/ferminet'],
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['pytest']},
    platforms=['any'],
    test_suite='setup.hydrogenqc_test_suite'
)
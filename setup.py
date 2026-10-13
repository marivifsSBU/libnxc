"""
pylibnxc
Machine-learned density functionals for electronic-structure codes
"""
from setuptools import setup, find_packages

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except OSError:
    long_description = ""

setup(
    name='pylibnxc',
    version='2.0.0',
    author='Marivi Fernandez-Serra, Sebastian Dick',
    description="Enables the use of machine-learned density functionals in electronic structure codes",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MPL-2.0',
    packages=find_packages(include=['pylibnxc', 'pylibnxc.*']),
    install_requires=["torch>=2.0", "numpy"],
    extras_require={
        'pyscf': ["pyscf>=2.3"],
        'export': ["jax>=0.4", "equinox>=0.11"],
        'test': ["pytest>=6.2", "pyscf>=2.3"],
    },
    python_requires=">=3.9",
)

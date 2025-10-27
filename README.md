"""
Simple VQE example for a GitHub repository
- Depends on: qiskit, scipy, numpy
- Run: python vqe_simple.py


This script implements a very small VQE for a 2-qubit Hamiltonian
H = Z0 + Z1 + 0.5 * X0 X1
using a simple parameterized ansatz (Ry rotations + entangling CNOT) and
SciPy's BFGS optimizer. It uses the statevector simulator to compute
expectation values (so no sampling noise) which keeps the example short
and dependency-light.


Place this file in your repository (e.g., `examples/`) and add a README
or instructions in your repo describing how to run it.
"""

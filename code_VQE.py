"""
This script implements a very small VQE for a 2-qubit Hamiltonian
H = Z0 + Z1 + 0.5 * X0 X1
using a simple parameterized ansatz (Ry rotations + entangling CNOT) and
SciPy's BFGS optimizer. It uses the statevector simulator to compute
expectation values (so no sampling noise) which keeps the example short
and dependency-light.
"""

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

# -----------------------------
# Hamiltonian definition
# -----------------------------
# We'll build H = Z0 + Z1 + 0.5 * X0 X1
# Represented as a list of (coefficient, pauli_string)
H_TERMS = [
    (1.0, 'Z0'),
    (1.0, 'Z1'),
    (0.5, 'X0 X1'),
]

# Convenience: single-qubit Pauli matrices
PAULI_MATRICES = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_term_matrix(pauli_string: str) -> np.ndarray:
    """Convert a pauli string like 'X0 X1' or 'Z0' into a 4x4 matrix for 2 qubits."""
    # start with identity for 2 qubits
    mats = [PAULI_MATRICES['I'], PAULI_MATRICES['I']]
    if pauli_string.strip() == '':
        return np.kron(mats[0], mats[1])
    tokens = pauli_string.split()
    for tok in tokens:
        p = tok[0]
        idx = int(tok[1])
        mats[idx] = PAULI_MATRICES[p]
    return np.kron(mats[0], mats[1])


def build_full_hamiltonian():
    H = np.zeros((4, 4), dtype=complex)
    for coef, pstr in H_TERMS:
        H += coef * pauli_term_matrix(pstr)
    return H


H_MATRIX = build_full_hamiltonian()

# -----------------------------
# Ansatz and energy evaluation
# -----------------------------

def ansatz_circuit(params: np.ndarray) -> QuantumCircuit:
    """2-qubit ansatz: Ry on each qubit, CNOT entangling, another Ry layer.
    params: length 4 -> [theta0_a, theta1_a, theta0_b, theta1_b]
    """
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    return qc


def statevector_from_circuit(qc: QuantumCircuit) -> Statevector:
    sv = Statevector.from_instruction(qc)
    return sv


def expectation_from_statevector(sv: Statevector, hamiltonian: np.ndarray) -> float:
    psi = sv.data.reshape(-1, 1)
    val = np.vdot(psi, hamiltonian.dot(psi)).real
    return float(np.real(val))


def energy_for_params(params: np.ndarray) -> float:
    qc = ansatz_circuit(params)
    sv = statevector_from_circuit(qc)
    E = expectation_from_statevector(sv, H_MATRIX)
    return E


# -----------------------------
# Run VQE (classical optimizer)
# -----------------------------

def run_vqe(initial_params=None):
    if initial_params is None:
        initial_params = 0.1 * np.random.randn(4)

    def objective(x):
        E = energy_for_params(x)
        print(f"params={x} -> E={E:.6f}")
        return E

    res = minimize(objective, initial_params, method='BFGS')
    return res


# -----------------------------
# Small driver
# -----------------------------
if __name__ == '__main__':
    print("Simple VQE example (statevector) - 2 qubits")
    print("Hamiltonian terms:")
    for c, p in H_TERMS:
        print(f"  {c} * {p}")

    # exact diagonalization for reference
    evals, evecs = np.linalg.eigh(H_MATRIX)
    print(f"\nExact ground energy (diag): {np.min(evals):.6f}")

    res = run_vqe()
    print('\nVQE result:')
    print(res)
    print(f"Approx ground energy (VQE): {res.fun:.6f}")
    print(f"Parameter vector: {res.x}")

    # compute fidelity with exact ground state
    gs_idx = np.argmin(evals)
    gs = evecs[:, gs_idx]
    qc_opt = ansatz_circuit(res.x)
    sv_opt = statevector_from_circuit(qc_opt)
    fidelity = np.abs(np.vdot(gs, sv_opt.data))**2
    print(f"Fidelity with exact ground state: {fidelity:.6f}")

import numpy as np
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
print(f" {c} * {p}")


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

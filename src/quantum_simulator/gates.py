"""
Quantum gates implementation.
"""

import numpy as np
from typing import List


class QuantumGate:
    """Base class for quantum gates."""
    
    def __init__(self, name: str, matrix: np.ndarray):
        """
        Initialize a quantum gate.
        
        Args:
            name: Name of the gate
            matrix: Unitary matrix representing the gate
        """
        self.name = name
        self.matrix = matrix
        self.num_qubits = int(np.log2(matrix.shape[0]))
    
    def apply(self, state_vector: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """
        Apply the gate to specific qubits.
        
        Args:
            state_vector: Current quantum state vector
            target_qubits: List of qubit indices to apply the gate to
            
        Returns:
            New state vector after applying the gate
        """
        n_qubits = int(np.log2(len(state_vector)))
        
        if self.num_qubits == 1 and len(target_qubits) == 1:
            # Single-qubit gate
            return self._apply_single_qubit_gate(state_vector, target_qubits[0], n_qubits)
        elif self.num_qubits == 2 and len(target_qubits) == 2:
            # Two-qubit gate (like CNOT)
            return self._apply_two_qubit_gate(state_vector, target_qubits[0], target_qubits[1], n_qubits)
        else:
            raise ValueError(f"Gate requires {self.num_qubits} qubits, got {len(target_qubits)}")
    
    def _apply_single_qubit_gate(self, state_vector: np.ndarray, target_qubit: int, n_qubits: int) -> np.ndarray:
        """Apply a single-qubit gate to the state vector."""
        new_state = np.zeros_like(state_vector)
        
        for i in range(len(state_vector)):
            # Extract the bit value of the target qubit from state index i
            bit_val = (i >> target_qubit) & 1
            
            # Apply gate matrix
            for new_bit_val in range(2):
                # Flip the target qubit bit to new_bit_val
                new_i = i ^ (bit_val << target_qubit) | (new_bit_val << target_qubit)
                new_state[new_i] += self.matrix[new_bit_val, bit_val] * state_vector[i]
        
        return new_state
    
    def _apply_two_qubit_gate(self, state_vector: np.ndarray, control_qubit: int, target_qubit: int, n_qubits: int) -> np.ndarray:
        """Apply a two-qubit gate to the state vector."""
        new_state = np.zeros_like(state_vector)
        
        for i in range(len(state_vector)):
            # Extract control and target qubit values
            control_bit = (i >> control_qubit) & 1
            target_bit = (i >> target_qubit) & 1
            
            # For CNOT: flip target if control is 1, otherwise leave unchanged
            if self.name == "CNOT":
                if control_bit == 1:
                    # Flip the target qubit
                    flipped_i = i ^ (1 << target_qubit)
                    new_state[flipped_i] += state_vector[i]
                else:
                    # Control is 0, no change
                    new_state[i] += state_vector[i]
        
        return new_state


# Common single-qubit gates
X_GATE = QuantumGate("X", np.array([[0, 1], [1, 0]], dtype=complex))
Y_GATE = QuantumGate("Y", np.array([[0, -1j], [1j, 0]], dtype=complex))
Z_GATE = QuantumGate("Z", np.array([[1, 0], [0, -1]], dtype=complex))
H_GATE = QuantumGate("H", np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))

# Rotation gates
def RY(theta: float) -> QuantumGate:
    """
    Create a rotation gate around Y-axis by angle theta.
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        QuantumGate: RY rotation gate
    """
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    matrix = np.array([
        [cos_half, -sin_half],
        [sin_half, cos_half]
    ], dtype=complex)
    return QuantumGate(f"RY({theta:.3f})", matrix)

def RZ(theta: float) -> QuantumGate:
    """
    Create a rotation gate around Z-axis by angle theta.
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        QuantumGate: RZ rotation gate
    """
    exp_neg = np.exp(-1j * theta / 2)
    exp_pos = np.exp(1j * theta / 2)
    matrix = np.array([
        [exp_neg, 0],
        [0, exp_pos]
    ], dtype=complex)
    return QuantumGate(f"RZ({theta:.3f})", matrix)

# Specific angles for W state construction
# RY gate with theta = arccos(sqrt(2/3)) ≈ 0.9553 radians
W_STATE_ANGLE_1 = np.arccos(np.sqrt(2/3))  # ~0.9553 radians
RY_W1 = RY(W_STATE_ANGLE_1)

# RY gate with theta = arccos(sqrt(1/2)) = π/4
W_STATE_ANGLE_2 = np.arccos(np.sqrt(1/2))  # π/4 radians
RY_W2 = RY(W_STATE_ANGLE_2)

# Two-qubit gates
CNOT_GATE = QuantumGate("CNOT", np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex))

# Controlled rotation gates for W state
def controlled_RY(theta: float) -> QuantumGate:
    """
    Create a controlled rotation gate around Y-axis.
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        QuantumGate: Controlled RY gate (2-qubit gate)
    """
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    
    # 4x4 matrix for controlled gate: I ⊗ |0⟩⟨0| + RY(θ) ⊗ |1⟩⟨1|
    matrix = np.array([
        [1, 0, 0, 0],           # |00⟩ → |00⟩
        [0, 1, 0, 0],           # |01⟩ → |01⟩
        [0, 0, cos_half, -sin_half],  # |10⟩ → cos(θ/2)|10⟩ - sin(θ/2)|11⟩
        [0, 0, sin_half, cos_half]    # |11⟩ → sin(θ/2)|10⟩ + cos(θ/2)|11⟩
    ], dtype=complex)
    
    return QuantumGate(f"CRY({theta:.3f})", matrix)

# Specific controlled rotation for W state
CRY_W = controlled_RY(W_STATE_ANGLE_2)
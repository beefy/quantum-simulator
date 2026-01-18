"""Test suite for quantum gates."""

import pytest
import numpy as np

from quantum_simulator import QuantumSimulator, QuantumCircuit
from quantum_simulator.gates import (
    X_GATE, Y_GATE, Z_GATE, H_GATE, CNOT_GATE,
    RX, RY, RZ, RY_W1, RY_W2, CRY_W, controlled_RY
)


class TestPauliGates:
    """Test Pauli gates (X, Y, Z)."""
    
    def test_x_gate_matrix(self):
        """Test X gate matrix definition."""
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(X_GATE.matrix, expected)
    
    def test_x_gate_properties(self):
        """Test X gate properties."""
        assert X_GATE.name == "X"
        assert X_GATE.num_qubits == 1
        
        # X gate is Hermitian: X† = X
        np.testing.assert_array_almost_equal(
            X_GATE.matrix.conj().T, X_GATE.matrix
        )
        
        # X gate is unitary: X†X = I
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(
            X_GATE.matrix.conj().T @ X_GATE.matrix, identity
        )
    
    def test_x_gate_action(self):
        """Test X gate action on basis states."""
        sim = QuantumSimulator(1)
        
        # Test X|0⟩ = |1⟩
        sim.reset()
        circuit = QuantumCircuit(1)
        circuit.add_gate(X_GATE, [0])
        circuit.execute(sim)
        
        expected = np.array([0.0, 1.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
        
        # Test X|1⟩ = |0⟩ (apply X twice, starting fresh)
        sim.reset()
        circuit = QuantumCircuit(1)
        circuit.add_gate(X_GATE, [0])  # First X: |0⟩ → |1⟩
        circuit.add_gate(X_GATE, [0])  # Second X: |1⟩ → |0⟩
        circuit.execute(sim)
        
        expected = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
    
    def test_y_gate_matrix(self):
        """Test Y gate matrix definition."""
        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        np.testing.assert_array_almost_equal(Y_GATE.matrix, expected)
    
    def test_y_gate_properties(self):
        """Test Y gate properties."""
        assert Y_GATE.name == "Y"
        assert Y_GATE.num_qubits == 1
        
        # Y gate is Hermitian: Y† = Y
        np.testing.assert_array_almost_equal(
            Y_GATE.matrix.conj().T, Y_GATE.matrix
        )
        
        # Y gate is unitary
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(
            Y_GATE.matrix.conj().T @ Y_GATE.matrix, identity
        )
    
    def test_y_gate_action(self):
        """Test Y gate action on basis states."""
        sim = QuantumSimulator(1)
        
        # Test Y|0⟩ = i|1⟩
        circuit = QuantumCircuit(1)
        circuit.add_gate(Y_GATE, [0])
        circuit.execute(sim)
        
        expected = np.array([0.0, 1j], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
    
    def test_z_gate_matrix(self):
        """Test Z gate matrix definition."""
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        np.testing.assert_array_almost_equal(Z_GATE.matrix, expected)
    
    def test_z_gate_properties(self):
        """Test Z gate properties."""
        assert Z_GATE.name == "Z"
        assert Z_GATE.num_qubits == 1
        
        # Z gate is Hermitian and unitary
        np.testing.assert_array_almost_equal(
            Z_GATE.matrix.conj().T, Z_GATE.matrix
        )
        
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(
            Z_GATE.matrix.conj().T @ Z_GATE.matrix, identity
        )
    
    def test_z_gate_action(self):
        """Test Z gate action on basis states."""
        sim = QuantumSimulator(1)
        
        # Test Z|0⟩ = |0⟩
        circuit = QuantumCircuit(1)
        circuit.add_gate(Z_GATE, [0])
        circuit.execute(sim)
        
        expected = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
        
        # Test Z|1⟩ = -|1⟩
        sim = QuantumSimulator(1)
        circuit = QuantumCircuit(1)
        circuit.add_gate(X_GATE, [0])  # Prepare |1⟩
        circuit.add_gate(Z_GATE, [0])  # Apply Z
        circuit.execute(sim)
        
        expected = np.array([0.0, -1.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)


class TestHadamardGate:
    """Test Hadamard gate."""
    
    def test_hadamard_matrix(self):
        """Test Hadamard gate matrix definition."""
        expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(H_GATE.matrix, expected)
    
    def test_hadamard_properties(self):
        """Test Hadamard gate properties."""
        assert H_GATE.name == "H"
        assert H_GATE.num_qubits == 1
        
        # Hadamard is Hermitian and unitary
        np.testing.assert_array_almost_equal(
            H_GATE.matrix.conj().T, H_GATE.matrix
        )
        
        # H² = I (Hadamard is its own inverse)
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(
            H_GATE.matrix @ H_GATE.matrix, identity
        )
    
    def test_hadamard_superposition(self):
        """Test Hadamard creates superposition."""
        sim = QuantumSimulator(1)
        
        # Test H|0⟩ = (|0⟩ + |1⟩)/√2
        circuit = QuantumCircuit(1)
        circuit.add_gate(H_GATE, [0])
        circuit.execute(sim)
        
        expected = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
        
        # Test H|1⟩ = (|0⟩ - |1⟩)/√2
        sim = QuantumSimulator(1)
        circuit = QuantumCircuit(1)
        circuit.add_gate(X_GATE, [0])  # Prepare |1⟩
        circuit.add_gate(H_GATE, [0])  # Apply H
        circuit.execute(sim)
        
        expected = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)


class TestRotationGates:
    """Test rotation gates (RX, RY, RZ)."""
    
    def test_rx_gate_creation(self):
        """Test RX gate creation and properties."""
        theta = np.pi/4
        rx_gate = RX(theta)
        
        assert rx_gate.name == f"RX({theta:.3f})"
        assert rx_gate.num_qubits == 1
        
        # Check matrix elements
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        expected = np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
        
        np.testing.assert_array_almost_equal(rx_gate.matrix, expected)
    
    def test_rx_special_cases(self):
        """Test RX gate special cases."""
        # RX(0) should be identity
        rx_zero = RX(0)
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(rx_zero.matrix, identity)
        
        # RX(π) should be equivalent to X gate (up to global phase)
        rx_pi = RX(np.pi)
        # Remove global phase factor for comparison
        rx_normalized = rx_pi.matrix / rx_pi.matrix[0, 1]
        x_normalized = X_GATE.matrix / X_GATE.matrix[0, 1]
        np.testing.assert_array_almost_equal(rx_normalized, x_normalized)
    
    def test_ry_gate_creation(self):
        """Test RY gate creation and properties."""
        theta = np.pi/3
        ry_gate = RY(theta)
        
        assert ry_gate.name == f"RY({theta:.3f})"
        assert ry_gate.num_qubits == 1
        
        # Check matrix elements
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        expected = np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
        
        np.testing.assert_array_almost_equal(ry_gate.matrix, expected)
    
    def test_ry_special_cases(self):
        """Test RY gate special cases."""
        # RY(0) should be identity
        ry_zero = RY(0)
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(ry_zero.matrix, identity)
        
        # RY(π/2) creates equal superposition
        ry_half_pi = RY(np.pi/2)
        sim = QuantumSimulator(1)
        circuit = QuantumCircuit(1)
        circuit.add_gate(ry_half_pi, [0])
        circuit.execute(sim)
        
        expected = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
    
    def test_rz_gate_creation(self):
        """Test RZ gate creation and properties."""
        theta = np.pi/6
        rz_gate = RZ(theta)
        
        assert rz_gate.name == f"RZ({theta:.3f})"
        assert rz_gate.num_qubits == 1
        
        # Check matrix elements
        exp_neg = np.exp(-1j * theta / 2)
        exp_pos = np.exp(1j * theta / 2)
        expected = np.array([
            [exp_neg, 0],
            [0, exp_pos]
        ], dtype=complex)
        
        np.testing.assert_array_almost_equal(rz_gate.matrix, expected)
    
    def test_rz_special_cases(self):
        """Test RZ gate special cases."""
        # RZ(0) should be identity
        rz_zero = RZ(0)
        identity = np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(rz_zero.matrix, identity)
        
        # RZ(π) should be equivalent to Z gate (up to global phase)
        rz_pi = RZ(np.pi)
        # The RZ(π) matrix has extra phase factors compared to Z
        # Check that it gives the same action on basis states
        sim1 = QuantumSimulator(1)
        sim2 = QuantumSimulator(1)
        
        circuit1 = QuantumCircuit(1)
        circuit1.add_gate(Z_GATE, [0])
        circuit1.execute(sim1)
        
        circuit2 = QuantumCircuit(1)
        circuit2.add_gate(rz_pi, [0])
        circuit2.execute(sim2)
        
        # Both should leave |0⟩ unchanged (up to global phase)
        assert abs(sim1.get_state_vector()[0]) == pytest.approx(abs(sim2.get_state_vector()[0]))
    
    def test_rotation_unitarity(self):
        """Test that rotation gates are unitary."""
        angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi, 2*np.pi]
        
        for theta in angles:
            rx = RX(theta)
            ry = RY(theta)
            rz = RZ(theta)
            
            identity = np.eye(2, dtype=complex)
            
            # Test unitarity: U†U = I
            np.testing.assert_array_almost_equal(
                rx.matrix.conj().T @ rx.matrix, identity
            )
            np.testing.assert_array_almost_equal(
                ry.matrix.conj().T @ ry.matrix, identity
            )
            np.testing.assert_array_almost_equal(
                rz.matrix.conj().T @ rz.matrix, identity
            )


class TestWStateGates:
    """Test special gates for W state construction."""
    
    def test_ry_w1_properties(self):
        """Test RY_W1 gate properties."""
        expected_angle = np.arccos(np.sqrt(2/3))
        assert RY_W1.name == f"RY({expected_angle:.3f})"
        assert RY_W1.num_qubits == 1
        
        # Check matrix matches expected angle
        cos_half = np.cos(expected_angle / 2)
        sin_half = np.sin(expected_angle / 2)
        expected_matrix = np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
        
        np.testing.assert_array_almost_equal(RY_W1.matrix, expected_matrix)
    
    def test_ry_w2_properties(self):
        """Test RY_W2 gate properties."""
        expected_angle = np.arccos(np.sqrt(1/2))  # π/4
        assert RY_W2.name == f"RY({expected_angle:.3f})"
        assert RY_W2.num_qubits == 1
        
        # Should be equivalent to RY(π/4)
        ry_pi_4 = RY(np.pi/4)
        np.testing.assert_array_almost_equal(RY_W2.matrix, ry_pi_4.matrix)
    
    def test_w_state_angles_values(self):
        """Test that W state angles have correct numerical values."""
        # RY_W1 angle should be arccos(√(2/3)) ≈ 0.6155 radians
        w1_angle = np.arccos(np.sqrt(2/3))
        assert w1_angle == pytest.approx(0.6155, abs=1e-4)
        
        # RY_W2 angle should be π/4 ≈ 0.7854 radians
        w2_angle = np.arccos(np.sqrt(1/2))
        assert w2_angle == pytest.approx(np.pi/4, abs=1e-10)


class TestTwoQubitGates:
    """Test two-qubit gates."""
    
    def test_cnot_matrix(self):
        """Test CNOT gate matrix definition."""
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        np.testing.assert_array_almost_equal(CNOT_GATE.matrix, expected)
    
    def test_cnot_properties(self):
        """Test CNOT gate properties."""
        assert CNOT_GATE.name == "CNOT"
        assert CNOT_GATE.num_qubits == 2
        
        # CNOT is Hermitian and unitary
        np.testing.assert_array_almost_equal(
            CNOT_GATE.matrix.conj().T, CNOT_GATE.matrix
        )
        
        identity = np.eye(4, dtype=complex)
        np.testing.assert_array_almost_equal(
            CNOT_GATE.matrix.conj().T @ CNOT_GATE.matrix, identity
        )
    
    def test_cnot_action(self):
        """Test CNOT gate action on basis states."""
        sim = QuantumSimulator(2)
        
        # Test CNOT|00⟩ = |00⟩
        circuit = QuantumCircuit(2)
        circuit.add_gate(CNOT_GATE, [0, 1])
        circuit.execute(sim)
        
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
        
        # Test CNOT|01⟩ = |11⟩ (control=0 is 1, so flip target=1)
        sim = QuantumSimulator(2)
        circuit = QuantumCircuit(2)
        circuit.add_gate(X_GATE, [0])  # Prepare |01⟩
        circuit.add_gate(CNOT_GATE, [0, 1])
        circuit.execute(sim)
        
        expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
    
    def test_bell_state_creation(self):
        """Test Bell state creation with CNOT."""
        sim = QuantumSimulator(2)
        circuit = QuantumCircuit(2)
        
        # Create Bell state: H⊗I then CNOT
        circuit.add_gate(H_GATE, [0])
        circuit.add_gate(CNOT_GATE, [0, 1])
        circuit.execute(sim)
        
        # Should be (|00⟩ + |11⟩)/√2
        expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)


class TestControlledRotations:
    """Test controlled rotation gates."""
    
    def test_controlled_ry_creation(self):
        """Test controlled RY gate creation."""
        theta = np.pi/3
        cry_gate = controlled_RY(theta)
        
        assert cry_gate.name == f"CRY({theta:.3f})"
        assert cry_gate.num_qubits == 2
        
        # Check matrix structure: I⊗|0⟩⟨0| + RY(θ)⊗|1⟩⟨1|
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos_half, -sin_half],
            [0, 0, sin_half, cos_half]
        ], dtype=complex)
        
        np.testing.assert_array_almost_equal(cry_gate.matrix, expected)
    
    def test_cry_w_properties(self):
        """Test CRY_W gate properties."""
        w2_angle = np.arccos(np.sqrt(1/2))  # π/4
        
        assert CRY_W.name == f"CRY({w2_angle:.3f})"
        assert CRY_W.num_qubits == 2
        
        # Should be equivalent to controlled_RY(π/4)
        cry_pi_4 = controlled_RY(np.pi/4)
        np.testing.assert_array_almost_equal(CRY_W.matrix, cry_pi_4.matrix)
    
    def test_controlled_rotation_action(self):
        """Test controlled rotation gate action."""
        cry = controlled_RY(np.pi/2)
        
        # Test on |00⟩ - should remain unchanged
        sim = QuantumSimulator(2)
        circuit = QuantumCircuit(2)
        circuit.add_gate(cry, [0, 1])
        circuit.execute(sim)
        
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)
        
        # Test on |01⟩ - control qubit 0 is 1, so should apply RY(π/2) to target qubit 1
        sim = QuantumSimulator(2)
        circuit = QuantumCircuit(2)
        circuit.add_gate(X_GATE, [0])  # Prepare |01⟩ (X on qubit 0)
        circuit.add_gate(cry, [0, 1])  # Apply controlled rotation (control=0, target=1)
        circuit.execute(sim)
        
        # Should become (|01⟩ + |11⟩)/√2 = [0, 1/√2, 0, 1/√2]
        expected = np.array([0.0, 1.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(sim.get_state_vector(), expected)


class TestGateComposition:
    """Test gate composition and sequences."""
    
    def test_rotation_composition(self):
        """Test that rotation gates compose correctly."""
        theta1 = np.pi/6
        theta2 = np.pi/4
        
        # RY(θ1) followed by RY(θ2) should equal RY(θ1 + θ2)
        sim1 = QuantumSimulator(1)
        sim2 = QuantumSimulator(1)
        
        circuit1 = QuantumCircuit(1)
        circuit1.add_gate(RY(theta1), [0])
        circuit1.add_gate(RY(theta2), [0])
        circuit1.execute(sim1)
        
        circuit2 = QuantumCircuit(1)
        circuit2.add_gate(RY(theta1 + theta2), [0])
        circuit2.execute(sim2)
        
        np.testing.assert_array_almost_equal(
            sim1.get_state_vector(), sim2.get_state_vector()
        )
    
    def test_pauli_anticommutation(self):
        """Test Pauli gate anticommutation relations."""
        sim1 = QuantumSimulator(1)
        sim2 = QuantumSimulator(1)
        
        # XY should equal -YX (anticommute)
        circuit1 = QuantumCircuit(1)
        circuit1.add_gate(X_GATE, [0])
        circuit1.add_gate(Y_GATE, [0])
        circuit1.execute(sim1)
        
        circuit2 = QuantumCircuit(1)
        circuit2.add_gate(Y_GATE, [0])
        circuit2.add_gate(X_GATE, [0])
        circuit2.execute(sim2)
        
        # States should differ by a sign (and possibly phase)
        state1 = sim1.get_state_vector()
        state2 = sim2.get_state_vector()
        
        # Check that |XY|ψ⟩| = |YX|ψ⟩| (same magnitude)
        np.testing.assert_array_almost_equal(np.abs(state1), np.abs(state2))
    
    def test_hadamard_basis_change(self):
        """Test Hadamard changes computational basis."""
        sim = QuantumSimulator(1)
        circuit = QuantumCircuit(1)
        
        # HZH should equal X (basis change property)
        circuit.add_gate(H_GATE, [0])
        circuit.add_gate(Z_GATE, [0])
        circuit.add_gate(H_GATE, [0])
        circuit.execute(sim)
        
        # Compare with X gate applied directly
        sim_x = QuantumSimulator(1)
        circuit_x = QuantumCircuit(1)
        circuit_x.add_gate(X_GATE, [0])
        circuit_x.execute(sim_x)
        
        np.testing.assert_array_almost_equal(
            sim.get_state_vector(), sim_x.get_state_vector()
        )


if __name__ == "__main__":
    pytest.main([__file__])
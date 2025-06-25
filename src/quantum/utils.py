"""
Quantum Utilities Module

This module provides utility functions for quantum computations, 
state manipulation, and common quantum operations.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

def pauli_matrices() -> Dict[str, np.ndarray]:
    """
    Return the standard Pauli matrices.
    
    Returns:
        Dictionary of Pauli matrices
    """
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

def computational_basis_states(n_qubits: int) -> List[np.ndarray]:
    """
    Generate computational basis states for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        List of basis state vectors
    """
    states = []
    for i in range(2**n_qubits):
        state = np.zeros(2**n_qubits, dtype=complex)
        state[i] = 1.0
        states.append(state)
    
    return states

def bell_states() -> Dict[str, np.ndarray]:
    """
    Return the four Bell states.
    
    Returns:
        Dictionary of Bell states
    """
    return {
        'phi_plus': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        'phi_minus': np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        'psi_plus': np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        'psi_minus': np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    }

def ghz_state(n_qubits: int) -> np.ndarray:
    """
    Generate GHZ state for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        GHZ state vector
    """
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)  # |00...0⟩
    state[-1] = 1.0 / np.sqrt(2)  # |11...1⟩
    
    return state

def w_state(n_qubits: int) -> np.ndarray:
    """
    Generate W state for n qubits.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        W state vector
    """
    state = np.zeros(2**n_qubits, dtype=complex)
    
    # Add states with exactly one qubit in |1⟩
    for i in range(n_qubits):
        index = 2**i
        state[index] = 1.0 / np.sqrt(n_qubits)
    
    return state

def random_quantum_state(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random quantum state.
    
    Args:
        n_qubits: Number of qubits
        seed: Random seed
        
    Returns:
        Random normalized quantum state
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random complex amplitudes
    dim = 2**n_qubits
    real_parts = np.random.normal(0, 1, dim)
    imag_parts = np.random.normal(0, 1, dim)
    
    state = real_parts + 1j * imag_parts
    
    # Normalize
    state = state / np.linalg.norm(state)
    
    return state

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate fidelity between two quantum states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity (0 to 1)
    """
    # Handle different input types
    if state1.ndim == 1 and state2.ndim == 1:
        # Both are state vectors
        return abs(np.vdot(state1, state2))**2
    elif state1.ndim == 2 and state2.ndim == 2:
        # Both are density matrices
        sqrt_rho1 = sqrtm(state1)
        return np.real(np.trace(sqrtm(sqrt_rho1 @ state2 @ sqrt_rho1))**2)
    else:
        # Mixed case
        if state1.ndim == 1:
            state1 = np.outer(state1, np.conj(state1))
        if state2.ndim == 1:
            state2 = np.outer(state2, np.conj(state2))
        
        sqrt_rho1 = sqrtm(state1)
        return np.real(np.trace(sqrtm(sqrt_rho1 @ state2 @ sqrt_rho1))**2)

def trace_distance(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate trace distance between two quantum states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Trace distance (0 to 1)
    """
    # Convert to density matrices if needed
    if state1.ndim == 1:
        state1 = np.outer(state1, np.conj(state1))
    if state2.ndim == 1:
        state2 = np.outer(state2, np.conj(state2))
    
    diff = state1 - state2
    eigenvals = np.linalg.eigvals(diff)
    
    return 0.5 * np.sum(np.abs(eigenvals))

def von_neumann_entropy(rho: np.ndarray, base: float = 2) -> float:
    """
    Calculate von Neumann entropy of a density matrix.
    
    Args:
        rho: Density matrix
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        von Neumann entropy
    """
    eigenvals = np.linalg.eigvals(rho)
    eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero eigenvalues
    
    if base == 2:
        return -np.sum(eigenvals * np.log2(eigenvals))
    else:
        return -np.sum(eigenvals * np.log(eigenvals))

def partial_trace(rho: np.ndarray, subsystem: Union[int, List[int]], dims: List[int]) -> np.ndarray:
    """
    Calculate partial trace of a density matrix.
    
    Args:
        rho: Density matrix
        subsystem: Subsystem(s) to trace out
        dims: Dimensions of each subsystem
        
    Returns:
        Partially traced density matrix
    """
    if isinstance(subsystem, int):
        subsystem = [subsystem]
    
    n_systems = len(dims)
    total_dim = np.prod(dims)
    
    # Reshape density matrix
    rho_reshaped = rho.reshape(dims + dims)
    
    # Trace out specified subsystems
    for sub in sorted(subsystem, reverse=True):
        rho_reshaped = np.trace(rho_reshaped, axis1=sub, axis2=sub + n_systems - len(subsystem))
        dims.pop(sub)
        n_systems -= 1
    
    # Reshape back to matrix form
    remaining_dim = np.prod(dims)
    return rho_reshaped.reshape(remaining_dim, remaining_dim)

def entanglement_entropy(state: np.ndarray, subsystem_size: int) -> float:
    """
    Calculate entanglement entropy between subsystems.
    
    Args:
        state: Quantum state (vector or density matrix)
        subsystem_size: Size of subsystem A
        
    Returns:
        Entanglement entropy
    """
    # Convert to density matrix if needed
    if state.ndim == 1:
        rho = np.outer(state, np.conj(state))
    else:
        rho = state
    
    # Determine dimensions
    total_dim = rho.shape[0]
    n_qubits = int(np.log2(total_dim))
    
    if subsystem_size > n_qubits:
        raise ValueError("Subsystem size cannot exceed total system size")
    
    # Calculate dimensions of subsystems
    dim_A = 2**subsystem_size
    dim_B = 2**(n_qubits - subsystem_size)
    
    # Partial trace over subsystem B
    rho_A = partial_trace(rho, list(range(subsystem_size, n_qubits)), [2]*n_qubits)
    
    # Calculate von Neumann entropy
    return von_neumann_entropy(rho_A)

def sqrtm(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate matrix square root.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Matrix square root
    """
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
    sqrt_eigenvals = np.sqrt(eigenvals)
    
    return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T.conj()

def tensor_product(*operators: np.ndarray) -> np.ndarray:
    """
    Calculate tensor product of multiple operators.
    
    Args:
        operators: Operators to tensor
        
    Returns:
        Tensor product result
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    
    return result

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate commutator [A, B] = AB - BA.
    
    Args:
        A: First operator
        B: Second operator
        
    Returns:
        Commutator
    """
    return A @ B - B @ A

def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate anticommutator {A, B} = AB + BA.
    
    Args:
        A: First operator
        B: Second operator
        
    Returns:
        Anticommutator
    """
    return A @ B + B @ A

def expectation_value(operator: np.ndarray, state: np.ndarray) -> complex:
    """
    Calculate expectation value of an operator.
    
    Args:
        operator: Quantum operator
        state: Quantum state (vector or density matrix)
        
    Returns:
        Expectation value
    """
    if state.ndim == 1:
        # State vector
        return np.conj(state) @ operator @ state
    else:
        # Density matrix
        return np.trace(operator @ state)

def bloch_vector(state: np.ndarray) -> np.ndarray:
    """
    Calculate Bloch vector for a 2-level system.
    
    Args:
        state: Quantum state (2D)
        
    Returns:
        Bloch vector [x, y, z]
    """
    if state.ndim == 1:
        rho = np.outer(state, np.conj(state))
    else:
        rho = state
    
    if rho.shape != (2, 2):
        raise ValueError("Bloch vector only defined for 2-level systems")
    
    pauli = pauli_matrices()
    
    x = np.real(np.trace(pauli['X'] @ rho))
    y = np.real(np.trace(pauli['Y'] @ rho))
    z = np.real(np.trace(pauli['Z'] @ rho))
    
    return np.array([x, y, z])

async def quantum_fourier_transform(n_qubits: int) -> np.ndarray:
    """
    Generate quantum Fourier transform matrix.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        QFT matrix
    """
    N = 2**n_qubits
    omega = np.exp(2j * np.pi / N)
    
    QFT = np.zeros((N, N), dtype=complex)
    
    for j in range(N):
        for k in range(N):
            QFT[j, k] = omega**(j * k) / np.sqrt(N)
    
    return QFT

def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary.
    
    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance
        
    Returns:
        True if unitary
    """
    identity = np.eye(matrix.shape[0])
    product = matrix @ matrix.T.conj()
    
    return np.allclose(product, identity, atol=tolerance)

def is_hermitian(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is Hermitian.
    
    Args:
        matrix: Matrix to check
        tolerance: Numerical tolerance
        
    Returns:
        True if Hermitian
    """
    return np.allclose(matrix, matrix.T.conj(), atol=tolerance)

def random_unitary(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random unitary matrix using QR decomposition.
    
    Args:
        n: Matrix dimension
        seed: Random seed
        
    Returns:
        Random unitary matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random complex matrix
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    
    # QR decomposition
    Q, R = np.linalg.qr(A)
    
    # Ensure proper phases
    D = np.diagonal(R)
    ph = D / np.abs(D)
    Q = Q @ np.diag(ph)
    
    return Q
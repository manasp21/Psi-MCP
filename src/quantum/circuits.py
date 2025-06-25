"""
Quantum Circuit Operations Module

This module provides comprehensive quantum circuit creation, manipulation,
simulation, and optimization functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import json
import numpy as np

logger = logging.getLogger(__name__)

class QuantumCircuitManager:
    """Manager for quantum circuit operations across different backends."""
    
    def __init__(self):
        self.circuits = {}
        self.circuit_counter = 0
    
    def generate_circuit_id(self) -> str:
        """Generate unique circuit ID."""
        self.circuit_counter += 1
        return f"circuit_{self.circuit_counter}"

# Global circuit manager
circuit_manager = QuantumCircuitManager()

async def create_circuit(
    num_qubits: int,
    circuit_type: str = "empty",
    backend: str = "qasm_simulator"
) -> Dict[str, Any]:
    """
    Create a quantum circuit with specified parameters.
    
    Args:
        num_qubits: Number of qubits
        circuit_type: Type of circuit ('empty', 'bell', 'ghz', 'random', 'qft')
        backend: Backend to target
        
    Returns:
        Dictionary with circuit information
    """
    logger.info(f"Creating {circuit_type} circuit with {num_qubits} qubits for {backend}")
    
    try:
        # Import based on backend
        if 'qiskit' in backend.lower():
            return await _create_qiskit_circuit(num_qubits, circuit_type, backend)
        elif 'cirq' in backend.lower():
            return await _create_cirq_circuit(num_qubits, circuit_type, backend)
        elif 'pennylane' in backend.lower():
            return await _create_pennylane_circuit(num_qubits, circuit_type, backend)
        else:
            # Default to Qiskit
            return await _create_qiskit_circuit(num_qubits, circuit_type, 'qasm_simulator')
            
    except Exception as e:
        logger.error(f"Error creating circuit: {e}")
        raise

async def _create_qiskit_circuit(num_qubits: int, circuit_type: str, backend: str) -> Dict[str, Any]:
    """Create circuit using Qiskit."""
    from qiskit import QuantumCircuit, ClassicalRegister
    
    # Create base circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    circuit_id = circuit_manager.generate_circuit_id()
    
    # Add gates based on circuit type
    if circuit_type == "bell":
        if num_qubits < 2:
            raise ValueError("Bell state requires at least 2 qubits")
        qc.h(0)
        qc.cx(0, 1)
        
    elif circuit_type == "ghz":
        if num_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
            
    elif circuit_type == "qft":
        # Quantum Fourier Transform
        for i in range(num_qubits):
            qc.h(i)
            for j in range(i+1, num_qubits):
                qc.cp(np.pi/2**(j-i), i, j)
        
        # Swap qubits
        for i in range(num_qubits//2):
            qc.swap(i, num_qubits-1-i)
            
    elif circuit_type == "random":
        # Random circuit
        np.random.seed(42)  # For reproducibility
        for _ in range(num_qubits * 3):
            qubit = np.random.randint(0, num_qubits)
            gate = np.random.choice(['h', 'x', 'y', 'z', 's', 't'])
            getattr(qc, gate)(qubit)
        
        # Add some random CNOT gates
        for _ in range(num_qubits):
            control = np.random.randint(0, num_qubits)
            target = np.random.randint(0, num_qubits)
            if control != target:
                qc.cx(control, target)
    
    # Store circuit
    circuit_info = {
        'id': circuit_id,
        'num_qubits': num_qubits,
        'type': circuit_type,
        'backend': backend,
        'depth': qc.depth(),
        'gate_count': len(qc.data),
        'qasm': qc.qasm()
    }
    
    circuit_manager.circuits[circuit_id] = {
        'circuit': qc,
        'info': circuit_info
    }
    
    return circuit_info

async def _create_cirq_circuit(num_qubits: int, circuit_type: str, backend: str) -> Dict[str, Any]:
    """Create circuit using Cirq."""
    import cirq
    
    # Create qubits
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    circuit_id = circuit_manager.generate_circuit_id()
    
    # Add gates based on circuit type
    if circuit_type == "bell":
        if num_qubits < 2:
            raise ValueError("Bell state requires at least 2 qubits")
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        
    elif circuit_type == "ghz":
        if num_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        circuit.append(cirq.H(qubits[0]))
        for i in range(1, num_qubits):
            circuit.append(cirq.CNOT(qubits[0], qubits[i]))
    
    # Store circuit
    circuit_info = {
        'id': circuit_id,
        'num_qubits': num_qubits,
        'type': circuit_type,
        'backend': backend,
        'depth': len(circuit),
        'gate_count': len(list(circuit.all_operations()))
    }
    
    circuit_manager.circuits[circuit_id] = {
        'circuit': circuit,
        'qubits': qubits,
        'info': circuit_info
    }
    
    return circuit_info

async def _create_pennylane_circuit(num_qubits: int, circuit_type: str, backend: str) -> Dict[str, Any]:
    """Create circuit using PennyLane."""
    import pennylane as qml
    
    circuit_id = circuit_manager.generate_circuit_id()
    
    # Create device
    dev = qml.device('default.qubit', wires=num_qubits)
    
    # Define circuit function based on type
    if circuit_type == "bell":
        if num_qubits < 2:
            raise ValueError("Bell state requires at least 2 qubits")
        
        @qml.qnode(dev)
        def bell_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()
            
        circuit_func = bell_circuit
        
    elif circuit_type == "ghz":
        if num_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
            
        @qml.qnode(dev)
        def ghz_circuit():
            qml.Hadamard(wires=0)
            for i in range(1, num_qubits):
                qml.CNOT(wires=[0, i])
            return qml.state()
            
        circuit_func = ghz_circuit
    
    else:
        # Empty circuit
        @qml.qnode(dev)
        def empty_circuit():
            return qml.state()
            
        circuit_func = empty_circuit
    
    # Store circuit
    circuit_info = {
        'id': circuit_id,
        'num_qubits': num_qubits,
        'type': circuit_type,
        'backend': backend,
        'framework': 'pennylane'
    }
    
    circuit_manager.circuits[circuit_id] = {
        'circuit': circuit_func,
        'device': dev,
        'info': circuit_info
    }
    
    return circuit_info

async def simulate_circuit(
    circuit_definition: str,
    shots: int = 1024,
    backend: str = "qasm_simulator",
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Simulate a quantum circuit and return results.
    
    Args:
        circuit_definition: Circuit ID or QASM string
        shots: Number of measurement shots
        backend: Simulation backend
        timeout: Execution timeout
        
    Returns:
        Simulation results
    """
    logger.info(f"Simulating circuit with {shots} shots on {backend}")
    
    try:
        # Check if it's a circuit ID
        if circuit_definition in circuit_manager.circuits:
            circuit_data = circuit_manager.circuits[circuit_definition]
            return await _simulate_stored_circuit(circuit_data, shots, backend, timeout)
        else:
            # Try to parse as QASM
            return await _simulate_qasm_circuit(circuit_definition, shots, backend, timeout)
            
    except Exception as e:
        logger.error(f"Error simulating circuit: {e}")
        raise

async def _simulate_stored_circuit(
    circuit_data: Dict[str, Any],
    shots: int,
    backend: str,
    timeout: int
) -> Dict[str, Any]:
    """Simulate a stored circuit."""
    circuit = circuit_data['circuit']
    info = circuit_data['info']
    
    if 'qiskit' in info.get('backend', '').lower():
        return await _simulate_qiskit_circuit(circuit, shots, backend, timeout)
    elif 'cirq' in info.get('backend', '').lower():
        return await _simulate_cirq_circuit(circuit, circuit_data['qubits'], shots, timeout)
    elif 'pennylane' in info.get('framework', '').lower():
        return await _simulate_pennylane_circuit(circuit, shots, timeout)
    else:
        raise ValueError(f"Unsupported circuit framework: {info}")

async def _simulate_qiskit_circuit(circuit, shots: int, backend_name: str, timeout: int) -> Dict[str, Any]:
    """Simulate Qiskit circuit."""
    from qiskit import transpile, execute
    from quantum import get_backend
    
    # Add measurements if not present
    if not any(isinstance(instr.operation, type(circuit.measure)) for instr in circuit.data):
        circuit.add_register(circuit.cregs[0] if circuit.cregs else circuit.add_register(circuit.num_qubits))
        circuit.measure_all()
    
    # Get backend
    backend = get_backend(backend_name)
    
    # Transpile and execute
    transpiled_circuit = transpile(circuit, backend)
    job = execute(transpiled_circuit, backend, shots=shots)
    result = job.result()
    
    # Extract results
    counts = result.get_counts()
    
    return {
        'counts': dict(counts),
        'shots': shots,
        'success': True,
        'execution_time': getattr(result, 'time_taken', 0),
        'backend': backend_name
    }

async def _simulate_cirq_circuit(circuit, qubits, shots: int, timeout: int) -> Dict[str, Any]:
    """Simulate Cirq circuit."""
    import cirq
    
    # Add measurements
    circuit.append(cirq.measure(*qubits, key='result'))
    
    # Run simulation
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=shots)
    
    # Process results
    measurements = result.measurements['result']
    counts = {}
    
    for measurement in measurements:
        bitstring = ''.join(str(bit) for bit in measurement)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    return {
        'counts': counts,
        'shots': shots,
        'success': True,
        'backend': 'cirq_simulator'
    }

async def _simulate_pennylane_circuit(circuit_func, shots: int, timeout: int) -> Dict[str, Any]:
    """Simulate PennyLane circuit."""
    # Execute circuit
    state = circuit_func()
    
    # Convert to counts (simplified)
    probabilities = np.abs(state) ** 2
    counts = {}
    
    # Sample from probabilities
    for i, prob in enumerate(probabilities):
        if prob > 1e-10:  # Only include significant probabilities
            bitstring = format(i, f'0{int(np.log2(len(probabilities)))}b')
            counts[bitstring] = int(prob * shots)
    
    return {
        'counts': counts,
        'shots': shots,
        'success': True,
        'state': state.tolist() if hasattr(state, 'tolist') else str(state),
        'backend': 'pennylane'
    }

async def optimize_circuit(
    circuit_definition: str,
    optimization_level: int = 1,
    target_backend: str = "qasm_simulator"
) -> Dict[str, Any]:
    """
    Optimize a quantum circuit for better performance.
    
    Args:
        circuit_definition: Circuit ID or definition
        optimization_level: Optimization level (0-3)
        target_backend: Target backend for optimization
        
    Returns:
        Optimization results
    """
    logger.info(f"Optimizing circuit with level {optimization_level} for {target_backend}")
    
    try:
        if circuit_definition in circuit_manager.circuits:
            circuit_data = circuit_manager.circuits[circuit_definition]
            return await _optimize_stored_circuit(circuit_data, optimization_level, target_backend)
        else:
            return await _optimize_qasm_circuit(circuit_definition, optimization_level, target_backend)
            
    except Exception as e:
        logger.error(f"Error optimizing circuit: {e}")
        raise

async def _optimize_stored_circuit(
    circuit_data: Dict[str, Any],
    optimization_level: int,
    target_backend: str
) -> Dict[str, Any]:
    """Optimize a stored circuit."""
    from qiskit import transpile
    from quantum import get_backend
    
    circuit = circuit_data['circuit']
    original_depth = circuit.depth()
    original_gates = len(circuit.data)
    
    # Get target backend
    backend = get_backend(target_backend)
    
    # Transpile with optimization
    optimized_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level
    )
    
    # Generate new circuit ID for optimized version
    optimized_id = circuit_manager.generate_circuit_id()
    
    # Store optimized circuit
    optimized_info = {
        'id': optimized_id,
        'num_qubits': optimized_circuit.num_qubits,
        'type': circuit_data['info']['type'] + '_optimized',
        'backend': target_backend,
        'depth': optimized_circuit.depth(),
        'gate_count': len(optimized_circuit.data),
        'optimization_level': optimization_level,
        'original_id': circuit_data['info']['id']
    }
    
    circuit_manager.circuits[optimized_id] = {
        'circuit': optimized_circuit,
        'info': optimized_info
    }
    
    return {
        'optimized_circuit_id': optimized_id,
        'original_depth': original_depth,
        'optimized_depth': optimized_circuit.depth(),
        'original_gates': original_gates,
        'optimized_gates': len(optimized_circuit.data),
        'depth_reduction': original_depth - optimized_circuit.depth(),
        'gate_reduction': original_gates - len(optimized_circuit.data),
        'optimization_level': optimization_level,
        'success': True
    }

async def get_circuit_info(circuit_id: str) -> Dict[str, Any]:
    """Get information about a stored circuit."""
    if circuit_id not in circuit_manager.circuits:
        raise ValueError(f"Circuit {circuit_id} not found")
    
    return circuit_manager.circuits[circuit_id]['info']

async def list_circuits() -> List[Dict[str, Any]]:
    """List all stored circuits."""
    return [data['info'] for data in circuit_manager.circuits.values()]
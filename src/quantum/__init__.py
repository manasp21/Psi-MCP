"""
Psi-MCP Quantum Computing Module

This module provides comprehensive quantum computing and quantum physics functionality
for the Psi-MCP server, including:

- Quantum circuit operations
- Open and closed quantum systems
- Quantum algorithms  
- Quantum chemistry calculations
- Many-body physics simulations
- Quantum machine learning
- Quantum field theory tools
- Advanced visualization
"""

import logging
from typing import Dict, Any, Optional
import asyncio

# Configure module logger
logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "Manas Pandey"

# Quantum backend registry
_backends = {}
_initialized = False

async def initialize_backends(config: Any) -> None:
    """Initialize quantum computing backends based on configuration."""
    global _backends, _initialized
    
    if _initialized:
        logger.info("Quantum backends already initialized")
        return
    
    logger.info("Initializing quantum computing backends...")
    
    # Initialize Qiskit backends
    try:
        from qiskit import Aer
        from qiskit.providers.fake_provider import FakeProvider
        
        # Add simulators
        _backends['qasm_simulator'] = Aer.get_backend('qasm_simulator')
        _backends['statevector_simulator'] = Aer.get_backend('statevector_simulator')
        _backends['unitary_simulator'] = Aer.get_backend('unitary_simulator')
        
        # Add noise models for realistic simulation
        fake_provider = FakeProvider()
        fake_backends = fake_provider.backends()
        if fake_backends:
            _backends['fake_backend'] = fake_backends[0]
        
        logger.info("Qiskit backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize Qiskit backends: {e}")
    
    # Initialize Cirq backends
    try:
        import cirq
        _backends['cirq_simulator'] = cirq.Simulator()
        logger.info("Cirq backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize Cirq backends: {e}")
    
    # Initialize PennyLane backends
    try:
        import pennylane as qml
        _backends['pennylane_default'] = qml.device('default.qubit', wires=config.max_qubits)
        
        # GPU backend if enabled
        if config.enable_gpu:
            try:
                _backends['pennylane_gpu'] = qml.device('default.qubit.torch', wires=config.max_qubits)
                logger.info("PennyLane GPU backend initialized")
            except Exception:
                logger.warning("PennyLane GPU backend not available")
        
        logger.info("PennyLane backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize PennyLane backends: {e}")
    
    # Initialize QuTiP for open systems
    try:
        import qutip
        logger.info(f"QuTiP initialized (version {qutip.__version__})")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize QuTiP: {e}")
    
    # Initialize quantum chemistry backends
    try:
        import openfermion
        import pyscf
        logger.info("Quantum chemistry backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize quantum chemistry backends: {e}")
    
    _initialized = True
    logger.info(f"Quantum backends initialization complete. Available backends: {list(_backends.keys())}")

def get_backend(name: str):
    """Get a quantum backend by name."""
    if not _initialized:
        raise RuntimeError("Backends not initialized. Call initialize_backends() first.")
    
    if name not in _backends:
        raise ValueError(f"Backend '{name}' not available. Available backends: {list(_backends.keys())}")
    
    return _backends[name]

def list_backends() -> Dict[str, str]:
    """List all available quantum backends."""
    if not _initialized:
        return {}
    
    backend_info = {}
    for name, backend in _backends.items():
        try:
            backend_type = type(backend).__name__
            backend_info[name] = backend_type
        except:
            backend_info[name] = "Unknown"
    
    return backend_info

# Import submodules for easier access
try:
    from . import circuits
    from . import systems
    from . import algorithms
    from . import chemistry
    from . import many_body
    from . import field_theory
    from . import ml
    from . import visualization
    from . import utils
    
    logger.info("All quantum submodules imported successfully")
    
except ImportError as e:
    logger.warning(f"Some quantum submodules failed to import: {e}")

# Export main functions
__all__ = [
    'initialize_backends',
    'get_backend', 
    'list_backends',
    'circuits',
    'systems',
    'algorithms',
    'chemistry',
    'many_body',
    'field_theory',
    'ml',
    'visualization',
    'utils',
]
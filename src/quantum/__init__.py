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
        
        # Add simulators
        _backends['qasm_simulator'] = Aer.get_backend('qasm_simulator')
        _backends['statevector_simulator'] = Aer.get_backend('statevector_simulator')
        _backends['unitary_simulator'] = Aer.get_backend('unitary_simulator')
        
        logger.info("Qiskit backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize Qiskit backends: {e}")
    except Exception as e:
        logger.warning(f"Error initializing Qiskit: {e}")
    
    # Initialize Cirq backends
    try:
        import cirq
        _backends['cirq_simulator'] = cirq.Simulator()
        logger.info("Cirq backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize Cirq backends: {e}")
    except Exception as e:
        logger.warning(f"Error initializing Cirq: {e}")
    
    # Initialize PennyLane backends
    try:
        import pennylane as qml
        _backends['pennylane_default'] = qml.device('default.qubit', wires=config.max_qubits)
        logger.info("PennyLane backends initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize PennyLane backends: {e}")
    except Exception as e:
        logger.warning(f"Error initializing PennyLane: {e}")
    
    # Initialize QuTiP for open systems
    try:
        import qutip
        logger.info(f"QuTiP initialized (version {qutip.__version__})")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize QuTiP: {e}")
    except Exception as e:
        logger.warning(f"Error initializing QuTiP: {e}")
    
    # Initialize quantum chemistry backends
    try:
        import openfermion
        logger.info("OpenFermion initialized")
        
    except ImportError as e:
        logger.warning(f"Failed to initialize OpenFermion: {e}")
    except Exception as e:
        logger.warning(f"Error initializing OpenFermion: {e}")
    
    # Add a default simulator if nothing else worked
    if not _backends:
        logger.warning("No quantum backends available, adding basic simulator")
        _backends['basic_simulator'] = 'basic'
    
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
_submodules = {}

try:
    from . import circuits
    _submodules['circuits'] = circuits
except ImportError as e:
    logger.warning(f"Failed to import circuits module: {e}")

try:
    from . import systems
    _submodules['systems'] = systems
except ImportError as e:
    logger.warning(f"Failed to import systems module: {e}")

try:
    from . import algorithms
    _submodules['algorithms'] = algorithms
except ImportError as e:
    logger.warning(f"Failed to import algorithms module: {e}")

try:
    from . import chemistry
    _submodules['chemistry'] = chemistry
except ImportError as e:
    logger.warning(f"Failed to import chemistry module: {e}")

try:
    from . import many_body
    _submodules['many_body'] = many_body
except ImportError as e:
    logger.warning(f"Failed to import many_body module: {e}")

try:
    from . import field_theory
    _submodules['field_theory'] = field_theory
except ImportError as e:
    logger.warning(f"Failed to import field_theory module: {e}")

try:
    from . import ml
    _submodules['ml'] = ml
except ImportError as e:
    logger.warning(f"Failed to import ml module: {e}")

try:
    from . import visualization
    _submodules['visualization'] = visualization
except ImportError as e:
    logger.warning(f"Failed to import visualization module: {e}")

try:
    from . import utils
    _submodules['utils'] = utils
except ImportError as e:
    logger.warning(f"Failed to import utils module: {e}")

logger.info(f"Quantum submodules loaded: {list(_submodules.keys())}")

# Export main functions
__all__ = [
    'initialize_backends',
    'get_backend', 
    'list_backends',
] + list(_submodules.keys())
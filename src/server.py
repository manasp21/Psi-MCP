#!/usr/bin/env python3
"""
Psi-MCP: Advanced Quantum Systems MCP Server

This server provides comprehensive quantum computing and quantum physics tools
for complex open and closed quantum systems calculations.
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mcp.server.fastapi import FastMCPServer
from mcp import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration model
class ServerConfig(BaseModel):
    """Server configuration from environment variables and query parameters."""
    computing_backend: str = Field(default="simulator", description="Quantum computing backend")
    max_qubits: int = Field(default=20, ge=1, le=30, description="Maximum qubits for simulation")
    precision: str = Field(default="double", description="Numerical precision")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    timeout_seconds: int = Field(default=300, ge=10, le=1800, description="Execution timeout")
    memory_limit_gb: int = Field(default=4, ge=1, le=16, description="Memory limit in GB")

# Global configuration
config = ServerConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting Psi-MCP Quantum Systems Server...")
    logger.info(f"Configuration: {config.dict()}")
    
    # Initialize quantum backends
    try:
        from quantum import initialize_backends
        await initialize_backends(config)
        logger.info("Quantum backends initialized successfully")
    except ImportError as e:
        logger.error(f"Failed to import quantum module: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize quantum backends: {e}")
        logger.info("Server will continue with limited quantum functionality")
    
    yield
    
    logger.info("Shutting down Psi-MCP server...")

# Create FastAPI app with MCP server
app = FastMCPServer(
    "psi-mcp",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "psi-mcp",
        "version": "1.0.0",
        "config": config.dict()
    }

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current server configuration."""
    return config.dict()

@app.post("/config")
async def update_config(new_config: Dict[str, Any]):
    """Update server configuration."""
    global config
    try:
        config = ServerConfig(**new_config)
        logger.info(f"Configuration updated: {config.dict()}")
        return {"status": "updated", "config": config.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

# Quantum Circuit Tools
@app.tool()
async def create_quantum_circuit(
    num_qubits: int = Field(description="Number of qubits in the circuit"),
    circuit_type: str = Field(default="empty", description="Type of circuit to create"),
    backend: Optional[str] = Field(default=None, description="Quantum backend to use")
) -> str:
    """Create a quantum circuit with specified parameters."""
    try:
        if num_qubits > config.max_qubits:
            raise ValueError(f"Number of qubits ({num_qubits}) exceeds maximum ({config.max_qubits})")
        
        try:
            from quantum.circuits import create_circuit
            circuit = await create_circuit(num_qubits, circuit_type, backend or config.computing_backend)
            return f"Successfully created {circuit_type} quantum circuit with {num_qubits} qubits using {backend or config.computing_backend} backend."
        except ImportError:
            return f"Quantum circuit functionality not available - missing quantum computing libraries. Please install qiskit, cirq, or pennylane."
    
    except Exception as e:
        logger.error(f"Error creating quantum circuit: {e}")
        return f"Error creating quantum circuit: {str(e)}"

@app.tool()
async def simulate_quantum_circuit(
    circuit_definition: str = Field(description="Quantum circuit definition or ID"),
    shots: int = Field(default=1024, description="Number of measurement shots"),
    backend: Optional[str] = Field(default=None, description="Simulation backend")
) -> str:
    """Simulate a quantum circuit and return results."""
    try:
        from quantum.circuits import simulate_circuit
        results = await simulate_circuit(
            circuit_definition, 
            shots, 
            backend or config.computing_backend,
            timeout=config.timeout_seconds
        )
        
        return f"Simulation completed successfully. Results: {results}"
    
    except Exception as e:
        logger.error(f"Error simulating quantum circuit: {e}")
        return f"Error simulating quantum circuit: {str(e)}"

@app.tool()
async def optimize_quantum_circuit(
    circuit_definition: str = Field(description="Quantum circuit to optimize"),
    optimization_level: int = Field(default=1, ge=0, le=3, description="Optimization level (0-3)"),
    target_backend: Optional[str] = Field(default=None, description="Target backend for optimization")
) -> str:
    """Optimize a quantum circuit for better performance."""
    try:
        from quantum.circuits import optimize_circuit
        optimized_circuit = await optimize_circuit(
            circuit_definition,
            optimization_level,
            target_backend or config.computing_backend
        )
        
        return f"Circuit optimized successfully. Optimization level: {optimization_level}"
    
    except Exception as e:
        logger.error(f"Error optimizing quantum circuit: {e}")
        return f"Error optimizing quantum circuit: {str(e)}"

@app.tool()
async def extended_hubbard_simulation(
    system_size: int = Field(default=20, ge=2, le=100, description="System size"),
    t: float = Field(default=1.0, description="Hopping parameter"),
    U: float = Field(default=4.0, description="On-site Coulomb repulsion"),
    V: float = Field(default=1.0, description="Nearest-neighbor interaction"),
    mu: float = Field(default=0.0, description="Chemical potential")
) -> str:
    """Simulate extended Hubbard model with electron-electron interactions."""
    try:
        from quantum.many_body import extended_hubbard_model
        result = await extended_hubbard_model(system_size, t, U, V, mu)
        
        if result['success']:
            return f"Extended Hubbard simulation completed. Phase: {result['phase']}, Energy/site: {result['energy_per_site']:.4f}, Filling: {result['filling']:.3f}"
        else:
            return f"Extended Hubbard simulation failed: {result['error']}"
    except Exception as e:
        return f"Error in extended Hubbard simulation: {str(e)}"

@app.tool()
async def kitaev_model_simulation(
    Jx: float = Field(default=1.0, description="X-direction coupling"),
    Jy: float = Field(default=1.0, description="Y-direction coupling"),
    Jz: float = Field(default=1.0, description="Z-direction coupling"),
    system_size: int = Field(default=24, ge=6, le=50, description="System size")
) -> str:
    """Simulate Kitaev honeycomb model for quantum spin liquids."""
    try:
        from quantum.many_body import kitaev_honeycomb_model
        result = await kitaev_honeycomb_model(Jx, Jy, Jz, system_size)
        
        if result['success']:
            return f"Kitaev model simulation completed. Phase: {result['phase']}, Central charge: {result['central_charge']}"
        else:
            return f"Kitaev model simulation failed: {result['error']}"
    except Exception as e:
        return f"Error in Kitaev model simulation: {str(e)}"

@app.tool()
async def frustrated_magnet_analysis(
    model_type: str = Field(default="j1_j2_heisenberg", description="Type of frustrated model"),
    J1: float = Field(default=1.0, description="Nearest-neighbor coupling"),
    J2: float = Field(default=0.5, description="Next-nearest-neighbor coupling"),
    system_size: int = Field(default=50, ge=4, le=100, description="System size")
) -> str:
    """Analyze frustrated magnetic systems and quantum spin liquids."""
    try:
        from quantum.many_body import frustrated_magnets
        result = await frustrated_magnets(model_type, J1, J2, system_size)
        
        if result['success']:
            return f"Frustrated magnet analysis completed. Model: {model_type}, Phase: {result['phase']}, Frustration ratio: {result['frustration_ratio']:.3f}"
        else:
            return f"Frustrated magnet analysis failed: {result['error']}"
    except Exception as e:
        return f"Error in frustrated magnet analysis: {str(e)}"

@app.tool()
async def generate_phase_diagram(
    model_type: str = Field(default="ising", description="Physical model type"),
    parameter_min: float = Field(default=0.1, description="Minimum parameter value"),
    parameter_max: float = Field(default=2.0, description="Maximum parameter value"),
    num_points: int = Field(default=20, ge=5, le=50, description="Number of parameter points"),
    system_size: int = Field(default=30, ge=4, le=100, description="System size")
) -> str:
    """Generate quantum phase diagrams for various models."""
    try:
        import numpy as np
        from quantum.many_body import quantum_phase_diagram
        
        parameter_range = np.linspace(parameter_min, parameter_max, num_points).tolist()
        result = await quantum_phase_diagram(model_type, parameter_range, system_size)
        
        if result['success']:
            num_phases = len(set(result['phases_identified']))
            num_boundaries = len(result['phase_boundaries'])
            return f"Phase diagram generated for {model_type}. Found {num_phases} distinct phases with {num_boundaries} phase boundaries."
        else:
            return f"Phase diagram generation failed: {result['error']}"
    except Exception as e:
        return f"Error generating phase diagram: {str(e)}"

@app.tool()
async def excited_state_analysis(
    hamiltonian_type: str = Field(default="heisenberg", description="Hamiltonian type"),
    system_size: int = Field(default=30, ge=4, le=100, description="System size"),
    num_excited_states: int = Field(default=3, ge=1, le=10, description="Number of excited states"),
    max_bond_dimension: int = Field(default=100, ge=2, le=512, description="Maximum bond dimension")
) -> str:
    """Analyze excited states using advanced DMRG."""
    try:
        from quantum.tensor_networks import excited_state_dmrg
        result = await excited_state_dmrg(hamiltonian_type, system_size, max_bond_dimension, 10, num_excited_states)
        
        if result['success']:
            gaps = result['energy_gaps']
            return f"Excited state analysis completed. Found {len(gaps)} excited states with gaps: {[f'{g:.4f}' for g in gaps[:3]]}"
        else:
            return f"Excited state analysis failed: {result['error']}"
    except Exception as e:
        return f"Error in excited state analysis: {str(e)}"

@app.tool()
async def finite_temperature_calculation(
    hamiltonian_type: str = Field(default="heisenberg", description="Hamiltonian type"),
    temperature: float = Field(default=1.0, ge=0.01, le=10.0, description="Temperature"),
    system_size: int = Field(default=30, ge=4, le=100, description="System size"),
    time_steps: int = Field(default=50, ge=10, le=200, description="Imaginary time steps")
) -> str:
    """Calculate finite temperature properties using thermal DMRG."""
    try:
        from quantum.tensor_networks import finite_temperature_dmrg
        result = await finite_temperature_dmrg(hamiltonian_type, system_size, temperature, 100, 10, time_steps)
        
        if result['success']:
            final_energy = result['thermal_properties']['internal_energy'][-1] if result['thermal_properties']['internal_energy'] else 0
            return f"Finite temperature calculation completed at T={temperature}. Final internal energy: {final_energy:.4f}"
        else:
            return f"Finite temperature calculation failed: {result['error']}"
    except Exception as e:
        return f"Error in finite temperature calculation: {str(e)}"

@app.tool()
async def time_evolution_simulation(
    hamiltonian_type: str = Field(default="heisenberg", description="Hamiltonian type"),
    evolution_type: str = Field(default="real_time", description="Evolution type (real_time/imaginary_time)"),
    total_time: float = Field(default=5.0, ge=0.1, le=20.0, description="Total evolution time"),
    system_size: int = Field(default=30, ge=4, le=100, description="System size"),
    time_steps: int = Field(default=100, ge=10, le=500, description="Number of time steps")
) -> str:
    """Simulate quantum time evolution using TEBD."""
    try:
        from quantum.tensor_networks import tebd_evolution
        result = await tebd_evolution(hamiltonian_type, system_size, 100, total_time, time_steps, "ground", evolution_type)
        
        if result['success']:
            final_entropy = result['entanglement_entropies'][-1] if result['entanglement_entropies'] else 0
            return f"Time evolution simulation completed. Type: {evolution_type}, Final entanglement entropy: {final_entropy:.4f}"
        else:
            return f"Time evolution simulation failed: {result['error']}"
    except Exception as e:
        return f"Error in time evolution simulation: {str(e)}"

@app.tool()
async def infinite_system_calculation(
    hamiltonian_type: str = Field(default="heisenberg", description="Hamiltonian type"),
    unit_cell_size: int = Field(default=2, ge=1, le=4, description="Unit cell size"),
    max_bond_dimension: int = Field(default=100, ge=2, le=512, description="Maximum bond dimension"),
    num_sweeps: int = Field(default=20, ge=5, le=100, description="Number of iDMRG sweeps")
) -> str:
    """Calculate thermodynamic limit properties using infinite DMRG."""
    try:
        from quantum.tensor_networks import infinite_dmrg
        result = await infinite_dmrg(hamiltonian_type, unit_cell_size, max_bond_dimension, num_sweeps)
        
        if result['success']:
            energy_per_site = result['energy_per_site']
            correlation_length = result['correlation_length']
            return f"Infinite system calculation completed. Energy/site: {energy_per_site:.6f}, Correlation length: {correlation_length:.3f}"
        else:
            return f"Infinite system calculation failed: {result['error']}"
    except Exception as e:
        return f"Error in infinite system calculation: {str(e)}"

@app.tool()
async def batch_phase_calculation(
    hamiltonian_type: str = Field(default="heisenberg", description="Hamiltonian type"),
    parameter_values: str = Field(description="Comma-separated parameter values"),
    system_size: int = Field(default=20, ge=4, le=100, description="System size"),
    max_sweeps: int = Field(default=5, ge=1, le=20, description="Maximum DMRG sweeps per calculation")
) -> str:
    """Perform batch DMRG calculations for phase diagram generation."""
    try:
        from quantum.tensor_networks import batch_dmrg_calculations
        
        # Parse parameter values
        param_list = [float(x.strip()) for x in parameter_values.split(',')]
        
        result = await batch_dmrg_calculations(param_list, hamiltonian_type, system_size, 64, max_sweeps)
        
        if result['success']:
            num_calcs = len(result['energies'])
            transitions = len(result.get('phase_transitions', []))
            gpu_used = result.get('acceleration_used', False)
            return f"Batch calculation completed: {num_calcs} points, {transitions} phase transitions detected. GPU acceleration: {'Yes' if gpu_used else 'No'}"
        else:
            return f"Batch calculation failed: {result['error']}"
    except Exception as e:
        return f"Error in batch calculation: {str(e)}"

@app.tool()
async def solve_master_equation(
    hamiltonian: str = Field(description="System Hamiltonian definition"),
    collapse_operators: str = Field(description="Collapse operators for dissipation"),
    initial_state: str = Field(description="Initial quantum state"),
    time_span: str = Field(description="Time evolution span (start, end, steps)"),
    solver_method: str = Field(default="mesolve", description="Master equation solver method")
) -> str:
    """Solve the master equation for open quantum systems."""
    try:
        try:
            from quantum.systems import solve_master_eq
            result = await solve_master_eq(
                hamiltonian,
                collapse_operators,
                initial_state,
                time_span,
                solver_method,
                config.precision
            )
            return f"Master equation solved successfully using {solver_method} method."
        except ImportError:
            return f"Open quantum systems functionality not available - missing QuTiP library. Please install qutip."
    
    except Exception as e:
        logger.error(f"Error solving master equation: {e}")
        return f"Error solving master equation: {str(e)}"

@app.tool()
async def visualize_quantum_state(
    state_definition: str = Field(description="Quantum state to visualize"),
    visualization_type: str = Field(default="bloch_sphere", description="Type of visualization"),
    save_path: Optional[str] = Field(default=None, description="Path to save the visualization")
) -> str:
    """Visualize quantum states using various methods."""
    try:
        try:
            from quantum.visualization import visualize_state
            plot_data = await visualize_state(
                state_definition,
                visualization_type,
                save_path
            )
            return f"Visualization created successfully: {visualization_type}"
        except ImportError:
            return f"Quantum visualization functionality not available - missing matplotlib and quantum libraries. Please install matplotlib, qutip, and quantum computing libraries."
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return f"Error creating visualization: {str(e)}"

# Configuration parsing for Smithery
@app.middleware("http")
async def parse_config_middleware(request: Request, call_next):
    """Parse configuration from query parameters (Smithery format)."""
    global config
    
    # Extract configuration from query parameters
    query_params = dict(request.query_params)
    config_updates = {}
    
    # Parse dot notation parameters
    for key, value in query_params.items():
        if key.startswith('config.'):
            config_key = key[7:]  # Remove 'config.' prefix
            
            # Type conversion based on field
            if config_key in ['max_qubits', 'timeout_seconds', 'memory_limit_gb']:
                config_updates[config_key] = int(value)
            elif config_key == 'enable_gpu':
                config_updates[config_key] = value.lower() in ('true', '1', 'yes')
            else:
                config_updates[config_key] = value
    
    # Update configuration if new parameters found
    if config_updates:
        try:
            config = ServerConfig(**{**config.dict(), **config_updates})
            logger.info(f"Configuration updated from query params: {config_updates}")
        except Exception as e:
            logger.warning(f"Invalid configuration in query params: {e}")
    
    response = await call_next(request)
    return response

def main():
    """Main entry point for the server."""
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Psi-MCP server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False
    )

if __name__ == "__main__":
    main()
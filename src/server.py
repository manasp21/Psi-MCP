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
    except Exception as e:
        logger.error(f"Failed to initialize quantum backends: {e}")
    
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
        
        from quantum.circuits import create_circuit
        circuit = await create_circuit(num_qubits, circuit_type, backend or config.computing_backend)
        
        return f"Successfully created {circuit_type} quantum circuit with {num_qubits} qubits using {backend or config.computing_backend} backend."
    
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
async def solve_master_equation(
    hamiltonian: str = Field(description="System Hamiltonian definition"),
    collapse_operators: str = Field(description="Collapse operators for dissipation"),
    initial_state: str = Field(description="Initial quantum state"),
    time_span: str = Field(description="Time evolution span (start, end, steps)"),
    solver_method: str = Field(default="mesolve", description="Master equation solver method")
) -> str:
    """Solve the master equation for open quantum systems."""
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
        from quantum.visualization import visualize_state
        plot_data = await visualize_state(
            state_definition,
            visualization_type,
            save_path
        )
        
        return f"Visualization created successfully: {visualization_type}"
    
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
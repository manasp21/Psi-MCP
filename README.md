# Psi-MCP: Advanced Quantum Systems MCP Server

<div align="center">

![Quantum Computing](https://img.shields.io/badge/Quantum-Computing-blue?style=for-the-badge)
![MCP Server](https://img.shields.io/badge/MCP-Server-green?style=for-the-badge)
![Smithery Compatible](https://img.shields.io/badge/Smithery-Compatible-purple?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-red?style=for-the-badge)

*The most comprehensive quantum physics MCP server for complex open and closed quantum systems calculations*

</div>

## 🌟 Overview

Psi-MCP is an advanced Model Context Protocol (MCP) server specifically designed for quantum systems analysis and simulation. It provides comprehensive tools for quantum computing, quantum chemistry, many-body physics, quantum machine learning, and quantum field theory calculations.

### Key Features

- 🔬 **Quantum Circuit Operations**: Create, simulate, optimize, and visualize quantum circuits
- ⚛️ **Open Quantum Systems**: Solve master equations, analyze decoherence, compute steady states
- 🧪 **Quantum Chemistry**: Molecular Hamiltonians, VQE, electronic structure calculations
- 🔗 **Many-Body Physics**: DMRG, tensor networks, phase transitions, correlation functions
- 🤖 **Quantum Machine Learning**: QNNs, variational classifiers, quantum kernels
- 🌊 **Quantum Field Theory**: Field quantization, path integrals, RG flow, anomalies
- 📊 **Advanced Visualization**: Bloch spheres, density matrices, Wigner functions
- 🚀 **Smithery Compatible**: Easy deployment and integration

## 🛠 Installation

### Prerequisites

- Python 3.11 or higher
- Docker (for containerized deployment)
- Git

### Core vs Optional Dependencies

**Core Dependencies (always installed):**
- FastAPI, Uvicorn (MCP server framework)
- Qiskit, Cirq, PennyLane (quantum computing)
- QuTiP (open quantum systems)
- OpenFermion (quantum chemistry)
- NumPy, SciPy, Matplotlib (numerical computing)

**Optional Dependencies (install separately if needed):**
- PySCF (advanced quantum chemistry)
- TensorFlow Quantum (quantum ML)
- NetKet (neural quantum states)
- Additional quantum libraries

### Quick Start with Smithery

```bash
# Install via Smithery CLI
npx @smithery/cli install psi-mcp --client cursor

# Or deploy via GitHub integration
git clone https://github.com/manasp21/Psi-MCP.git
cd Psi-MCP
# Push to your GitHub repository and connect to Smithery
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/manasp21/Psi-MCP.git
cd Psi-MCP

# Install dependencies
pip install -r requirements.txt

# Run the server
python src/server.py
```

### Docker Deployment

```bash
# Build the container
docker build -t psi-mcp .

# Run with configuration
docker run -p 8000:8000 \
  -e computing_backend=simulator \
  -e max_qubits=20 \
  -e precision=double \
  psi-mcp
```

## 🔧 Configuration

### Smithery Configuration

Configure via the Smithery dashboard or query parameters:

```yaml
computing_backend: "simulator"  # qasm_simulator, statevector_simulator
max_qubits: 20                  # Maximum qubits (1-30)
precision: "double"             # single, double, extended
enable_gpu: false               # GPU acceleration
timeout_seconds: 300            # Calculation timeout
memory_limit_gb: 4              # Memory limit
```

### Environment Variables

```bash
PORT=8000                       # Server port
HOST=0.0.0.0                   # Server host
COMPUTING_BACKEND=simulator     # Default backend
MAX_QUBITS=20                  # Default max qubits
```

## 🚀 Usage

### Quantum Circuit Operations

#### Create Quantum Circuits

```python
# Create a Bell state circuit
create_quantum_circuit(
    num_qubits=2,
    circuit_type="bell",
    backend="qasm_simulator"
)

# Create a GHZ state
create_quantum_circuit(
    num_qubits=4,
    circuit_type="ghz",
    backend="statevector_simulator"
)

# Create quantum Fourier transform
create_quantum_circuit(
    num_qubits=3,
    circuit_type="qft",
    backend="simulator"
)
```

#### Simulate Circuits

```python
# Simulate with measurements
simulate_quantum_circuit(
    circuit_definition="circuit_1",
    shots=1024,
    backend="qasm_simulator"
)

# Get statevector
simulate_quantum_circuit(
    circuit_definition="circuit_2",
    shots=1,
    backend="statevector_simulator"
)
```

#### Optimize Circuits

```python
# Optimize for specific backend
optimize_quantum_circuit(
    circuit_definition="circuit_1",
    optimization_level=2,
    target_backend="qasm_simulator"
)
```

### Open Quantum Systems

#### Master Equation Solving

```python
# Solve Lindblad master equation
solve_master_equation(
    hamiltonian="pauli_z",
    collapse_operators="spontaneous_emission",
    initial_state="excited",
    time_span="0,10,100",
    solver_method="mesolve"
)

# Analyze decoherence
analyze_decoherence(
    system_hamiltonian="pauli_x",
    environment_coupling="dephasing",
    temperature=0.1,
    analysis_type="dephasing"
)
```

### Quantum Chemistry

#### Molecular Calculations

```python
# Generate molecular Hamiltonian
generate_molecular_hamiltonian(
    molecule="H2",
    basis="sto-3g",
    charge=0,
    multiplicity=1
)

# Run VQE for electronic structure
vqe_chemistry(
    molecule="H2O",
    basis="6-31g",
    ansatz="uccsd",
    optimizer="cobyla"
)

# Simulate chemical reactions
simulate_chemical_reaction(
    reactants=["H2", "O2"],
    products=["H2O"],
    method="vqe"
)
```

### Quantum Algorithms

#### Shor's Algorithm

```python
# Factor integers
shors_algorithm(
    N=15,
    backend="qasm_simulator",
    shots=1024
)
```

#### Grover's Search

```python
# Search marked items
grovers_search(
    marked_items=[3, 7],
    search_space_size=16,
    backend="simulator"
)
```

#### VQE Optimization

```python
# Variational quantum eigensolver
vqe_optimization(
    hamiltonian="ising",
    ansatz_type="ry",
    optimizer="cobyla",
    max_iterations=100
)
```

### Many-Body Physics

#### DMRG Simulations

```python
# Run DMRG for spin chains
dmrg_simulation(
    hamiltonian_type="heisenberg",
    system_size=20,
    bond_dimension=100,
    max_sweeps=10
)

# Phase transition analysis
phase_transition_analysis(
    model_type="ising",
    parameter_range=[0.0, 2.0],
    n_points=20,
    system_size=16
)
```

### Quantum Machine Learning

#### Neural Networks

```python
# Train quantum neural network
quantum_neural_network(
    input_data=[[0.1, 0.2], [0.3, 0.4]],
    labels=[0, 1],
    n_qubits=4,
    n_layers=2,
    epochs=50
)

# Variational classifier
variational_classifier(
    training_data=train_X,
    training_labels=train_y,
    test_data=test_X,
    ansatz_type="hardware_efficient"
)
```

### Visualization

#### Quantum States

```python
# Bloch sphere visualization
visualize_quantum_state(
    state_definition="superposition",
    visualization_type="bloch_sphere"
)

# Density matrix plot
visualize_quantum_state(
    state_definition="bell",
    visualization_type="density_matrix"
)

# Wigner function
visualize_quantum_state(
    state_definition="coherent",
    visualization_type="wigner_function"
)
```

## 📚 API Reference

### Core Tools

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `create_quantum_circuit` | Create quantum circuits | `num_qubits`, `circuit_type`, `backend` |
| `simulate_quantum_circuit` | Simulate circuits | `circuit_definition`, `shots`, `backend` |
| `solve_master_equation` | Solve open system dynamics | `hamiltonian`, `collapse_operators`, `initial_state` |
| `vqe_optimization` | Variational quantum eigensolver | `hamiltonian`, `ansatz_type`, `optimizer` |
| `dmrg_simulation` | Many-body simulations | `hamiltonian_type`, `system_size`, `bond_dimension` |
| `quantum_neural_network` | Train QNNs | `input_data`, `labels`, `n_qubits`, `n_layers` |

### Supported Backends

- **Qiskit**: `qasm_simulator`, `statevector_simulator`, `unitary_simulator`
- **Cirq**: `cirq_simulator`
- **PennyLane**: `default.qubit`, `default.qubit.torch`

### Circuit Types

- `empty`: Empty circuit
- `bell`: Bell state preparation
- `ghz`: GHZ state preparation
- `qft`: Quantum Fourier transform
- `random`: Random circuit

## 🏗 Architecture

```
Psi-MCP/
├── src/
│   ├── server.py              # Main MCP server
│   └── quantum/               # Quantum modules
│       ├── __init__.py        # Backend initialization
│       ├── circuits.py        # Circuit operations
│       ├── systems.py         # Open quantum systems
│       ├── algorithms.py      # Quantum algorithms
│       ├── chemistry.py       # Quantum chemistry
│       ├── many_body.py       # Many-body physics
│       ├── field_theory.py    # Quantum field theory
│       ├── ml.py             # Quantum ML
│       ├── visualization.py   # Visualization tools
│       └── utils.py          # Utility functions
├── tests/                     # Test suite
├── docs/                      # Documentation
├── smithery.yaml             # Smithery configuration
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🧪 Examples

### Complete Workflow Example

```python
# 1. Create and simulate a quantum circuit
circuit_result = create_quantum_circuit(
    num_qubits=3,
    circuit_type="ghz",
    backend="qasm_simulator"
)

# 2. Simulate the circuit
simulation_result = simulate_quantum_circuit(
    circuit_definition=circuit_result['id'],
    shots=1000,
    backend="qasm_simulator"
)

# 3. Visualize the results
plot_result = plot_measurement_results(
    counts=simulation_result['counts'],
    title="GHZ State Measurement"
)

# 4. Analyze entanglement
entropy = calculate_entanglement_entropy(
    circuit_definition=circuit_result['id'],
    subsystem_size=1
)
```

### Quantum Chemistry Workflow

```python
# 1. Generate molecular Hamiltonian
hamiltonian = generate_molecular_hamiltonian(
    molecule="H2",
    basis="sto-3g"
)

# 2. Run VQE calculation
vqe_result = vqe_chemistry(
    molecule="H2",
    basis="sto-3g",
    ansatz="uccsd"
)

# 3. Compute molecular properties
properties = compute_molecular_properties(
    molecule="H2",
    method="hf",
    basis="sto-3g"
)
```

## 🔬 Advanced Features

### Custom Hamiltonians

```python
# Define custom spin chain
solve_master_equation(
    hamiltonian=json.dumps([[1, 0], [0, -1]]),
    collapse_operators="custom_operators",
    initial_state="custom_state"
)
```

### GPU Acceleration

```python
# Enable GPU support
configure_server(
    enable_gpu=True,
    computing_backend="gpu_simulator"
)
```

### Parallel Processing

```python
# Parallel circuit simulation
simulate_circuits_parallel(
    circuit_definitions=["circuit_1", "circuit_2", "circuit_3"],
    shots=1000
)
```

## 📊 Performance

### Benchmarks

| Operation | System Size | Execution Time | Memory Usage |
|-----------|-------------|----------------|--------------|
| Circuit Simulation | 20 qubits | ~2s | ~100MB |
| VQE Optimization | H2O molecule | ~30s | ~200MB |
| DMRG Calculation | 50 sites | ~60s | ~500MB |
| QNN Training | 100 samples | ~45s | ~150MB |

### Scaling

- **Quantum Circuits**: Up to 30 qubits (simulator dependent)
- **Many-Body Systems**: Up to 100 sites with DMRG
- **Molecular Systems**: Up to 20 orbitals with VQE
- **ML Training**: Up to 1000 samples efficiently

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install development dependencies
git clone https://github.com/manasp21/Psi-MCP.git
cd Psi-MCP
pip install -r requirements.txt
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qiskit Team** for quantum computing framework
- **QuTiP Developers** for open quantum systems tools
- **PennyLane Team** for quantum machine learning
- **OpenFermion Contributors** for quantum chemistry tools
- **Smithery Platform** for MCP server hosting

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/manasp21/Psi-MCP/issues)
- **Documentation**: [Full API documentation](https://github.com/manasp21/Psi-MCP/docs)
- **Examples**: [Jupyter notebooks with examples](https://github.com/manasp21/Psi-MCP/examples)

## 🗺 Roadmap

### v1.1.0 (Next Release)
- [ ] ITensor integration for tensor networks
- [ ] NetKet support for neural quantum states
- [ ] Advanced error mitigation tools
- [ ] Quantum error correction codes

### v1.2.0 (Future)
- [ ] Hardware backend support (IBM, Google, IonQ)
- [ ] Advanced visualization dashboard
- [ ] Quantum advantage benchmarks
- [ ] Multi-user collaboration features

---

<div align="center">

**Built with ❤️ for the quantum computing community**

[🌟 Star us on GitHub](https://github.com/manasp21/Psi-MCP) | [📖 Read the Docs](https://github.com/manasp21/Psi-MCP/docs) | [🚀 Deploy on Smithery](https://smithery.ai)

</div>
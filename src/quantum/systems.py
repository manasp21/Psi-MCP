"""
Quantum Systems Module

This module provides functionality for open and closed quantum systems,
including master equation solving, decoherence analysis, and system dynamics.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

async def solve_master_eq(
    hamiltonian: str,
    collapse_operators: str,
    initial_state: str,
    time_span: str,
    solver_method: str = "mesolve",
    precision: str = "double"
) -> Dict[str, Any]:
    """
    Solve the master equation for open quantum systems.
    
    Args:
        hamiltonian: System Hamiltonian definition
        collapse_operators: Collapse operators for dissipation
        initial_state: Initial quantum state
        time_span: Time evolution span (start,end,steps)
        solver_method: Solver method (mesolve, sesolve, mcsolve)
        precision: Numerical precision
        
    Returns:
        Evolution results
    """
    logger.info(f"Solving master equation using {solver_method}")
    
    try:
        # Import QuTiP
        import qutip as qt
        
        # Parse inputs
        H = _parse_hamiltonian(hamiltonian)
        c_ops = _parse_collapse_operators(collapse_operators)
        psi0 = _parse_initial_state(initial_state)
        times = _parse_time_span(time_span)
        
        # Set precision
        if precision == "single":
            qt.settings.atol = 1e-6
            qt.settings.rtol = 1e-6
        elif precision == "extended":
            qt.settings.atol = 1e-14
            qt.settings.rtol = 1e-14
        
        # Solve based on method
        if solver_method == "mesolve":
            result = qt.mesolve(H, psi0, times, c_ops)
        elif solver_method == "sesolve":
            result = qt.sesolve(H, psi0, times)
        elif solver_method == "mcsolve":
            result = qt.mcsolve(H, psi0, times, c_ops, ntraj=100)
        else:
            raise ValueError(f"Unknown solver method: {solver_method}")
        
        # Process results
        return {
            'success': True,
            'solver': solver_method,
            'times': times.tolist(),
            'num_states': len(result.states),
            'final_fidelity': qt.fidelity(result.states[-1], psi0) if result.states else 0,
            'expectation_values': _compute_expectation_values(result),
            'entropy': _compute_entropy(result.states[-1]) if result.states else 0
        }
        
    except Exception as e:
        logger.error(f"Error solving master equation: {e}")
        return {'success': False, 'error': str(e)}

def _parse_hamiltonian(hamiltonian_str: str):
    """Parse Hamiltonian from string definition."""
    import qutip as qt
    
    # Handle common cases
    if hamiltonian_str.lower() == "pauli_x":
        return qt.sigmax()
    elif hamiltonian_str.lower() == "pauli_y":
        return qt.sigmay()
    elif hamiltonian_str.lower() == "pauli_z":
        return qt.sigmaz()
    elif hamiltonian_str.lower() == "harmonic_oscillator":
        return qt.num(20)  # 20 levels
    elif "spin_chain" in hamiltonian_str.lower():
        # Simple spin chain
        N = 4  # Default chain length
        H = 0
        for i in range(N-1):
            H += qt.tensor([qt.sigmax() if j==i else qt.qeye(2) for j in range(N)]) * \
                 qt.tensor([qt.sigmax() if j==i+1 else qt.qeye(2) for j in range(N)])
        return H
    else:
        # Try to parse as matrix
        try:
            import json
            matrix_data = json.loads(hamiltonian_str)
            return qt.Qobj(np.array(matrix_data))
        except:
            # Default to Pauli-Z
            return qt.sigmaz()

def _parse_collapse_operators(collapse_str: str):
    """Parse collapse operators from string definition."""
    import qutip as qt
    
    collapse_ops = []
    
    if not collapse_str or collapse_str.lower() == "none":
        return collapse_ops
    
    # Common cases
    if "spontaneous_emission" in collapse_str.lower():
        collapse_ops.append(qt.sigmam())
    elif "dephasing" in collapse_str.lower():
        collapse_ops.append(qt.sigmaz())
    elif "thermal" in collapse_str.lower():
        collapse_ops.append(qt.destroy(20))  # Bosonic
        collapse_ops.append(qt.create(20))
    
    return collapse_ops

def _parse_initial_state(state_str: str):
    """Parse initial state from string definition."""
    import qutip as qt
    
    if state_str.lower() == "ground":
        return qt.basis(2, 0)
    elif state_str.lower() == "excited":
        return qt.basis(2, 1)
    elif state_str.lower() == "superposition":
        return (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    elif state_str.lower() == "coherent":
        return qt.coherent(20, 1.0)  # Alpha = 1
    else:
        # Default to ground state
        return qt.basis(2, 0)

def _parse_time_span(time_str: str):
    """Parse time span from string definition."""
    try:
        parts = time_str.split(',')
        if len(parts) == 3:
            start, end, steps = map(float, parts)
            return np.linspace(start, end, int(steps))
        else:
            # Default time span
            return np.linspace(0, 10, 100)
    except:
        return np.linspace(0, 10, 100)

def _compute_expectation_values(result):
    """Compute expectation values for common observables."""
    import qutip as qt
    
    try:
        # Pauli matrices for 2-level system
        sx_expect = qt.expect(qt.sigmax(), result.states)
        sy_expect = qt.expect(qt.sigmay(), result.states)
        sz_expect = qt.expect(qt.sigmaz(), result.states)
        
        return {
            'sigma_x': sx_expect.tolist() if hasattr(sx_expect, 'tolist') else [float(sx_expect)],
            'sigma_y': sy_expect.tolist() if hasattr(sy_expect, 'tolist') else [float(sy_expect)],
            'sigma_z': sz_expect.tolist() if hasattr(sz_expect, 'tolist') else [float(sz_expect)]
        }
    except:
        return {}

def _compute_entropy(state):
    """Compute von Neumann entropy of a quantum state."""
    import qutip as qt
    
    try:
        if state.type == 'ket':
            rho = state * state.dag()
        else:
            rho = state
        
        return qt.entropy_vn(rho)
    except:
        return 0.0

async def compute_steady_state(
    hamiltonian: str,
    collapse_operators: str,
    method: str = "direct"
) -> Dict[str, Any]:
    """
    Compute the steady state of an open quantum system.
    
    Args:
        hamiltonian: System Hamiltonian
        collapse_operators: Dissipation operators
        method: Solution method
        
    Returns:
        Steady state information
    """
    logger.info(f"Computing steady state using {method} method")
    
    try:
        import qutip as qt
        
        H = _parse_hamiltonian(hamiltonian)
        c_ops = _parse_collapse_operators(collapse_operators)
        
        # Compute steady state
        if method == "direct":
            rho_ss = qt.steadystate(H, c_ops)
        elif method == "iterative":
            rho_ss = qt.steadystate(H, c_ops, method='iterative-gmres')
        else:
            rho_ss = qt.steadystate(H, c_ops)
        
        # Analyze steady state
        purity = qt.expect(rho_ss, rho_ss)
        entropy = qt.entropy_vn(rho_ss)
        
        return {
            'success': True,
            'purity': float(purity),
            'entropy': float(entropy),
            'trace': float(rho_ss.tr()),
            'eigenvalues': [float(x) for x in rho_ss.eigenenergies()[:5]],  # Top 5
            'method': method
        }
        
    except Exception as e:
        logger.error(f"Error computing steady state: {e}")
        return {'success': False, 'error': str(e)}

async def analyze_decoherence(
    system_hamiltonian: str,
    environment_coupling: str,
    temperature: float = 0.0,
    analysis_type: str = "dephasing"
) -> Dict[str, Any]:
    """
    Analyze decoherence effects in quantum systems.
    
    Args:
        system_hamiltonian: System Hamiltonian
        environment_coupling: System-environment coupling
        temperature: Environment temperature
        analysis_type: Type of decoherence analysis
        
    Returns:
        Decoherence analysis results
    """
    logger.info(f"Analyzing {analysis_type} decoherence at T={temperature}")
    
    try:
        import qutip as qt
        
        # Simple decoherence model
        H_sys = _parse_hamiltonian(system_hamiltonian)
        
        # Create decoherence operators based on type
        if analysis_type == "dephasing":
            gamma_dephasing = 0.1
            c_ops = [np.sqrt(gamma_dephasing) * qt.sigmaz()]
        elif analysis_type == "relaxation":
            gamma_relax = 0.05
            c_ops = [np.sqrt(gamma_relax) * qt.sigmam()]
        else:
            # Combined
            c_ops = [np.sqrt(0.1) * qt.sigmaz(), np.sqrt(0.05) * qt.sigmam()]
        
        # Add thermal effects if temperature > 0
        if temperature > 0:
            n_th = 1 / (np.exp(1.0 / temperature) - 1)  # Simplified
            c_ops.append(np.sqrt(n_th + 1) * qt.sigmam())
            c_ops.append(np.sqrt(n_th) * qt.sigmap())
        
        # Compute decoherence times
        times = np.linspace(0, 10, 100)
        psi0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        
        result = qt.mesolve(H_sys, psi0, times, c_ops)
        
        # Compute coherence measures
        coherences = []
        for state in result.states:
            if state.type == 'ket':
                rho = state * state.dag()
            else:
                rho = state
            coherences.append(abs(rho[0, 1]))
        
        # Fit exponential decay
        coherences = np.array(coherences)
        try:
            from scipy.optimize import curve_fit
            def exp_decay(t, A, gamma):
                return A * np.exp(-gamma * t)
            
            popt, _ = curve_fit(exp_decay, times, coherences)
            decoherence_time = 1 / popt[1] if popt[1] > 0 else float('inf')
        except:
            decoherence_time = None
        
        return {
            'success': True,
            'analysis_type': analysis_type,
            'temperature': temperature,
            'decoherence_time': decoherence_time,
            'final_coherence': float(coherences[-1]),
            'coherence_decay': coherences.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing decoherence: {e}")
        return {'success': False, 'error': str(e)}
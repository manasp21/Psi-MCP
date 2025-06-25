"""
Quantum Visualization Module

This module provides comprehensive visualization capabilities for quantum states,
circuits, and simulation results.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64

logger = logging.getLogger(__name__)

# Set up matplotlib for non-interactive use
plt.switch_backend('Agg')
sns.set_style("whitegrid")

async def visualize_state(
    state_definition: str,
    visualization_type: str = "bloch_sphere",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Visualize quantum states using various methods.
    
    Args:
        state_definition: Quantum state definition
        visualization_type: Type of visualization
        save_path: Path to save visualization
        
    Returns:
        Visualization results
    """
    logger.info(f"Creating {visualization_type} visualization")
    
    try:
        # Parse state
        state = _parse_quantum_state(state_definition)
        
        if visualization_type == "bloch_sphere":
            return await _create_bloch_sphere(state, save_path)
        elif visualization_type == "density_matrix":
            return await _create_density_matrix_plot(state, save_path)
        elif visualization_type == "wigner_function":
            return await _create_wigner_plot(state, save_path)
        elif visualization_type == "bar_plot":
            return await _create_state_bar_plot(state, save_path)
        elif visualization_type == "phase_plot":
            return await _create_phase_plot(state, save_path)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
            
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return {'success': False, 'error': str(e)}

def _parse_quantum_state(state_definition: str):
    """Parse quantum state from string definition."""
    import qutip as qt
    
    if state_definition.lower() == "ground":
        return qt.basis(2, 0)
    elif state_definition.lower() == "excited":
        return qt.basis(2, 1)  
    elif state_definition.lower() == "superposition":
        return (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    elif state_definition.lower() == "bell":
        return (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + 
                qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
    elif state_definition.lower() == "coherent":
        return qt.coherent(20, 1.0)
    elif state_definition.lower() == "thermal":
        return qt.thermal_dm(20, 1.0)
    else:
        # Try to parse as JSON array
        try:
            import json
            state_data = json.loads(state_definition)
            return qt.Qobj(np.array(state_data))
        except:
            # Default to superposition
            return (qt.basis(2, 0) + qt.basis(2, 1)).unit()

async def _create_bloch_sphere(state, save_path: Optional[str]) -> Dict[str, Any]:
    """Create Bloch sphere visualization."""
    import qutip as qt
    
    try:
        # Convert to density matrix if needed
        if state.type == 'ket':
            rho = state * state.dag()
        else:
            rho = state
        
        # Extract Bloch vector for 2-level system
        if rho.shape[0] == 2:
            # Pauli matrices
            sx = qt.sigmax()
            sy = qt.sigmay() 
            sz = qt.sigmaz()
            
            # Bloch vector components
            x = (rho * sx).tr().real
            y = (rho * sy).tr().real
            z = (rho * sz).tr().real
            
            # Create Bloch sphere
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Draw sphere
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
            
            # Draw axes
            ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
            ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
            ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
            
            # Draw state vector
            ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
            
            # Labels
            ax.text(1.1, 0, 0, '|+⟩', fontsize=12)
            ax.text(-1.1, 0, 0, '|-⟩', fontsize=12)
            ax.text(0, 1.1, 0, '|+i⟩', fontsize=12)
            ax.text(0, -1.1, 0, '|-i⟩', fontsize=12)
            ax.text(0, 0, 1.1, '|0⟩', fontsize=12)
            ax.text(0, 0, -1.1, '|1⟩', fontsize=12)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Bloch Sphere Representation')
            
            # Save or return as base64
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return {'success': True, 'saved_path': save_path, 'bloch_vector': [x, y, z]}
            else:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return {
                    'success': True,
                    'bloch_vector': [float(x), float(y), float(z)],
                    'image_base64': image_base64,
                    'purity': float((rho * rho).tr().real)
                }
        else:
            return {'success': False, 'error': 'Bloch sphere only supports 2-level systems'}
            
    except Exception as e:
        logger.error(f"Error creating Bloch sphere: {e}")
        return {'success': False, 'error': str(e)}

async def _create_density_matrix_plot(state, save_path: Optional[str]) -> Dict[str, Any]:
    """Create density matrix heatmap."""
    import qutip as qt
    
    try:
        # Convert to density matrix
        if state.type == 'ket':
            rho = state * state.dag()
        else:
            rho = state
        
        # Create figure with subplots for real and imaginary parts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Real part
        im1 = ax1.imshow(rho.data.real, cmap='RdBu_r', interpolation='nearest')
        ax1.set_title('Real Part')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1)
        
        # Imaginary part
        im2 = ax2.imshow(rho.data.imag, cmap='RdBu_r', interpolation='nearest')
        ax2.set_title('Imaginary Part')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        # Calculate properties
        purity = (rho * rho).tr().real
        trace = rho.tr().real
        entropy = qt.entropy_vn(rho)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return {
                'success': True,
                'saved_path': save_path,
                'purity': float(purity),
                'trace': float(trace),
                'entropy': float(entropy)
            }
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                'success': True,
                'image_base64': image_base64,
                'purity': float(purity),
                'trace': float(trace),
                'entropy': float(entropy)
            }
            
    except Exception as e:
        logger.error(f"Error creating density matrix plot: {e}")
        return {'success': False, 'error': str(e)}

async def _create_state_bar_plot(state, save_path: Optional[str]) -> Dict[str, Any]:
    """Create bar plot of state amplitudes."""
    try:
        # Get state vector
        if state.type == 'ket':
            amplitudes = state.data.toarray().flatten()
        else:
            # For density matrix, show diagonal elements
            amplitudes = np.diag(state.data.toarray())
        
        n_states = len(amplitudes)
        indices = range(n_states)
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Amplitude magnitudes
        magnitudes = np.abs(amplitudes)
        ax1.bar(indices, magnitudes, alpha=0.7, color='blue')
        ax1.set_ylabel('|Amplitude|')
        ax1.set_title('State Amplitude Magnitudes')
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f'|{i}⟩' for i in indices])
        
        # Phases
        phases = np.angle(amplitudes)
        ax2.bar(indices, phases, alpha=0.7, color='red')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_xlabel('Basis State')
        ax2.set_title('State Phases')
        ax2.set_xticks(indices)
        ax2.set_xticklabels([f'|{i}⟩' for i in indices])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return {'success': True, 'saved_path': save_path}
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                'success': True,
                'image_base64': image_base64,
                'amplitudes': amplitudes.tolist(),
                'probabilities': (magnitudes**2).tolist()
            }
            
    except Exception as e:
        logger.error(f"Error creating bar plot: {e}")
        return {'success': False, 'error': str(e)}

async def visualize_circuit(
    circuit_id: str,
    style: str = "default"
) -> Dict[str, Any]:
    """
    Visualize quantum circuits.
    
    Args:
        circuit_id: Circuit identifier
        style: Visualization style
        
    Returns:
        Circuit visualization
    """
    logger.info(f"Visualizing circuit {circuit_id} with {style} style")
    
    try:
        from quantum.circuits import circuit_manager
        
        if circuit_id not in circuit_manager.circuits:
            return {'success': False, 'error': f'Circuit {circuit_id} not found'}
        
        circuit_data = circuit_manager.circuits[circuit_id]
        circuit = circuit_data['circuit']
        
        # Use Qiskit visualization
        from qiskit.visualization import circuit_drawer
        import matplotlib.pyplot as plt
        
        fig = circuit_drawer(circuit, output='mpl', style=style)
        
        # Convert to base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'success': True,
            'circuit_id': circuit_id,
            'image_base64': image_base64,
            'circuit_info': circuit_data['info']
        }
        
    except Exception as e:
        logger.error(f"Error visualizing circuit: {e}")
        return {'success': False, 'error': str(e)}

async def plot_measurement_results(
    counts: Dict[str, int],
    title: str = "Measurement Results"
) -> Dict[str, Any]:
    """
    Plot measurement results as bar chart.
    
    Args:
        counts: Measurement counts
        title: Plot title
        
    Returns:
        Plot data
    """
    try:
        # Sort by bit string
        sorted_counts = dict(sorted(counts.items()))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        states = list(sorted_counts.keys())
        values = list(sorted_counts.values())
        total_shots = sum(values)
        
        # Create bar plot
        bars = ax.bar(states, values, alpha=0.7)
        
        # Add probability labels
        for i, (state, count) in enumerate(sorted_counts.items()):
            probability = count / total_shots
            ax.text(i, count + max(values) * 0.01, 
                   f'{probability:.3f}', 
                   ha='center', va='bottom')
        
        ax.set_xlabel('Measurement Outcome')
        ax.set_ylabel('Counts')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'success': True,
            'image_base64': image_base64,
            'total_shots': total_shots,
            'unique_outcomes': len(states)
        }
        
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        return {'success': False, 'error': str(e)}
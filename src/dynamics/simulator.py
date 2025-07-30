"""Trajectory simulation and data generation."""

import numpy as np
import pickle
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm

from .systems import DynamicalSystem, create_system
from ..utils.config import Config


class TrajectorySimulator:
    """Simulate trajectories from dynamical systems."""
    
    def __init__(self, system: DynamicalSystem, config: Config):
        """Initialize trajectory simulator.
        
        Args:
            system: Dynamical system to simulate
            config: Configuration object
        """
        self.system = system
        self.config = config
        self.trajectory_config = config['trajectory']
        
    def generate_initial_conditions(self) -> np.ndarray:
        """Generate initial conditions from configuration bounds.
        
        Returns:
            Array of initial conditions, shape (num_samples, state_dim)
        """
        ic_config = self.config['initial_conditions']
        bounds = ic_config['bounds']
        num_samples = ic_config['num_samples']
        sampling_method = ic_config.get('sampling_method', 'random')
        
        # Get state names and bounds
        state_names = self.system.get_state_names()
        state_dim = self.system.get_state_dimension()
        
        # Validate bounds
        if not all(name in bounds for name in state_names):
            missing = [name for name in state_names if name not in bounds]
            raise ValueError(f"Missing bounds for state variables: {missing}")
        
        # Extract bounds arrays
        lower_bounds = np.array([bounds[name][0] for name in state_names])
        upper_bounds = np.array([bounds[name][1] for name in state_names])
        
        # Generate initial conditions based on sampling method
        if sampling_method == 'random':
            np.random.seed(self.config.get('dataset.random_seed', 42))
            initial_conditions = np.random.uniform(
                lower_bounds, upper_bounds, (num_samples, state_dim)
            )
        elif sampling_method == 'grid':
            # Create grid sampling
            n_per_dim = int(np.ceil(num_samples ** (1/state_dim)))
            grids = [np.linspace(lower_bounds[i], upper_bounds[i], n_per_dim) 
                    for i in range(state_dim)]
            grid_points = np.meshgrid(*grids)
            initial_conditions = np.column_stack([g.ravel() for g in grid_points])
            # Trim to exact number requested
            initial_conditions = initial_conditions[:num_samples]
        elif sampling_method == 'sobol':
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=state_dim, seed=self.config.get('dataset.random_seed', 42))
                unit_samples = sampler.random(num_samples)
                initial_conditions = qmc.scale(unit_samples, lower_bounds, upper_bounds)
            except ImportError:
                print("Warning: scipy.stats.qmc not available, using random sampling")
                np.random.seed(self.config.get('dataset.random_seed', 42))
                initial_conditions = np.random.uniform(
                    lower_bounds, upper_bounds, (num_samples, state_dim)
                )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        return initial_conditions
    
    def simulate_trajectory(self, initial_condition: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a single trajectory.
        
        Args:
            initial_condition: Initial state vector
            
        Returns:
            Tuple of (time_points, trajectory) arrays
        """
        time_span = self.trajectory_config['time_span']
        time_points = np.linspace(time_span[0], time_span[1], 
                                 self.trajectory_config['time_points'])
        
        # Solve ODE
        solution = solve_ivp(
            fun=lambda t, y: self.system.dynamics(t, y),
            t_span=time_span,
            y0=initial_condition,
            t_eval=time_points,
            method=self.trajectory_config.get('solver', 'RK45'),
            rtol=self.trajectory_config.get('rtol', 1e-8),
            atol=self.trajectory_config.get('atol', 1e-10)
        )
        
        if not solution.success:
            raise RuntimeError(f"ODE integration failed: {solution.message}")
        
        trajectory = solution.y.T  # Shape: (time_points, state_dim)
        
        # Remove transients if specified
        remove_transient = self.trajectory_config.get('remove_transient', 0)
        if remove_transient > 0:
            trajectory = trajectory[remove_transient:]
            time_points = time_points[remove_transient:]
        
        return time_points, trajectory
    
    def simulate_all_trajectories(self) -> Dict[str, Any]:
        """Simulate all trajectories from initial conditions.
        
        Returns:
            Dictionary containing all trajectory data
        """
        print("Generating initial conditions...")
        initial_conditions = self.generate_initial_conditions()
        
        print(f"Simulating {len(initial_conditions)} trajectories...")
        trajectories = []
        time_points_list = []
        successful_ics = []
        failed_count = 0
        
        for i, ic in enumerate(tqdm(initial_conditions, desc="Simulating")):
            try:
                time_points, trajectory = self.simulate_trajectory(ic)
                trajectories.append(trajectory)
                time_points_list.append(time_points)
                successful_ics.append(ic)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Only print first few failures
                    print(f"Warning: Failed to simulate trajectory {i}: {e}")
        
        if failed_count > 0:
            print(f"Warning: {failed_count} trajectories failed to simulate")
        
        # Convert to arrays
        trajectories = np.array(trajectories)  # Shape: (num_traj, time_points, state_dim)
        successful_ics = np.array(successful_ics)
        
        # Normalize if specified
        if self.trajectory_config.get('normalize', False):
            trajectories, normalization_stats = self._normalize_trajectories(trajectories)
        else:
            normalization_stats = None
        
        # Create dataset dictionary
        dataset = {
            'trajectories': trajectories,
            'initial_conditions': successful_ics,
            'time_points': time_points_list[0] if time_points_list else None,
            'system_type': self.config['dynamics']['type'],
            'system_parameters': self.config['dynamics']['parameters'],
            'trajectory_config': self.trajectory_config,
            'normalization_stats': normalization_stats,
            'state_names': self.system.get_state_names(),
            'num_successful': len(successful_ics),
            'num_failed': failed_count
        }
        
        return dataset
    
    def _normalize_trajectories(self, trajectories: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Normalize trajectories to zero mean and unit variance.
        
        Args:
            trajectories: Array of trajectories
            
        Returns:
            Tuple of (normalized_trajectories, normalization_stats)
        """
        # Flatten trajectories for computing statistics
        flat_trajectories = trajectories.reshape(-1, trajectories.shape[-1])
        
        # Compute statistics
        mean = np.mean(flat_trajectories, axis=0)
        std = np.std(flat_trajectories, axis=0)
        
        # Avoid division by zero
        std = np.where(std > 1e-8, std, 1.0)
        
        # Normalize
        normalized_trajectories = (trajectories - mean) / std
        
        normalization_stats = {
            'mean': mean,
            'std': std
        }
        
        return normalized_trajectories, normalization_stats
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: Path):
        """Save dataset to pickle file.
        
        Args:
            dataset: Dataset dictionary
            filepath: Path to save the dataset
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {filepath}")
        print(f"Dataset contains {dataset['num_successful']} successful trajectories")
        if dataset['num_failed'] > 0:
            print(f"Warning: {dataset['num_failed']} trajectories failed")


def generate_dataset(config: Config) -> Path:
    """Generate complete dataset from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Path to the saved dataset file
    """
    # Create dynamical system
    dynamics_config = config['dynamics']
    system = create_system(dynamics_config['type'], dynamics_config)
    
    # Create simulator
    simulator = TrajectorySimulator(system, config)
    
    # Generate dataset
    dataset = simulator.simulate_all_trajectories()
    
    # Create output path
    output_dir = config.create_output_dir()
    dataset_filename = f"{config['system_name']}_dataset.pkl"
    dataset_path = output_dir / 'data' / dataset_filename
    
    # Save dataset
    simulator.save_dataset(dataset, dataset_path)
    
    return dataset_path


def load_dataset(filepath: Path) -> Dict[str, Any]:
    """Load dataset from pickle file.
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        Dataset dictionary
    """
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset 
"""Predefined dynamical systems and custom system support."""

import numpy as np
from typing import List, Callable, Dict, Any
from abc import ABC, abstractmethod


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems."""
    
    def __init__(self, parameters: Dict[str, float]):
        """Initialize the dynamical system with parameters.
        
        Args:
            parameters: Dictionary of system parameters
        """
        self.parameters = parameters
        self.state_dim = self.get_state_dimension()
    
    @abstractmethod
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the time derivative of the state.
        
        Args:
            t: Current time
            state: Current state vector
            
        Returns:
            Time derivative of the state
        """
        pass
    
    @abstractmethod
    def get_state_dimension(self) -> int:
        """Return the dimension of the state space."""
        pass
    
    @abstractmethod
    def get_state_names(self) -> List[str]:
        """Return names of state variables."""
        pass


class LorenzSystem(DynamicalSystem):
    """Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz"""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize Lorenz system.
        
        Args:
            parameters: System parameters (sigma, rho, beta)
        """
        default_params = {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute Lorenz system dynamics."""
        x, y, z = state
        sigma, rho, beta = self.parameters['sigma'], self.parameters['rho'], self.parameters['beta']
        
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        
        return np.array([dxdt, dydt, dzdt])
    
    def get_state_dimension(self) -> int:
        return 3
    
    def get_state_names(self) -> List[str]:
        return ['x', 'y', 'z']


class RosslerSystem(DynamicalSystem):
    """Rossler system: dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c)"""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize Rossler system.
        
        Args:
            parameters: System parameters (a, b, c)
        """
        default_params = {'a': 0.2, 'b': 0.2, 'c': 5.7}
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute Rossler system dynamics."""
        x, y, z = state
        a, b, c = self.parameters['a'], self.parameters['b'], self.parameters['c']
        
        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
        
        return np.array([dxdt, dydt, dzdt])
    
    def get_state_dimension(self) -> int:
        return 3
    
    def get_state_names(self) -> List[str]:
        return ['x', 'y', 'z']


class VanDerPolSystem(DynamicalSystem):
    """Van der Pol oscillator: dx/dt = y, dy/dt = μ(1-x²)y - x"""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize Van der Pol system.
        
        Args:
            parameters: System parameters (mu)
        """
        default_params = {'mu': 1.0}
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute Van der Pol system dynamics."""
        x, y = state
        mu = self.parameters['mu']
        
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        
        return np.array([dxdt, dydt])
    
    def get_state_dimension(self) -> int:
        return 2
    
    def get_state_names(self) -> List[str]:
        return ['x', 'y']


class LinearSystem(DynamicalSystem):
    """3D Linear dynamical system dy/dt = A*y where A is a 3x3 matrix."""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize linear system with parameters mu and lambda."""
        default_params = {'mu': 1.0, 'lambda': 2.0}
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
        
        # Extract parameters
        self.mu = self.parameters['mu']
        self.lam = self.parameters['lambda']  # lambda is a reserved word
        
        # Create the system matrix A
        self.A = np.array([
            [self.mu,    0,         0],
            [0,          self.lam,  -self.lam],
            [0,          0,         2*self.mu]
        ])
        
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Linear dynamics: dy/dt = A * y
        """
        return self.A @ state
    
    def get_state_dimension(self) -> int:
        return 3
    
    def get_state_names(self) -> List[str]:
        return ['y1', 'y2', 'y3']


class BruntonSystem(DynamicalSystem):
    """Brunton system: dx₁/dt = μx₁, dx₂/dt = λ(x₂ - x₁²)"""
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize Brunton system.
        
        Args:
            parameters: System parameters (mu, lambda)
        """
        default_params = {'mu': -0.05, 'lambda': -1.0}
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute Brunton system dynamics."""
        x1, x2 = state
        mu, lam = self.parameters['mu'], self.parameters['lambda']
        
        dx1dt = mu * x1
        dx2dt = lam * (x2 - x1**2)
        
        return np.array([dx1dt, dx2dt])
    
    def get_state_dimension(self) -> int:
        return 2
    
    def get_state_names(self) -> List[str]:
        return ['x1', 'x2']


class CustomSystem(DynamicalSystem):
    """Custom dynamical system defined by user equations."""
    
    def __init__(self, dynamics_func: Callable, state_names: List[str], parameters: Dict[str, float] = None):
        """Initialize custom system.
        
        Args:
            dynamics_func: Function that computes dynamics (t, state, params) -> derivative
            state_names: Names of state variables
            parameters: System parameters
        """
        self.dynamics_func = dynamics_func
        self.state_names = state_names
        parameters = parameters or {}
        super().__init__(parameters)
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute custom system dynamics."""
        return self.dynamics_func(t, state, self.parameters)
    
    def get_state_dimension(self) -> int:
        return len(self.state_names)
    
    def get_state_names(self) -> List[str]:
        return self.state_names


class NonlinearTransformedSystem(DynamicalSystem):
    """Nonlinear system constructed from linear system via differentiable injection.
    
    Based on the transformation: dx/dt = J_F(x)^(-1) * A * F(x)
    where F(x) = [f1(x1); f2(x2); ...; fn(xn)] is a component-wise transformation.
    """
    
    def __init__(self, parameters: Dict[str, float] = None):
        """Initialize nonlinear transformed system.
        
        Args:
            parameters: System parameters including:
                - 'mu': Parameter for linear system matrix A
                - 'lambda': Parameter for linear system matrix A
                - 'alpha': Nonlinearity strength for coordinate transformation
                - 'beta': Additional nonlinearity parameter
        """
        default_params = {
            'mu': -0.2,      # Parameter μ in the linear system matrix A (distinct eigenvalue)
            'lambda': -0.4,  # Parameter λ in the linear system matrix A (distinct eigenvalue)
            'alpha': 0.3,    # Nonlinearity strength for coordinate transformation (small)
            'beta': 0.5      # Additional nonlinearity parameter (small)
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(default_params)
    
    def _coordinate_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply coordinate transformation F(x) = [f1(x1); f2(x2); f3(x3)].
        
        Using interesting nonlinear functions:
        f1(x1) = x1 + α * sin(β * x1)
        f2(x2) = x2 + α * tanh(β * x2)
        f3(x3) = x3 + α * (1 - exp(-β * x3))
        """
        alpha, beta = self.parameters['alpha'], self.parameters['beta']
        
        f1 = x[0] + alpha * np.sin(beta * x[0])
        f2 = x[1] + alpha * np.tanh(beta * x[1])
        f3 = x[2] + alpha * (1 - np.exp(-beta * x[2]))
        
        return np.array([f1, f2, f3])
    
    def _jacobian_inverse(self, x: np.ndarray) -> np.ndarray:
        """Compute J_F(x)^(-1) where J_F is the Jacobian of the coordinate transformation.
        
        J_F(x) = diag(df1/dx1, df2/dx2, df3/dx3)
        J_F^(-1)(x) = diag(1/(df1/dx1), 1/(df2/dx2), 1/(df3/dx3))
        """
        alpha, beta = self.parameters['alpha'], self.parameters['beta']
        
        # Derivatives of the coordinate transformation functions
        df1_dx1 = 1 + alpha * beta * np.cos(beta * x[0])
        df2_dx2 = 1 + alpha * beta * (1 - np.tanh(beta * x[1])**2)
        df3_dx3 = 1 + alpha * beta * np.exp(-beta * x[2])
        
        # Inverse Jacobian (diagonal matrix)
        return np.diag([1/df1_dx1, 1/df2_dx2, 1/df3_dx3])
    
    def _linear_system_matrix(self) -> np.ndarray:
        """Construct the linear system matrix A."""
        mu, lam = self.parameters['mu'], self.parameters['lambda']
        
        # Create a simple upper triangular matrix with distinct eigenvalues
        # This makes it easy to verify the learned A matrix
        A = np.array([
            [mu, 0.1, 0.2],
            [0.0, lam, 0.3],
            [0.0, 0.0, mu + lam]
        ])
        
        return A
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute nonlinear system dynamics: dx/dt = J_F(x)^(-1) * A * F(x)."""
        # Apply coordinate transformation
        F_x = self._coordinate_transform(state)
        
        # Get linear system matrix
        A = self._linear_system_matrix()
        
        # Compute A * F(x)
        A_F_x = A @ F_x
        
        # Get inverse Jacobian
        J_inv = self._jacobian_inverse(state)
        
        # Final dynamics: dx/dt = J_F(x)^(-1) * A * F(x)
        dx_dt = J_inv @ A_F_x
        
        return dx_dt
    
    def get_state_dimension(self) -> int:
        return 3
    
    def get_state_names(self) -> List[str]:
        return ['x1', 'x2', 'x3']


def create_system(system_type: str, config: Dict[str, Any]) -> DynamicalSystem:
    """Factory function to create dynamical systems.
    
    Args:
        system_type: Type of system ('lorenz', 'rossler', 'vanderpol', 'custom')
        config: Configuration dictionary
    
    Returns:
        DynamicalSystem instance
    """
    parameters = config.get('parameters', {})
    
    if system_type == 'lorenz':
        return LorenzSystem(parameters)
    elif system_type == 'rossler':
        return RosslerSystem(parameters)
    elif system_type == 'vanderpol':
        return VanDerPolSystem(parameters)
    elif system_type == 'linear':
        return LinearSystem(parameters)
    elif system_type == 'brunton':
        return BruntonSystem(parameters)
    elif system_type == 'nonlinear_transformed':
        return NonlinearTransformedSystem(parameters)
    elif system_type == 'custom':
        # For custom systems, execute the custom equations
        custom_code = config.get('custom_equations', '')
        if not custom_code:
            raise ValueError("Custom equations must be provided for custom systems")
        
        # Create a namespace for executing the custom code
        namespace = {'np': np, 'numpy': np}
        exec(custom_code, namespace)
        
        # Extract the dynamics function
        if 'dynamics_func' not in namespace:
            raise ValueError("Custom equations must define 'dynamics_func'")
        
        dynamics_func = namespace['dynamics_func']
        
        # Infer state names from bounds if not provided
        state_names = list(config.get('state_names', ['x', 'y', 'z']))
        
        return CustomSystem(dynamics_func, state_names, parameters)
    else:
        raise ValueError(f"Unknown system type: {system_type}")


# Predefined system configurations for easy access
PREDEFINED_SYSTEMS = {
    'lorenz': {
        'class': LorenzSystem,
        'default_params': {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0},
        'typical_bounds': {'x': [-20, 20], 'y': [-30, 30], 'z': [0, 50]}
    },
    'rossler': {
        'class': RosslerSystem,
        'default_params': {'a': 0.2, 'b': 0.2, 'c': 5.7},
        'typical_bounds': {'x': [-15, 15], 'y': [-20, 20], 'z': [0, 25]}
    },
    'vanderpol': {
        'class': VanDerPolSystem,
        'default_params': {'mu': 1.0},
        'typical_bounds': {'x': [-3, 3], 'y': [-3, 3]}
    },
    'brunton': {
        'class': BruntonSystem,
        'default_params': {'mu': -0.05, 'lambda': -1.0},
        'typical_bounds': {'x1': [-4, 4], 'x2': [-4, 4]}
    },
    'nonlinear_transformed': {
        'class': NonlinearTransformedSystem,
        'default_params': {'mu': -0.2, 'lambda': -0.4, 'alpha': 0.3, 'beta': 0.5},
        'typical_bounds': {'x1': [-2, 2], 'x2': [-2, 2], 'x3': [-2, 2]}
    }
} 
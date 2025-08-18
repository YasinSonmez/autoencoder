"""State-space autoencoder for finding manifold coordinates."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np


class AlmostIdentityLinear(nn.Module):
    """Linear map y = A x with A equal to identity except one learnable entry.

    - If variable_index = (i, i), then A[i, i] is learnable (initialized to 1.0), others are fixed to identity.
    - If variable_index = (i, j) with i != j, then A = I + a * E_{i,j} so only off-diagonal (i,j) is learnable.
    """

    def __init__(
        self,
        latent_dim: int,
        variable_index: tuple[int, int] = (0, 0),
        init_value: float = 1.0,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.row_index, self.col_index = variable_index
        # Single scalar parameter controlling the variable entry
        self.a = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))
        self.a.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Efficient application without constructing full matrix each time
        if self.row_index == self.col_index:
            # Diagonal element variable: y = I x with y[:,i] = a * x[:,i]
            y = x.clone()
            y[:, self.row_index] = self.a * x[:, self.row_index]
            return y
        else:
            # Off-diagonal element variable: y = x + a * e_i e_j^T x => y_i += a * x_j
            y = x.clone()
            y[:, self.row_index] = x[:, self.row_index] + self.a * x[:, self.col_index]
            return y

    @torch.no_grad()
    def get_matrix(self) -> torch.Tensor:
        A = torch.eye(self.latent_dim, dtype=torch.float32)
        if self.row_index == self.col_index:
            A[self.row_index, self.col_index] = self.a.detach()
        else:
            A[self.row_index, self.col_index] = self.a.detach()
        return A


class MonotonicNetwork(nn.Module):
    """Neural network with monotonic dynamics enforced through non-negative weights.
    
    This network ensures monotonicity by using bounded activation functions (tanh) and
    enforcing non-negative weights through reparameterization techniques.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_layers: list[int] = [32, 32],
        weight_reparam: str = "squared",
        requires_grad: bool = True
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.weight_reparam = weight_reparam
        self.requires_grad = requires_grad
        
        # Validate weight reparameterization method
        valid_reparams = ["squared", "exponential", "softplus", "absolute"]
        if weight_reparam not in valid_reparams:
            raise ValueError(f"Invalid weight_reparam: {weight_reparam}. Must be one of {valid_reparams}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        prev_dim = latent_dim
        
        for hidden_dim in hidden_layers:
            # Create layer with reparameterized weights
            layer = self._create_reparametrized_layer(prev_dim, hidden_dim)
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Final layer back to latent space
        final_layer = self._create_reparametrized_layer(prev_dim, latent_dim)
        self.layers.append(final_layer)
    
    def _create_reparametrized_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a layer with reparameterized weights for non-negativity."""
        if self.weight_reparam == "squared":
            return SquaredWeightLayer(in_dim, out_dim, requires_grad=self.requires_grad)
        elif self.weight_reparam == "exponential":
            return ExponentialWeightLayer(in_dim, out_dim, requires_grad=self.requires_grad)
        elif self.weight_reparam == "softplus":
            return SoftplusWeightLayer(in_dim, out_dim, requires_grad=self.requires_grad)
        elif self.weight_reparam == "absolute":
            return AbsoluteWeightLayer(in_dim, out_dim, requires_grad=self.requires_grad)
        else:
            raise ValueError(f"Unknown weight reparameterization: {self.weight_reparam}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the monotonic network."""
        current = x
        for i, layer in enumerate(self.layers):
            current = layer(current)
            # Apply tanh activation between layers (except after the last layer)
            if i < len(self.layers) - 1:
                current = torch.tanh(current)
        return current
    
    def __len__(self) -> int:
        """Return the number of layers in the network."""
        return len(self.layers)


class SquaredWeightLayer(nn.Module):
    """Linear layer with weights reparameterized as w = x² to ensure non-negativity."""
    
    def __init__(self, in_dim: int, out_dim: int, requires_grad: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize raw parameters so that after squaring, we get reasonable initial weights
        # We want initial weights around 0.1, so initialize raw_weight around sqrt(0.1) ≈ 0.316
        self.raw_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.316 + 0.316, requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Square the raw weights to ensure non-negativity
        weight = self.raw_weight ** 2
        return torch.nn.functional.linear(x, weight, self.bias)


class ExponentialWeightLayer(nn.Module):
    """Linear layer with weights reparameterized as w = exp(x) to ensure strict positivity."""
    
    def __init__(self, in_dim: int, out_dim: int, requires_grad: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize raw parameters so that after exp, we get reasonable initial weights
        # We want initial weights around 0.1, so initialize raw_weight around log(0.1) ≈ -2.3
        self.raw_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1 - 2.3, requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exponentiate the raw weights to ensure strict positivity
        weight = torch.exp(self.raw_weight)
        return torch.nn.functional.linear(x, weight, self.bias)


class SoftplusWeightLayer(nn.Module):
    """Linear layer with weights reparameterized as w = softplus(x) to ensure non-negativity."""
    
    def __init__(self, in_dim: int, out_dim: int, requires_grad: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize raw parameters so that after softplus, we get reasonable initial weights
        # We want initial weights around 0.1, so initialize raw_weight around log(exp(0.1) - 1) ≈ -2.3
        self.raw_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1 - 2.3, requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply softplus to the raw weights to ensure non-negativity
        weight = torch.nn.functional.softplus(self.raw_weight)
        return torch.nn.functional.linear(x, weight, self.bias)


class AbsoluteWeightLayer(nn.Module):
    """Linear layer with weights reparameterized as w = |x| to ensure non-negativity."""
    
    def __init__(self, in_dim: int, out_dim: int, requires_grad: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize raw parameters so that after abs, we get reasonable initial weights
        # We want initial weights around 0.1, so initialize raw_weight around ±0.1
        self.raw_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1, requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply absolute value to the raw weights to ensure non-negativity
        weight = torch.abs(self.raw_weight)
        return torch.nn.functional.linear(x, weight, self.bias)


class StateAutoencoder(nn.Module):
    """Autoencoder for individual state points (not trajectories)."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int, mode: str = "reconstruction"):
        """Initialize state autoencoder.
        
        Args:
            config: Autoencoder configuration
            state_dim: Dimension of state space (e.g., 3 for Lorenz)
            mode: "reconstruction" or "prediction"
        """
        super().__init__()
        
        self.config = config
        self.state_dim = state_dim
        self.latent_dim = config['latent_dim']
        self.mode = mode
        
        # Create encoder
        encoder_config = config['encoder']
        encoder_layers = []
        prev_dim = state_dim
        
        for hidden_dim in encoder_config['hidden_layers']:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if encoder_config.get('batch_norm', False):
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            encoder_layers.append(self._get_activation(encoder_config.get('activation', 'relu')))
            
            if encoder_config.get('dropout', 0) > 0:
                encoder_layers.append(nn.Dropout(encoder_config['dropout']))
            
            prev_dim = hidden_dim
        
        # Final layer to latent space
        encoder_layers.append(nn.Linear(prev_dim, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder
        decoder_config = config['decoder']
        decoder_layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in decoder_config['hidden_layers']:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if decoder_config.get('batch_norm', False):
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            decoder_layers.append(self._get_activation(decoder_config.get('activation', 'relu')))
            
            if decoder_config.get('dropout', 0) > 0:
                decoder_layers.append(nn.Dropout(decoder_config['dropout']))
            
            prev_dim = hidden_dim
        
        # Final layer back to state space
        decoder_layers.append(nn.Linear(prev_dim, state_dim))
        
        # Add output activation if specified
        output_activation = decoder_config.get('output_activation', 'linear')
        if output_activation != 'linear':
            decoder_layers.append(self._get_activation(output_activation))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Create latent dynamics network for prediction mode
        if self.mode == "prediction":
            dynamics_config = config.get('latent_dynamics', {})
            
            # Check for mutual exclusivity of dynamics types
            dynamics_types = []
            if dynamics_config.get('linear', False):
                dynamics_types.append('linear')
            if dynamics_config.get('monotonic', False):
                dynamics_types.append('monotonic')
            if not dynamics_config.get('linear', False) and not dynamics_config.get('monotonic', False):
                dynamics_types.append('nonlinear')
            
            if len(dynamics_types) > 1:
                raise ValueError(f"Only one dynamics type can be enabled. Found: {dynamics_types}")
            
            # Check if linear latent dynamics is enabled
            if dynamics_config.get('linear', False):
                # Option: keep A as identity but allow a single diagonal (e.g., 3,3) entry to be learnable
                allow_single_diag = dynamics_config.get('allow_single_diagonal_variable', False)
                if allow_single_diag:
                    # 1-based index in config for user friendliness; convert to 0-based
                    diag_index = int(dynamics_config.get('variable_diagonal_index', 3)) - 1
                    diag_index = max(0, min(diag_index, self.latent_dim - 1))
                    self.latent_dynamics = AlmostIdentityLinear(self.latent_dim, variable_index=(diag_index, diag_index))
                    self.linear_dynamics = True
                else:
                    # Linear latent dynamics: z_k+1 = A * z_k using a standard Linear layer
                    self.latent_dynamics = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
                    # Initialize with identity matrix and optionally freeze updates
                    with torch.no_grad():
                        self.latent_dynamics.weight.copy_(torch.eye(self.latent_dim))
                        # self.latent_dynamics.weight.copy_(torch.zeros(self.latent_dim, self.latent_dim))
                    # Default behavior: allow A to be updated during training
                    freeze_A = dynamics_config.get('freeze_A', False)
                    if freeze_A:
                        self.latent_dynamics.weight.requires_grad = False
                    self.linear_dynamics = True
            
            # Check if monotonic latent dynamics is enabled
            elif dynamics_config.get('monotonic', False):
                # Get monotonic network configuration
                hidden_layers = dynamics_config.get('monotonic_hidden_layers', [32, 32])
                weight_reparam = dynamics_config.get('weight_reparam', 'squared')
                
                self.latent_dynamics = MonotonicNetwork(
                    latent_dim=self.latent_dim,
                    hidden_layers=hidden_layers,
                    weight_reparam=weight_reparam,
                    requires_grad=dynamics_config.get('requires_grad', True)
                )
                self.linear_dynamics = False
            
            else:
                # Nonlinear latent dynamics (original implementation)
                dynamics_layers = []
                prev_dim = self.latent_dim
                
                hidden_layers = dynamics_config.get('hidden_layers', [64, 64])
                for hidden_dim in hidden_layers:
                    dynamics_layers.append(nn.Linear(prev_dim, hidden_dim))
                    dynamics_layers.append(self._get_activation(dynamics_config.get('activation', 'tanh')))
                    
                    if dynamics_config.get('dropout', 0) > 0:
                        dynamics_layers.append(nn.Dropout(dynamics_config['dropout']))
                    
                    prev_dim = hidden_dim
                
                # Final layer back to latent space
                dynamics_layers.append(nn.Linear(prev_dim, self.latent_dim))
                self.latent_dynamics = nn.Sequential(*dynamics_layers)
                self.linear_dynamics = False
        else:
            self.latent_dynamics = None
            self.linear_dynamics = False
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'linear': nn.Identity()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.
        
        Args:
            x: Input state points, shape (batch_size, state_dim)
            
        Returns:
            Tuple of (latent_representation, reconstruction)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode state points to latent space.
        
        Args:
            x: Input state points, shape (batch_size, state_dim)
            
        Returns:
            Latent representations, shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to state space.
        
        Args:
            z: Latent representations, shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed state points, shape (batch_size, state_dim)
        """
        return self.decoder(z)
    
    def predict_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Predict next latent state (only available in prediction mode).
        
        Args:
            z: Current latent representations, shape (batch_size, latent_dim)
            
        Returns:
            Predicted next latent state, shape (batch_size, latent_dim)
        """
        if self.mode != "prediction" or self.latent_dynamics is None:
            raise ValueError("Latent prediction only available in prediction mode")
        return self.latent_dynamics(z)
    
    def forward_prediction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for prediction mode: x_k -> z_k -> z_k+1 -> x_k+1.
        
        Args:
            x: Input state points, shape (batch_size, state_dim)
            
        Returns:
            Tuple of (z_k, x_k_reconstructed, z_k+1, x_k+1_predicted)
        """
        if self.mode != "prediction":
            raise ValueError("forward_prediction only available in prediction mode")
        
        # Encode current state
        z_k = self.encoder(x)
        
        # Reconstruct current state
        x_k_reconstructed = self.decoder(z_k)
        
        # Predict next latent state
        z_k_plus_1 = self.latent_dynamics(z_k)
        
        # Decode predicted latent state
        x_k_plus_1_predicted = self.decoder(z_k_plus_1)
        
        return z_k, x_k_reconstructed, z_k_plus_1, x_k_plus_1_predicted
    
    def forward_multistep_prediction(self, x: torch.Tensor, k_steps: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass for multi-step prediction mode: x_k -> z_k -> z_k+1, z_k+2, ..., z_k+K -> x_k+1, x_k+2, ..., x_k+K.
        
        Args:
            x: Input state points, shape (batch_size, state_dim)
            k_steps: Number of future steps to predict
            
        Returns:
            Tuple of (z_k, x_k_reconstructed, [z_k+1, z_k+2, ...], [x_k+1_predicted, x_k+2_predicted, ...])
        """
        if self.mode != "prediction":
            raise ValueError("forward_multistep_prediction only available in prediction mode")
        
        # Encode current state
        z_k = self.encoder(x)
        
        # Reconstruct current state
        x_k_reconstructed = self.decoder(z_k)
        
        # Multi-step prediction
        predicted_latents = []
        predicted_states = []
        
        z_current = z_k
        for step in range(k_steps):
            # Predict next latent state
            z_next = self.latent_dynamics(z_current)
            predicted_latents.append(z_next)
            
            # Decode predicted latent state
            x_next = self.decoder(z_next)
            predicted_states.append(x_next)
            
            # Update for next iteration
            z_current = z_next
        
        return z_k, x_k_reconstructed, predicted_latents, predicted_states
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'mode': self.mode,
            'state_dim': self.state_dim,
            'latent_dim': self.latent_dim,
            'num_parameters': self.get_num_parameters(),
            'encoder_layers': len(self.encoder),
            'decoder_layers': len(self.decoder)
        }
        
        if self.mode == "prediction" and self.latent_dynamics is not None:
            if hasattr(self, 'linear_dynamics') and self.linear_dynamics:
                info['latent_dynamics_type'] = 'linear'
                info['latent_dynamics_layers'] = 1
                info['linear_dynamics'] = True  # Add this flag
                # Get the A matrix
                with torch.no_grad():
                    if isinstance(self.latent_dynamics, AlmostIdentityLinear):
                        A_matrix = self.latent_dynamics.get_matrix().cpu().numpy()
                    elif isinstance(self.latent_dynamics, nn.Linear):
                        A_matrix = self.latent_dynamics.weight.data.cpu().numpy()
                    else:
                        A_matrix = None
                    info['A_matrix'] = A_matrix
            elif isinstance(self.latent_dynamics, MonotonicNetwork):
                info['latent_dynamics_type'] = 'monotonic'
                info['latent_dynamics_layers'] = len(self.latent_dynamics.layers)
                info['linear_dynamics'] = False  # Add this flag
                info['monotonic_config'] = {
                    'hidden_layers': self.latent_dynamics.hidden_layers,
                    'weight_reparam': self.latent_dynamics.weight_reparam
                }
            else:
                info['latent_dynamics_type'] = 'nonlinear'
                info['latent_dynamics_layers'] = len(self.latent_dynamics)
                info['linear_dynamics'] = False  # Add this flag
        
        return info


def prepare_state_data(trajectories: np.ndarray) -> torch.Tensor:
    """Prepare state data for autoencoder training.
    
    Args:
        trajectories: Array of trajectories, shape (num_traj, time_points, state_dim)
        
    Returns:
        Flattened state tensor, shape (num_traj * time_points, state_dim)
    """
    num_traj, time_points, state_dim = trajectories.shape
    # Flatten all state points into one big dataset
    states = trajectories.reshape(-1, state_dim)
    return torch.FloatTensor(states)


def prepare_sequential_data(trajectories: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare sequential data for prediction training.
    
    Args:
        trajectories: Array of trajectories, shape (num_traj, time_points, state_dim)
        
    Returns:
        Tuple of (x_k, x_k_plus_1) tensors for prediction training
    """
    num_traj, time_points, state_dim = trajectories.shape
    
    # Create sequential pairs from all trajectories
    x_k_list = []
    x_k_plus_1_list = []
    
    for traj in trajectories:
        # For each trajectory, create pairs (x_k, x_k+1)
        for t in range(time_points - 1):
            x_k_list.append(traj[t])
            x_k_plus_1_list.append(traj[t + 1])
    
    x_k = torch.FloatTensor(np.array(x_k_list))
    x_k_plus_1 = torch.FloatTensor(np.array(x_k_plus_1_list))
    
    return x_k, x_k_plus_1


def prepare_multistep_sequential_data(trajectories: np.ndarray, k_steps: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Prepare multi-step sequential data for prediction training.
    
    Args:
        trajectories: Array of trajectories, shape (num_traj, time_points, state_dim)
        k_steps: Number of future steps to predict
        
    Returns:
        Tuple of (x_k, [x_k+1, x_k+2, ..., x_k+k]) tensors for multi-step prediction training
    """
    num_traj, time_points, state_dim = trajectories.shape
    
    # Create multi-step sequences from all trajectories
    x_k_list = []
    x_future_lists = [[] for _ in range(k_steps)]  # One list for each future step
    
    for traj in trajectories:
        # For each trajectory, create sequences (x_k, x_k+1, x_k+2, ..., x_k+k)
        for t in range(time_points - k_steps):
            x_k_list.append(traj[t])
            
            # Collect all future steps
            for step in range(k_steps):
                x_future_lists[step].append(traj[t + step + 1])
    
    # Convert to tensors
    x_k = torch.FloatTensor(np.array(x_k_list))
    x_future = [torch.FloatTensor(np.array(future_list)) for future_list in x_future_lists]
    
    return x_k, x_future


def create_state_autoencoder(config: Dict[str, Any], state_dim: int, mode: str = "reconstruction") -> StateAutoencoder:
    """Factory function to create state autoencoder.
    
    Args:
        config: Autoencoder configuration
        state_dim: Dimension of state space
        mode: "reconstruction" or "prediction"
        
    Returns:
        StateAutoencoder model
    """
    return StateAutoencoder(config, state_dim, mode)


class MLPDynamicsModel(nn.Module):
    """MLP-based dynamics model that operates directly in state space."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int):
        """Initialize MLP dynamics model.
        
        Args:
            config: Model configuration
            state_dim: Dimension of state space
        """
        super().__init__()
        
        self.config = config
        self.state_dim = state_dim
        
        # Create MLP dynamics network
        dynamics_config = config.get('mlp_dynamics', {})
        dynamics_layers = []
        prev_dim = state_dim
        
        hidden_layers = dynamics_config.get('hidden_layers', [64, 64])
        for hidden_dim in hidden_layers:
            dynamics_layers.append(nn.Linear(prev_dim, hidden_dim))
            dynamics_layers.append(self._get_activation(dynamics_config.get('activation', 'tanh')))
            
            if dynamics_config.get('dropout', 0) > 0:
                dynamics_layers.append(nn.Dropout(dynamics_config['dropout']))
            
            prev_dim = hidden_dim
        
        # Final layer back to state space
        dynamics_layers.append(nn.Linear(prev_dim, state_dim))
        self.mlp_dynamics = nn.Sequential(*dynamics_layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'linear': nn.Identity()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next state directly: x_k -> x_k+1.
        
        Args:
            x: Current state, shape (batch_size, state_dim)
            
        Returns:
            Predicted next state, shape (batch_size, state_dim)
        """
        return self.mlp_dynamics(x)
    
    def forward_multistep_prediction(self, x: torch.Tensor, k_steps: int) -> List[torch.Tensor]:
        """Forward pass for multi-step prediction: x_k -> x_k+1, x_k+2, ..., x_k+K.
        
        Args:
            x: Input state points, shape (batch_size, state_dim)
            k_steps: Number of future steps to predict
            
        Returns:
            List of predicted states [x_k+1, x_k+2, ..., x_k+K]
        """
        predicted_states = []
        x_current = x
        
        for step in range(k_steps):
            # Predict next state
            x_next = self.mlp_dynamics(x_current)
            predicted_states.append(x_next)
            
            # Update for next iteration
            x_current = x_next
        
        return predicted_states
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'mlp_dynamics',
            'state_dim': self.state_dim,
            'num_parameters': self.get_num_parameters(),
            'dynamics_layers': len(self.mlp_dynamics)
        }


def create_mlp_dynamics_model(config: Dict[str, Any], state_dim: int) -> MLPDynamicsModel:
    """Factory function to create MLP dynamics model.
    
    Args:
        config: Model configuration
        state_dim: Dimension of state space
        
    Returns:
        MLPDynamicsModel instance
    """
    return MLPDynamicsModel(config, state_dim) 
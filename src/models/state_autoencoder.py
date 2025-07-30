"""State-space autoencoder for finding manifold coordinates."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np


class StateAutoencoder(nn.Module):
    """Autoencoder for individual state points (not trajectories)."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int):
        """Initialize state autoencoder.
        
        Args:
            config: Autoencoder configuration
            state_dim: Dimension of state space (e.g., 3 for Lorenz)
        """
        super().__init__()
        
        self.config = config
        self.state_dim = state_dim
        self.latent_dim = config['latent_dim']
        
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
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'state_dim': self.state_dim,
            'latent_dim': self.latent_dim,
            'num_parameters': self.get_num_parameters(),
            'encoder_layers': len(self.encoder),
            'decoder_layers': len(self.decoder)
        }


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


def create_state_autoencoder(config: Dict[str, Any], state_dim: int) -> StateAutoencoder:
    """Factory function to create state autoencoder.
    
    Args:
        config: Autoencoder configuration
        state_dim: Dimension of state space
        
    Returns:
        StateAutoencoder model
    """
    return StateAutoencoder(config, state_dim) 
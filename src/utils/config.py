"""Configuration management utilities."""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager for the dynamics autoencoder project."""
    
    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        from datetime import datetime
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
        # Generate timestamp once at initialization for consistent folder naming
        self._cached_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        required_sections = [
            'system_name', 'dynamics', 'initial_conditions', 
            'trajectory', 'dataset', 'autoencoder', 'training', 
            'evaluation', 'output'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific subsections
        self._validate_dynamics()
        self._validate_initial_conditions()
        self._validate_trajectory()
        self._validate_autoencoder()
        self._validate_training()
    
    def _validate_dynamics(self):
        """Validate dynamics configuration."""
        dynamics = self.config['dynamics']
        if 'type' not in dynamics:
            raise ValueError("Dynamics type must be specified")
        
        if dynamics['type'] not in ['lorenz', 'rossler', 'vanderpol', 'linear', 'brunton', 'nonlinear_transformed', 'custom']:
            raise ValueError("Invalid dynamics type")
        
        if dynamics['type'] == 'custom' and 'custom_equations' not in dynamics:
            raise ValueError("Custom equations must be provided for custom dynamics")
    
    def _validate_initial_conditions(self):
        """Validate initial conditions configuration."""
        ic = self.config['initial_conditions']
        if 'bounds' not in ic or 'num_samples' not in ic:
            raise ValueError("Initial conditions must specify bounds and num_samples")
    
    def _validate_trajectory(self):
        """Validate trajectory configuration."""
        traj = self.config['trajectory']
        required_fields = ['time_span', 'time_points', 'solver']
        for field in required_fields:
            if field not in traj:
                raise ValueError(f"Trajectory configuration missing: {field}")
    
    def _validate_autoencoder(self):
        """Validate autoencoder configuration."""
        ae = self.config['autoencoder']
        if 'latent_dim' not in ae:
            raise ValueError("Autoencoder latent dimension must be specified")
    
    def _validate_training(self):
        """Validate training configuration."""
        training = self.config['training']
        required_fields = ['batch_size', 'epochs', 'learning_rate', 'optimizer']
        for field in required_fields:
            if field not in training:
                raise ValueError(f"Training configuration missing: {field}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with optional default."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def create_output_dir(self) -> Path:
        """Create output directory structure with timestamp to avoid overwriting."""
        from datetime import datetime
        
        base_dir = Path(self.get('output.base_dir', 'results'))
        exp_name = self.get('output.experiment_name', 'experiment')
        
        # Use the cached timestamp generated at initialization
        
        mode = self.get('mode', 'reconstruction')
        timestamped_name = f"{exp_name}_{mode}_{self._cached_timestamp}"
        
        output_dir = base_dir / exp_name / timestamped_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'models').mkdir(exist_ok=True)
        (output_dir / 'plots').mkdir(exist_ok=True)
        (output_dir / 'data').mkdir(exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
        (output_dir / 'data_inspection').mkdir(exist_ok=True)
        
        return output_dir
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting."""
        self.set(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return self.config 
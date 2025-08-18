"""Unified training pipeline for state-space models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, List
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ..models.state_autoencoder import create_state_autoencoder, create_mlp_dynamics_model
from ..utils.config import Config


class BaseStateTrainer:
    """Unified trainer that treats all training as sweeps - single or multiple models."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], device: str = None):
        self.config = config
        self.dataset = dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = dataset['trajectories'].shape[2]
        
        # Get model configuration
        self.models_config = config.get('models', {})
        self.enabled_models = [name for name, cfg in self.models_config.items() if cfg.get('enabled', True)]
        
        # Prepare data once for all models
        self._prepare_data()
        self._setup_training()
        
        print(f"Training {len(self.enabled_models)} model(s): {', '.join(self.enabled_models)}")
        if hasattr(self, 'k_steps') and self.k_steps > 1:
            decay_factor = self.config['training'].get('losses', {}).get('decay_factor', 0.8)
            print(f"Multi-step prediction: k={self.k_steps} steps, decay_factor={decay_factor}")
        print(f"Device: {self.device}")
    
    def _prepare_data(self):
        """Prepare data for training - same for all models."""
        trajectories = self.dataset['trajectories']
        num_traj = len(trajectories)
        
        # Split trajectories
        np.random.seed(42)
        indices = np.random.permutation(num_traj)
        train_size, val_size = int(0.8 * num_traj), int(0.1 * num_traj)
        train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
        
        train_traj, val_traj, test_traj = trajectories[train_idx], trajectories[val_idx], trajectories[test_idx]
        self.test_trajectories = test_traj
        
        # Always prepare multi-step data for prediction training
        k_steps = self.config['training'].get('losses', {}).get('S_p', 1)
        self.k_steps = k_steps
        
        # Prepare multi-step data
        multi_data = [self._prepare_multistep_sequential_data(traj, k_steps) for traj in [train_traj, val_traj, test_traj]]
        
        # Filter out None results and create datasets
        valid_datasets = []
        for i, (x_k, x_future) in enumerate(multi_data):
            if x_k is not None and x_future is not None:
                valid_datasets.append(TensorDataset(x_k, *x_future))
        
        # Ensure we have exactly 3 datasets
        if len(valid_datasets) != 3:
            raise ValueError(f"Expected 3 datasets (train, val, test), but got {len(valid_datasets)}. "
                           f"This may happen if trajectories are too short for k_steps={k_steps}. "
                           f"Consider reducing k_steps or increasing trajectory length.")
        
        self.train_dataset, self.val_dataset, self.test_dataset = valid_datasets
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Data splits - Train: {len(train_traj)}, Val: {len(val_traj)}, Test: {len(test_traj)}")
    
    def _prepare_multistep_sequential_data(self, trajectories, k_steps):
        """Prepare multi-step data (x_k, x_k+1, ..., x_k+k) for prediction."""
        # Ensure we have enough time steps for k_steps
        if trajectories.shape[1] <= k_steps:
            return None, None
            
        # Calculate the maximum time steps we can use
        max_time_steps = trajectories.shape[1] - k_steps
        
        # x_k: all trajectories, first max_time_steps time points
        x_k = torch.FloatTensor(trajectories[:, :max_time_steps].reshape(-1, trajectories.shape[-1]))
        
        # x_future: for each step, get the corresponding future time step from all trajectories
        x_future = []
        for i in range(k_steps):
            # Get future time step i+1 from all trajectories
            future_traj = trajectories[:, i+1:i+1+max_time_steps]
            # Reshape to match x_k exactly: [num_trajectories * max_time_steps, state_dim]
            future_tensor = torch.FloatTensor(future_traj.reshape(-1, trajectories.shape[-1]))
            # Ensure the future tensor has the same number of samples as x_k
            if future_tensor.shape[0] != x_k.shape[0]:
                print(f"    WARNING: Shape mismatch - x_k: {x_k.shape}, x_future[{i}]: {future_tensor.shape}")
                print(f"    future_traj shape: {future_traj.shape}, max_time_steps: {max_time_steps}")
                continue
            x_future.append(future_tensor)
        
        # Debug: print data shapes and sample values
        print(f"    trajectories shape: {trajectories.shape}, k_steps: {k_steps}")
        print(f"    max_time_steps: {max_time_steps}")
        print(f"    x_k shape: {x_k.shape}")
        print(f"    x_future shapes: {[f.shape for f in x_future]}")
        
        # Debug: print sample values to see if they make sense
        if len(x_future) > 0:
            print(f"    Sample x_k[0]: {x_k[0]}")
            print(f"    Sample x_future[0][0]: {x_future[0][0]}")
            print(f"    Sample x_future[1][0]: {x_future[1][0] if len(x_future) > 1 else 'N/A'}")
        
        # Only return data if we have all k_steps future trajectories
        if len(x_future) == k_steps:
            return (x_k, x_future)
        else:
            return None, None
    
    def _create_model(self, model_name: str, model_config: Dict[str, Any]):
        """Create a specific model based on configuration."""
        if model_name == 'state_mlp':
            # State-space MLP model
            return create_mlp_dynamics_model(self.config.to_dict(), self.state_dim).to(self.device)
        else:
            # Autoencoder-based models
            autoencoder_config = self.config['autoencoder'].copy()
            
            # Set latent dimension if specified
            if 'latent_dims' in model_config:
                # For sweeps, we'll handle this in the training loop
                pass
            
            # Configure latent dynamics type
            if model_name == 'latent_linear':
                autoencoder_config['latent_dynamics']['linear'] = True
                autoencoder_config['latent_dynamics']['monotonic'] = False
            elif model_name == 'latent_monotonic':
                autoencoder_config['latent_dynamics']['linear'] = False
                autoencoder_config['latent_dynamics']['monotonic'] = True
            else:  # latent_nn
                autoencoder_config['latent_dynamics']['linear'] = False
                autoencoder_config['latent_dynamics']['monotonic'] = False
            
            return create_state_autoencoder(autoencoder_config, self.state_dim, 'prediction').to(self.device)
    
    def _setup_training(self):
        """Setup training parameters."""
        self.lr = self.config['training']['learning_rate']
        self.criterion = nn.MSELoss()
        self.patience = self.config['training'].get('early_stopping', {}).get('patience')
    
    def _compute_loss(self, model, batch_data, return_components=False):
        """Compute loss for a given model."""
        # Use Brunton-style composite losses
        losses = self._compute_brunton_style_losses(model, batch_data)
        total = losses['total']
        if return_components:
            return total, losses
        return total

    def _compute_brunton_style_losses(self, model, batch_data):
        """Compute Brunton-style composite losses with configurable terms."""
        cfg_losses = self.config.get('training.losses', {}) or {}
        
        # Loss term switches
        use_recon = cfg_losses.get('use_reconstruction', True) and hasattr(model, 'encode')
        use_pred = cfg_losses.get('use_prediction', True)
        use_lin = cfg_losses.get('use_latent_linearity', True) and hasattr(model, 'encode')
        use_linf = cfg_losses.get('use_l_inf', True)
        use_w2 = cfg_losses.get('use_weight_decay_regularizer', False)

        # Loss weights
        alpha1 = float(cfg_losses.get('alpha1', 0.1))
        alpha2 = float(cfg_losses.get('alpha2', 1e-7))
        alpha3 = float(cfg_losses.get('alpha3', 1e-15))

        # Prediction steps
        effective_steps = min(self.k_steps, int(cfg_losses.get('S_p', self.k_steps)))

        # Unpack batch (always multi-step format)
        x_k = batch_data[0].to(self.device)
        x_futures = [batch_data[i+1].to(self.device) for i in range(self.k_steps)]

        # Reconstruction loss (only for autoencoder)
        L_recon = torch.tensor(0.0, device=self.device)
        if use_recon:
            z_k = model.encode(x_k)
            x_k_recon = model.decode(z_k)
            L_recon = self.criterion(x_k_recon, x_k)

        # Prediction and latent linearity losses
        L_pred_terms = []
        L_lin_terms = []

        # Multi-step rollout
        if hasattr(model, 'encode'):
            # Autoencoder: latent space prediction
            z_roll = model.encode(x_k)
            pred_states = []
            for m in range(effective_steps):
                z_roll = model.predict_latent(z_roll)
                x_m_pred = model.decode(z_roll)
                pred_states.append(x_m_pred)
                if m < len(x_futures):
                    if use_pred:
                        pred_loss = self.criterion(x_m_pred, x_futures[m])
                        L_pred_terms.append(pred_loss)

                    if use_lin:
                        z_true = model.encode(x_futures[m])
                        L_lin_terms.append(self.criterion(z_roll, z_true))
        else:
            # MLP: direct state-space prediction
            z_roll = x_k
            pred_states = []
            for m in range(effective_steps):
                z_roll = model(z_roll)
                pred_states.append(z_roll)
                if m < len(x_futures):
                    if use_pred:
                        pred_loss = self.criterion(z_roll, x_futures[m])
                        L_pred_terms.append(pred_loss)

        # Compute average losses
        L_pred = (sum(L_pred_terms) / len(L_pred_terms)) if L_pred_terms else torch.tensor(0.0, device=self.device)
        L_lin = (sum(L_lin_terms) / len(L_lin_terms)) if L_lin_terms else torch.tensor(0.0, device=self.device)

        # L_inf: max absolute error
        L_inf = torch.tensor(0.0, device=self.device)
        if use_linf:
            if use_recon:
                recon_err = torch.abs(x_k - model.decode(model.encode(x_k)))
                recon_inf = torch.amax(recon_err)
            else:
                recon_inf = torch.tensor(0.0, device=self.device)
            
            m_for_inf = min(int(cfg_losses.get('l_inf_m_steps', 1)), len(pred_states))
            pred_inf_total = torch.tensor(0.0, device=self.device)
            for m in range(m_for_inf):
                if m < len(x_futures):
                    pred_err = torch.abs(pred_states[m] - x_futures[m])
                    pred_inf_total = pred_inf_total + torch.amax(pred_err)
            L_inf = recon_inf + pred_inf_total

        # Weight decay regularization
        W2 = torch.tensor(0.0, device=self.device)
        if use_w2:
            targets = cfg_losses.get('weight_decay_targets', ['encoder', 'decoder'])
            for target in targets:
                if hasattr(model, target):
                    module = getattr(model, target)
                    for p in module.parameters():
                        if p.requires_grad:
                            W2 = W2 + torch.sum(p.pow(2))

        # Total loss
        total = alpha1 * (L_recon + L_pred) + L_lin + alpha2 * L_inf + alpha3 * W2
        
        return {
            'total': total,
            'reconstruction': L_recon,
            'prediction': L_pred,
            'linearity': L_lin,
            'linf': L_inf,
            'w2': W2,
        }
    
    def _train_model(self, model, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model."""
        epochs = self.config['training']['epochs']
        print(f"\nTraining {model_name}...")
        print(f"Model info: {model.get_model_info() if hasattr(model, 'get_model_info') else type(model).__name__}")
        print(f"Starting training for {epochs} epochs...")
        
        # Setup optimizer for this model
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # Training history
        training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        best_val_loss = float('inf')
        best_val_components = None
        best_test_components = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_components = {'reconstruction': 0.0, 'prediction': 0.0}
            
            for batch_data in self.train_loader:
                optimizer.zero_grad()
                loss, components = self._compute_loss(model, batch_data, return_components=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Accumulate components
                for key in train_components:
                    if key in components:
                        train_components[key] += components[key].item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_components = {key: val / len(self.train_loader) for key, val in train_components.items()}
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_components = {'reconstruction': 0.0, 'prediction': 0.0}
            
            with torch.no_grad():
                for batch_data in self.val_loader:
                    loss, components = self._compute_loss(model, batch_data, return_components=True)
                    val_loss += loss.item()
                    
                    # Accumulate components
                    for key in val_components:
                        if key in components:
                            val_components[key] += components[key].item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            avg_val_components = {key: val / len(self.val_loader) for key, val in val_components.items()}
            
            # Record history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['epochs'].append(epoch)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs}: Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
                print(f"  Train Components - Recon: {avg_train_components.get('reconstruction', 0):.6f}, Pred: {avg_train_components.get('prediction', 0):.6f}")
                print(f"  Val Components   - Recon: {avg_val_components.get('reconstruction', 0):.6f}, Pred: {avg_val_components.get('prediction', 0):.6f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_components = avg_val_components
                patience_counter = 0
                
                # Evaluate test performance at best validation epoch
                test_loss, test_components = self._evaluate_model(model)
                best_test_components = test_components
            else:
                patience_counter += 1
            
            if self.patience and patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        test_loss, test_components = self._evaluate_model(model)
        
        print(f"\nFinal Test Loss Components:")
        for key, value in test_components.items():
            print(f"  {key.capitalize()}: {value:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_loss_components': test_components,
            'best_test_loss_components': best_test_components,
            'epochs_trained': len(training_history['epochs']),
            'training_history': training_history
        }
    
    def _evaluate_model(self, model) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single model on test set."""
        model.eval()
        total_loss = 0.0
        total_components = {'reconstruction': 0.0, 'prediction': 0.0}
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                loss, components = self._compute_loss(model, batch_data, return_components=True)
                total_loss += loss.item()
        
                # Accumulate components
                for key in total_components:
                    if key in components:
                        total_components[key] += components[key].item()
        
        avg_loss = total_loss / len(self.test_loader)
        avg_components = {key: val / len(self.test_loader) for key, val in total_components.items()}
        
        return avg_loss, avg_components
    
    def train(self) -> Dict[str, Any]:
        """Main training method - trains all specified models."""
        output_dir = self.config.create_output_dir()
        models_dir = output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Train each enabled model
        for model_name in self.enabled_models:
            model_config = self.models_config[model_name]
            
            # Handle latent dimension sweeps
            if 'latent_dims' in model_config:
                latent_dims = model_config['latent_dims']
                model_results = []
                
                for d in latent_dims:
                    print(f"\nTraining {model_name} with latent_dim={d}...")
                    
                    # Create model with specific latent dimension
                    if model_name != 'state_mlp':
                        # For autoencoder models, set the latent dimension
                        temp_config = self.config.to_dict()
                        temp_config['autoencoder']['latent_dim'] = d
                        model = create_state_autoencoder(temp_config['autoencoder'], self.state_dim, 'prediction').to(self.device)
                    else:
                        model = self._create_model(model_name, model_config)
                    
                    # Train the model
                    train_results = self._train_model(model, f"{model_name} (d={d})", model_config)
                    
                    # Save model
                    model_path = models_dir / f'{model_name}_d{d}_best_model.pth'
                    torch.save(model.state_dict(), model_path)
                    
                    # Store results
                    result_entry = {
                        'latent_dim': d,
                        'prediction_loss': train_results['best_test_loss_components']['prediction'],
                        **train_results
                    }
                    model_results.append(result_entry)
                    
                    print(f"  {model_name} (d={d}): {train_results['best_test_loss_components']['prediction']:.6f}")
                
                results[f'{model_name}_sweep'] = model_results
            else:
                # Single model training
                print(f"\nTraining {model_name}...")
                model = self._create_model(model_name, model_config)
                train_results = self._train_model(model, model_name, model_config)
                
                # Save model
                model_path = models_dir / f'{model_name}_best_model.pth'
                torch.save(model.state_dict(), model_path)
                
                results[model_name] = train_results
                print(f"{model_name} prediction loss: {train_results['best_test_loss_components']['prediction']:.6f}")
        
        # Generate comparison plots if multiple models
        if len(self.enabled_models) > 1:
            self._plot_comparison_results(results, output_dir)
        
        return results
    
    def _plot_comparison_results(self, results: Dict[str, Any], output_dir: Path):
        """Generate comparison plots when multiple models are trained."""
        k_steps = self.config['training'].get('losses', {}).get('S_p', 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Overall comparison
        labels, values, colors = [], [], []
        
        # Add single models
        for model_name in self.enabled_models:
            if model_name in results:
                labels.append(model_name.replace('_', '-').title())
                values.append(results[model_name]['best_test_loss_components']['prediction'])
                colors.append(self._get_model_color(model_name))
        
        # Add sweep results
        for model_name in self.enabled_models:
            sweep_key = f'{model_name}_sweep'
            if sweep_key in results:
                for result in results[sweep_key]:
                    if model_name == 'latent_linear':
                        labels.append(f"Latent-Linear (d={result['latent_dim']})")
                    else:
                        labels.append(f"{model_name.replace('_', '-').title()} (d={result['latent_dim']})")
                    values.append(result['prediction_loss'])
                    colors.append(self._get_model_color(model_name))
        
        # Bar plot
        ax1.set_yscale('log')
        xs = np.arange(len(labels))
        ax1.bar(xs, values, color=colors, alpha=0.8)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Prediction Loss (MSE) - Log Scale')
        ax1.set_title(f'Prediction Loss Comparison (K={k_steps})')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: Dimension sweep (if applicable)
        sweep_models = [name for name in self.enabled_models if f'{name}_sweep' in results]
        if sweep_models:
            ax2.set_yscale('log')
            
            for model_name in sweep_models:
                sweep_key = f'{model_name}_sweep'
                if sweep_key in results:
                    latent_dims = [r['latent_dim'] for r in results[sweep_key]]
                    losses = [r['prediction_loss'] for r in results[sweep_key]]
                    ax2.plot(latent_dims, losses, 'o-', label=model_name.replace('_', ' ').title(), 
                             linewidth=2, markersize=8, color=self._get_model_color(model_name))
            
            # Add single model baselines
            for model_name in self.enabled_models:
                if model_name in results:
                    ax2.axhline(y=results[model_name]['best_test_loss_components']['prediction'], 
                               color=self._get_model_color(model_name), linestyle='--', 
                               linewidth=2, label=f"{model_name.replace('_', '-').title()}: {results[model_name]['best_test_loss_components']['prediction']:.6f}")
            
            ax2.set_xlabel('Latent Dimension')
            ax2.set_ylabel('Prediction Loss (MSE) - Log Scale')
            ax2.set_title('Latent Dimension Sweep')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / 'comprehensive_prediction_loss_comparison.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison plot saved to: {save_path}")

    def _get_model_color(self, model_name):
        """Get color for model type."""
        colors = {
            'state_mlp': '#4c72b0',
            'latent_nn': '#55a868',
            'latent_monotonic': '#c44e52', 
            'latent_linear': '#ff7f0e'
        }
        return colors.get(model_name, '#666666')


def create_trainer(config: Config, dataset: Dict[str, Any], device: str = None):
    """Create trainer - now always returns the unified trainer."""
    return BaseStateTrainer(config, dataset, device)
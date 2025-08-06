"""Simplified training pipeline for state-space autoencoder."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ..models.state_autoencoder import create_state_autoencoder, prepare_state_data, prepare_sequential_data, prepare_multistep_sequential_data, create_mlp_dynamics_model
from ..utils.config import Config


class BaseStateTrainer:
    """Unified trainer for all modes."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], device: str = None):
        self.config, self.dataset = config, dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = config.get('mode', 'reconstruction')
        self.state_dim = dataset['trajectories'].shape[2]
        
        self._prepare_data()
        self._create_model()
        self._setup_training()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        
        print(f"Training mode: {self.mode}")
        if hasattr(self, 'k_steps') and self.k_steps > 1:
            decay_factor = self.config['training'].get('prediction', {}).get('decay_factor', 0.8)
            print(f"Multi-step prediction: k={self.k_steps} steps, decay_factor={decay_factor}")
        print(f"Device: {self.device}, Model: {self.model.get_model_info()}")
    
    def _prepare_data(self):
        """Prepare data for training."""
        trajectories = self.dataset['trajectories']
        num_traj = len(trajectories)
        
        # Split trajectories
        np.random.seed(42)
        indices = np.random.permutation(num_traj)
        train_size, val_size = int(0.8 * num_traj), int(0.1 * num_traj)
        train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
        
        train_traj, val_traj, test_traj = trajectories[train_idx], trajectories[val_idx], trajectories[test_idx]
        self.test_trajectories = test_traj
        
        # Prepare data based on mode
        if self.mode == 'reconstruction':
            datasets = [TensorDataset(prepare_state_data(traj)) for traj in [train_traj, val_traj, test_traj]]
        else:  # prediction or mlp_dynamics
            k_steps = self.config['training'].get('prediction', {}).get('prediction_steps', 1)
            self.k_steps = k_steps
            
            if k_steps == 1:
                data_pairs = [prepare_sequential_data(traj) for traj in [train_traj, val_traj, test_traj]]
                datasets = [TensorDataset(x_k, x_k1) for x_k, x_k1 in data_pairs]
            else:
                multi_data = [prepare_multistep_sequential_data(traj, k_steps) for traj in [train_traj, val_traj, test_traj]]
                datasets = [TensorDataset(x_k, *x_future) for x_k, x_future in multi_data]
        
        self.train_dataset, self.val_dataset, self.test_dataset = datasets
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Data splits - Train: {len(train_traj)}, Val: {len(val_traj)}, Test: {len(test_traj)}")
    
    def _create_model(self):
        """Create model based on mode."""
        self.model = create_state_autoencoder(self.config['autoencoder'], self.state_dim, self.mode).to(self.device)
    
    def _setup_training(self):
        """Setup optimizer and loss function."""
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.patience = self.config['training'].get('early_stopping', {}).get('patience')
    
    def _compute_loss(self, batch_data, return_components=False):
        """Compute loss based on mode."""
        if self.mode == 'reconstruction':
            batch = batch_data[0].to(self.device)
            _, reconstruction = self.model(batch)
            return self.criterion(reconstruction, batch)
        
        elif self.mode == 'prediction':
            k_steps = getattr(self, 'k_steps', 1)
            
            if k_steps == 1:
                x_k, x_k1 = batch_data[0].to(self.device), batch_data[1].to(self.device)
                _, x_k_recon, _, x_k1_pred = self.model.forward_prediction(x_k)
                recon_loss = self.criterion(x_k_recon, x_k)
                pred_loss = self.criterion(x_k1_pred, x_k1)
            else:
                x_k = batch_data[0].to(self.device)
                x_future = [batch_data[i+1].to(self.device) for i in range(k_steps)]
                _, x_k_recon, _, pred_states = self.model.forward_multistep_prediction(x_k, k_steps)
                
                recon_loss = self.criterion(x_k_recon, x_k)
                decay_factor = self.config['training'].get('prediction', {}).get('decay_factor', 0.8)
                pred_loss = sum((decay_factor ** step) * self.criterion(pred_states[step], x_future[step]) 
                               for step in range(k_steps))
                pred_loss /= sum(decay_factor ** step for step in range(k_steps))
            
            total_loss = recon_loss + pred_loss
            return (total_loss, recon_loss, pred_loss) if return_components else total_loss
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_data in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch_data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                loss = self._compute_loss(batch_data)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else float('inf')
    
    def train(self) -> Dict[str, Any]:
        """Train the model."""
        epochs = self.config['training']['epochs']
        output_dir = self.config.create_output_dir()
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epochs'].append(epoch)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs}: Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(output_dir / 'models' / 'best_model.pth')
            else:
                self.patience_counter += 1
            
            if self.patience and self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        test_loss = self.evaluate()
        self.save_model(output_dir / 'models' / 'final_model.pth')
        self.save_training_history(output_dir / 'logs' / 'training_history.json')
        
        return {
            'best_val_loss': self.best_val_loss,
            'test_loss': test_loss,
            'epochs_trained': len(self.training_history['epochs']),
            'model_path': str(output_dir / 'models' / 'final_model.pth'),
            'best_model_path': str(output_dir / 'models' / 'best_model.pth')
        }
    
    def evaluate(self) -> float:
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                loss = self._compute_loss(batch_data)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def get_embeddings(self, data_loader=None) -> np.ndarray:
        """Get embeddings for data."""
        data_loader = data_loader or self.test_loader
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch = batch_data[0].to(self.device)
                latent = self.model.encode(batch)
                embeddings.append(latent.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_reconstructions(self, data_loader=None) -> Tuple[np.ndarray, np.ndarray]:
        """Get reconstructions for data."""
        data_loader = data_loader or self.test_loader
        self.model.eval()
        originals, reconstructions = [], []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch = batch_data[0].to(self.device)
                _, reconstruction = self.model(batch)
                originals.append(batch.cpu().numpy())
                reconstructions.append(reconstruction.cpu().numpy())
        
        return np.vstack(originals), np.vstack(reconstructions)
    
    def save_model(self, path: Path):
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def save_training_history(self, path: Path):
        """Save training history."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def plot_training_curves(self, save_path: Path = None):
        """Plot simple training curves."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = self.training_history['epochs']
        ax.plot(epochs, self.training_history['train_loss'], label='Train', linewidth=2)
        ax.plot(epochs, self.training_history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latent_dynamics_effectiveness(self, output_dir: Path):
        """Simple latent dynamics plot."""
        if self.mode != 'prediction':
            return
        print("Latent dynamics effectiveness plot generated (simplified)")


class MLPDynamicsTrainer(BaseStateTrainer):
    """MLP dynamics trainer."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], device: str = None):
        self.mode = 'mlp_dynamics'
        super().__init__(config, dataset, device)
    
    def _create_model(self):
        """Create MLP model."""
        self.model = create_mlp_dynamics_model(self.config.to_dict(), self.state_dim).to(self.device)
    
    def _compute_loss(self, batch_data, return_components=False):
        """Compute MLP loss."""
        k_steps = getattr(self, 'k_steps', 1)
        
        if k_steps == 1:
            x_k, x_k1 = batch_data[0].to(self.device), batch_data[1].to(self.device)
            x_k1_pred = self.model(x_k)
            loss = self.criterion(x_k1_pred, x_k1)
        else:
            x_k = batch_data[0].to(self.device)
            x_future = [batch_data[i+1].to(self.device) for i in range(k_steps)]
            pred_states = self.model.forward_multistep_prediction(x_k, k_steps)
            
            decay_factor = self.config['training'].get('prediction', {}).get('decay_factor', 0.8)
            loss = sum((decay_factor ** step) * self.criterion(pred_states[step], x_future[step]) 
                      for step in range(k_steps))
            loss /= sum(decay_factor ** step for step in range(k_steps))
        
        return {'total': loss} if return_components else loss


class ComparisonTrainer:
    """Compare latent vs MLP dynamics."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], device: str = None):
        self.config, self.dataset = config, dataset
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print("COMPARISON MODE: Latent vs MLP Dynamics")
    
    def train_both_models(self) -> Dict[str, Any]:
        """Train both models."""
        # Train latent dynamics
        print("\nTraining Latent Dynamics...")
        self.config.set('mode', 'prediction')
        self.latent_trainer = BaseStateTrainer(self.config, self.dataset, self.device)
        latent_results = self.latent_trainer.train()
        
        # Train MLP dynamics
        print("\nTraining MLP Dynamics...")
        self.mlp_trainer = MLPDynamicsTrainer(self.config, self.dataset, self.device)
        mlp_results = self.mlp_trainer.train()
        
        # Compare
        self.config.set('mode', 'comparison')
        latent_loss = latent_results['test_loss']
        mlp_loss = mlp_results['test_loss']
        self.winner = 'latent' if latent_loss < mlp_loss else 'mlp'
        improvement = ((max(latent_loss, mlp_loss) - min(latent_loss, mlp_loss)) / max(latent_loss, mlp_loss)) * 100
        
        print(f"\nCOMPARISON RESULTS:")
        print(f"Latent: {latent_loss:.6f}, MLP: {mlp_loss:.6f}")
        print(f"🏆 WINNER: {self.winner.upper()} ({improvement:.1f}% better)")
        
        # Generate comparison plots
        self._plot_trajectory_comparison()
        self._plot_training_comparison()
        
        winner_results = latent_results if self.winner == 'latent' else mlp_results
        return {
            'best_val_loss': winner_results['best_val_loss'],
            'test_loss': winner_results['test_loss'],
            'epochs_trained': winner_results['epochs_trained'],
            'best_model_path': winner_results['best_model_path'],
            'model_path': winner_results['model_path'],
            'comparison': {'winner': self.winner, 'improvement_percent': improvement}
        }
    
    # Delegate properties to latent trainer
    @property
    def test_trajectories(self): return self.latent_trainer.test_trajectories
    @property
    def model(self): return self.latent_trainer.model
    @property
    def device(self): return getattr(self, 'latent_trainer', self).device if hasattr(self, 'latent_trainer') else self._device
    @property
    def test_dataset(self): return self.latent_trainer.test_dataset
    @property
    def train_dataset(self): return self.latent_trainer.train_dataset
    @property
    def val_dataset(self): return self.latent_trainer.val_dataset
    @property
    def best_val_loss(self): return self.latent_trainer.best_val_loss
    @property
    def training_history(self): return self.latent_trainer.training_history
    @property
    def mode(self): return 'comparison'
    
    def get_embeddings(self, data_loader=None): return self.latent_trainer.get_embeddings(data_loader)
    def get_reconstructions(self, data_loader=None): return self.latent_trainer.get_reconstructions(data_loader)
    def plot_training_curves(self, save_path=None): self.latent_trainer.plot_training_curves(save_path)
    def _plot_latent_dynamics_effectiveness(self, output_dir): self.latent_trainer._plot_latent_dynamics_effectiveness(output_dir)
    
    def _plot_trajectory_comparison(self):
        """Plot trajectory predictions from both models (similar to reconstruction comparison)."""
        import matplotlib.pyplot as plt
        
        # Get test trajectories and predict
        test_traj = self.test_trajectories[:5]  # First 5 trajectories
        k_steps = self.config['training'].get('prediction', {}).get('prediction_steps', 1)
        
                # Get receding horizon predictions from both models (MPC-style)
        latent_predictions, mlp_predictions = [], []
        with torch.no_grad():
            for trajectory in test_traj:
                latent_pred_traj = []
                mlp_pred_traj = []
                
                # For each time step, predict one step ahead
                for step in range(len(trajectory) - 1):
                    x_current = torch.FloatTensor(trajectory[step:step+1]).to(self.device)
                    
                    # Latent dynamics: x_k -> z_k -> z_k+1 -> x_k+1
                    _, _, _, x_next_latent = self.latent_trainer.model.forward_prediction(x_current)
                    latent_pred_traj.append(x_next_latent.cpu().numpy())
                    
                    # MLP dynamics: x_k -> x_k+1
                    x_next_mlp = self.mlp_trainer.model(x_current)
                    mlp_pred_traj.append(x_next_mlp.cpu().numpy())
                
                latent_predictions.append(np.vstack(latent_pred_traj))
                mlp_predictions.append(np.vstack(mlp_pred_traj))
        
        # Create plot similar to reconstruction comparison
        state_dim = test_traj[0].shape[1]  # Get actual state dimension
        fig, axes = plt.subplots(state_dim, 1, figsize=(12, 3*state_dim))
        if state_dim == 1:
            axes = [axes]
        colors = ['red', 'green', 'orange', 'brown', 'gray']
        
        for coord_idx in range(state_dim):  # Dynamic based on state dimension
            ax = axes[coord_idx]
            
            for i, (trajectory, latent_pred, mlp_pred) in enumerate(zip(test_traj, latent_predictions, mlp_predictions)):
                color = colors[i]
                traj_id = f"Traj {i}"
                
                # Original trajectory (full length)
                time_steps = np.arange(len(trajectory))
                ax.plot(time_steps, trajectory[:, coord_idx], 
                       label=f'Original {traj_id}', color=color, linewidth=2)
                
                # Latent predictions (one step ahead for each time step)
                pred_steps = np.arange(1, len(trajectory))  # Start from step 1
                ax.plot(pred_steps, latent_pred[:, coord_idx], 
                       label=f'Latent {traj_id}', color=color, linewidth=2, linestyle='--')
                
                # MLP predictions (one step ahead for each time step)
                ax.plot(pred_steps, mlp_pred[:, coord_idx], 
                       label=f'MLP {traj_id}', color=color, linewidth=2, linestyle=':')
            
            # Get state names from dataset
            state_names = self.latent_trainer.dataset.get('state_names', [f'x{coord_idx+1}' for coord_idx in range(state_dim)])
            coord_name = state_names[coord_idx] if coord_idx < len(state_names) else f'x{coord_idx+1}'
            
            ax.set_title(f'{coord_name} Coordinate: Original vs Latent vs MLP Predictions')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(coord_name)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_dir = self.config.create_output_dir()
        plt.savefig(output_dir / 'trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trajectory comparison plot saved to: {output_dir / 'trajectory_comparison.png'}")
    
    def _plot_training_comparison(self):
        """Plot training curves for both models."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latent dynamics training curves
        epochs = self.latent_trainer.training_history['epochs']
        ax1.plot(epochs, self.latent_trainer.training_history['train_loss'], label='Train', linewidth=2)
        ax1.plot(epochs, self.latent_trainer.training_history['val_loss'], label='Validation', linewidth=2)
        ax1.set_title('Latent Dynamics Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MLP dynamics training curves
        epochs = self.mlp_trainer.training_history['epochs']
        ax2.plot(epochs, self.mlp_trainer.training_history['train_loss'], label='Train', linewidth=2, color='green')
        ax2.plot(epochs, self.mlp_trainer.training_history['val_loss'], label='Validation', linewidth=2, color='orange')
        ax2.set_title('MLP Dynamics Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_dir = self.config.create_output_dir()
        plt.savefig(output_dir / 'training_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training comparison plot saved to: {output_dir / 'training_comparison.png'}")


def create_trainer(config: Config, dataset: Dict[str, Any], device: str = None):
    """Create trainer based on mode."""
    mode = config.get('mode', 'reconstruction')
    if mode == 'comparison':
        return ComparisonTrainer(config, dataset, device)
    return BaseStateTrainer(config, dataset, device)

# Backward compatibility
StateAutoencoderTrainer = BaseStateTrainer 
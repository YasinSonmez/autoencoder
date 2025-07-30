"""Training pipeline for state-space autoencoder."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt

from ..models.state_autoencoder import create_state_autoencoder, prepare_state_data
from ..utils.config import Config


class StateAutoencoderTrainer:
    """Trainer for state-space autoencoder."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], device: str = None):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            dataset: Dataset dictionary from trajectory simulation
            device: Device to use for training
        """
        self.config = config
        self.dataset = dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training configuration
        self.training_config = config['training']
        self.autoencoder_config = config['autoencoder']
        
        # Get state dimension
        self.state_dim = dataset['trajectories'].shape[2]
        
        # Initialize model and data
        self._prepare_data()
        self._create_model()
        self._setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        print(f"Training on device: {self.device}")
        print(f"Model info: {self.model.get_model_info()}")
    
    def _prepare_data(self):
        """Prepare data for training."""
        trajectories = self.dataset['trajectories']
        num_traj, time_points, state_dim = trajectories.shape
        
        # Split trajectories (not individual points) into train/val/test
        dataset_config = self.config['dataset']
        train_split = dataset_config.get('train_split', 0.8)
        val_split = dataset_config.get('val_split', 0.1)
        test_split = dataset_config.get('test_split', 0.1)
        
        # Calculate trajectory splits
        train_traj_size = int(train_split * num_traj)
        val_traj_size = int(val_split * num_traj)
        test_traj_size = num_traj - train_traj_size - val_traj_size
        
        # Randomly shuffle trajectory indices
        np.random.seed(dataset_config.get('random_seed', 42))
        traj_indices = np.random.permutation(num_traj)
        
        # Split trajectory indices
        train_traj_idx = traj_indices[:train_traj_size]
        val_traj_idx = traj_indices[train_traj_size:train_traj_size + val_traj_size]
        test_traj_idx = traj_indices[train_traj_size + val_traj_size:]
        
        # Store trajectory indices for later use
        self.train_traj_idx = train_traj_idx
        self.val_traj_idx = val_traj_idx
        self.test_traj_idx = test_traj_idx
        
        # Extract trajectory data
        train_trajectories = trajectories[train_traj_idx]
        val_trajectories = trajectories[val_traj_idx]
        test_trajectories = trajectories[test_traj_idx]
        
        # Store test trajectories for evaluation
        self.test_trajectories = test_trajectories
        
        # Convert to state points for training
        train_data = prepare_state_data(train_trajectories)
        val_data = prepare_state_data(val_trajectories)
        test_data = prepare_state_data(test_trajectories)
        
        print(f"Trajectory splits - Train: {len(train_traj_idx)}, Val: {len(val_traj_idx)}, Test: {len(test_traj_idx)}")
        print(f"State point splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create datasets
        self.train_dataset = TensorDataset(train_data)
        self.val_dataset = TensorDataset(val_data)
        self.test_dataset = TensorDataset(test_data)
        
        # Create data loaders
        batch_size = self.training_config['batch_size']
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
    
    def _create_model(self):
        """Create state autoencoder model."""
        self.model = create_state_autoencoder(
            self.autoencoder_config, self.state_dim
        ).to(self.device)
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        optimizer_name = self.training_config['optimizer'].lower()
        lr = self.training_config['learning_rate']
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_config = self.training_config.get('scheduler')
        if scheduler_config:
            scheduler_type = scheduler_config['type']
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # Loss function
        loss_name = self.training_config.get('loss_function', 'mse').lower()
        if loss_name == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_name == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        # Early stopping
        self.early_stopping_config = self.training_config.get('early_stopping')
        if self.early_stopping_config:
            self.patience = self.early_stopping_config['patience']
            self.min_delta = self.early_stopping_config.get('min_delta', 1e-6)
        else:
            self.patience = None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in self.train_loader:
            batch = batch_data[0].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            latent, reconstruction = self.model(batch)
            loss = self.criterion(reconstruction, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                batch = batch_data[0].to(self.device)
                latent, reconstruction = self.model(batch)
                loss = self.criterion(reconstruction, batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self) -> Dict[str, Any]:
        """Train the autoencoder."""
        epochs = self.training_config['epochs']
        
        print(f"Starting training for {epochs} epochs...")
        
        # Create output directory
        output_dir = self.config.create_output_dir()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epochs'].append(epoch)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {lr:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if self.training_config.get('save_best_model', True):
                    self.save_model(output_dir / 'models' / 'best_model.pth')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience and self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience: {self.patience})")
                break
        
        # Final evaluation
        test_loss = self.evaluate()
        
        # Save final model
        final_model_path = output_dir / 'models' / 'final_model.pth'
        self.save_model(final_model_path)
        
        # Save training history
        history_path = output_dir / 'logs' / 'training_history.json'
        self.save_training_history(history_path)
        
        results = {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'test_loss': test_loss,
            'epochs_trained': len(self.training_history['epochs']),
            'model_path': str(final_model_path),
            'best_model_path': str(output_dir / 'models' / 'best_model.pth')
        }
        
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Test loss: {test_loss:.6f}")
        
        return results
    
    def evaluate(self) -> float:
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                batch = batch_data[0].to(self.device)
                latent, reconstruction = self.model(batch)
                loss = self.criterion(reconstruction, batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def get_embeddings(self, data_loader = None) -> np.ndarray:
        """Get embeddings for data."""
        if data_loader is None:
            data_loader = self.test_loader
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch = batch_data[0].to(self.device)
                latent = self.model.encode(batch)
                embeddings.append(latent.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_reconstructions(self, data_loader = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get reconstructions for data."""
        if data_loader is None:
            data_loader = self.test_loader
        
        self.model.eval()
        originals = []
        reconstructions = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch = batch_data[0].to(self.device)
                latent, reconstruction = self.model(batch)
                originals.append(batch.cpu().numpy())
                reconstructions.append(reconstruction.cpu().numpy())
        
        return np.vstack(originals), np.vstack(reconstructions)
    
    def save_model(self, path: Path):
        """Save model state dict."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def save_training_history(self, path: Path):
        """Save training history to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def plot_training_curves(self, save_path: Path = None):
        """Plot training curves with log scale."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = self.training_history['epochs']
        train_loss = self.training_history['train_loss']
        val_loss = self.training_history['val_loss']
        
        # Linear scale plot
        ax1.plot(epochs, train_loss, label='Training Loss', alpha=0.8, linewidth=2)
        ax1.plot(epochs, val_loss, label='Validation Loss', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Curves (Linear Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log scale plot
        ax2.semilogy(epochs, train_loss, label='Training Loss', alpha=0.8, linewidth=2)
        ax2.semilogy(epochs, val_loss, label='Validation Loss', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (log scale)')
        ax2.set_title('Training Curves (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add best validation loss marker
        best_epoch = val_loss.index(min(val_loss))
        best_val_loss = min(val_loss)
        
        ax1.scatter(epochs[best_epoch], best_val_loss, color='red', s=100, zorder=5)
        ax1.annotate(f'Best: {best_val_loss:.6f}', 
                    xy=(epochs[best_epoch], best_val_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax2.scatter(epochs[best_epoch], best_val_loss, color='red', s=100, zorder=5)
        ax2.annotate(f'Best: {best_val_loss:.6f}', 
                    xy=(epochs[best_epoch], best_val_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close() 
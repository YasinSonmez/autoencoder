"""Evaluation for state-space autoencoder (manifold learning)."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from typing import Dict, Any, Tuple
from pathlib import Path
import json
import torch

from ..utils.config import Config


class ModelEvaluator:
    """Simplified evaluator that works with just a model instance."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], model, output_dir: Path):
        """Initialize evaluator with just a model.
        
        Args:
            config: Configuration object
            dataset: Original dataset
            model: Trained model instance
            output_dir: Output directory for plots and results
        """
        self.config = config
        self.dataset = dataset
        self.model = model
        self.output_dir = output_dir
        
        # Get data shapes
        self.trajectory_shape = dataset['trajectories'].shape
        self.state_dim = self.trajectory_shape[2]
        self.state_names = dataset['state_names']
        
        # Create plots directory
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Run evaluation for manifold learning."""
        print("Running manifold evaluation...")
        
        results = {}
        
        # Check if this is an MLP model (no latent space)
        is_mlp_model = not hasattr(self.model, 'encode')
        
        if is_mlp_model:
            # For MLP models, focus on dynamics prediction rather than reconstruction
            print("MLP model detected - evaluating dynamics prediction capabilities")
            results['model_type'] = 'mlp_dynamics'
            results['reconstruction_error'] = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mean_correlation': 0.0, 'per_dim_mse': [], 'correlations': []}
            results['manifold_quality'] = {'distance_correlation': 0.0, 'mean_embedding': [], 'std_embedding': [], 'eigenvalues': [], 'explained_variance_ratio': [], 'effective_dimensionality': 0.0, 'embedding_range': {'min': [], 'max': []}}
            results['pca_comparison'] = {'pca_mse': 0.0, 'ae_mse': 0.0, 'improvement_over_pca': 0.0}
            
            # Generate MLP-specific plots
            self.plot_mlp_dynamics_analysis()
            
            # Add trajectory dynamics prediction plots for MLP models
            self.plot_trajectory_dynamics_prediction()
        else:
            # For autoencoder models, run full evaluation
            # Get embeddings and reconstructions
            embeddings = self.get_embeddings()
            originals, reconstructions = self.get_reconstructions()
            
            # Basic reconstruction metrics
            results['reconstruction_error'] = self.evaluate_reconstruction_error(
                originals, reconstructions
            )
            
            # Manifold quality metrics
            results['manifold_quality'] = self.evaluate_manifold_quality(
                originals, embeddings
            )
            
            # Compare with PCA
            results['pca_comparison'] = self.compare_with_pca(originals, embeddings)
            
            # Generate visualizations
            self.plot_manifold_coordinates(embeddings)
            self.plot_reconstruction_quality(originals, reconstructions)
            self.plot_trajectories_in_manifold(embeddings)
            
            # Add monotonicity analysis for monotonic dynamics
            if hasattr(self.model, 'latent_dynamics') and self.model.latent_dynamics is not None:
                if hasattr(self.model.latent_dynamics, 'weight_reparam'):
                    print(f"\nDetected monotonic dynamics with {self.model.latent_dynamics.weight_reparam} reparameterization")
                    print("Generating monotonicity analysis plots...")
                    self.plot_latent_trajectories_for_monotonicity(embeddings)
            
            self.plot_trajectory_reconstruction_comparison(originals, reconstructions)
            
            # Add trajectory dynamics prediction plots
            self.plot_trajectory_dynamics_prediction()
        
        # Save results
        self.save_evaluation_results(results)
        
        print("Evaluation completed!")
        return results
    
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings for data (autoencoder only)."""
        if not hasattr(self.model, 'encode'):
            # For MLP models, no latent space exists
            return np.array([])
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            # Get test data from the dataset
            test_traj = self.dataset['trajectories'][-50:]  # Use last 50 trajectories as test
            test_states = test_traj.reshape(-1, test_traj.shape[-1])
            test_tensor = torch.FloatTensor(test_states)
            
            # Get embeddings in batches
            batch_size = 1000
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]
                latent = self.model.encode(batch)
                embeddings.append(latent.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_reconstructions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get reconstructions for data (autoencoder only)."""
        if not hasattr(self.model, 'decode'):
            # For MLP models, no reconstruction exists
            return np.array([]), np.array([])
        
        self.model.eval()
        originals, reconstructions = [], []
        
        with torch.no_grad():
            # Get test data from the dataset
            test_traj = self.dataset['trajectories'][-50:]  # Use last 50 trajectories as test
            test_states = test_traj.reshape(-1, test_traj.shape[-1])
            test_tensor = torch.FloatTensor(test_states)
            
            # Get reconstructions in batches
            batch_size = 1000
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]
                _, reconstruction = self.model(batch)
                originals.append(batch.cpu().numpy())
                reconstructions.append(reconstruction.cpu().numpy())
        
        return np.vstack(originals), np.vstack(reconstructions)
    
    def evaluate_reconstruction_error(self, originals: np.ndarray, 
                                    reconstructions: np.ndarray) -> Dict[str, float]:
        """Evaluate reconstruction error for state points."""
        if len(originals) == 0:
            return {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mean_correlation': 0.0, 'per_dim_mse': [], 'correlations': []}
        
        # Mean squared error
        mse = np.mean((originals - reconstructions) ** 2)
        
        # Mean absolute error
        mae = np.mean(np.abs(originals - reconstructions))
        
        # Per-dimension errors
        per_dim_mse = np.mean((originals - reconstructions) ** 2, axis=0)
        
        # Correlation per dimension
        correlations = []
        for i in range(originals.shape[1]):
            corr = np.corrcoef(originals[:, i], reconstructions[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        mean_correlation = np.mean(correlations)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'mean_correlation': float(mean_correlation),
            'per_dim_mse': per_dim_mse.tolist(),
            'correlations': correlations
        }
    
    def evaluate_manifold_quality(self, originals: np.ndarray, 
                                 embeddings: np.ndarray) -> Dict[str, Any]:
        """Evaluate quality of learned manifold coordinates."""
        if len(embeddings) == 0:
            return {'distance_correlation': 0.0, 'mean_embedding': [], 'std_embedding': [], 
                   'eigenvalues': [], 'explained_variance_ratio': [], 'effective_dimensionality': 0.0, 
                   'embedding_range': {'min': [], 'max': []}}
        
        # Distance preservation
        sample_size = min(5000, len(originals))
        orig_sample = originals[:sample_size]
        embed_sample = embeddings[:sample_size]
        
        orig_distances = pairwise_distances(orig_sample)
        embed_distances = pairwise_distances(embed_sample.reshape(-1, 1) if embeddings.shape[1] == 1 else embed_sample)
        
        # Flatten upper triangular matrices
        orig_flat = orig_distances[np.triu_indices_from(orig_distances, k=1)]
        embed_flat = embed_distances[np.triu_indices_from(embed_distances, k=1)]
        
        distance_correlation = np.corrcoef(orig_flat, embed_flat)[0, 1]
        if np.isnan(distance_correlation):
            distance_correlation = 0.0
        
        # Embedding statistics
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        # Effective dimensionality
        if embeddings.shape[1] == 1:
            variance = np.var(embeddings[:, 0])
            eigenvalues = np.array([variance])
            explained_variance_ratio = np.array([1.0])
            effective_dim = 1.0
        else:
            cov_matrix = np.cov(embeddings.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]
            
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            effective_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        return {
            'distance_correlation': float(distance_correlation),
            'mean_embedding': mean_embedding.tolist(),
            'std_embedding': std_embedding.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'effective_dimensionality': float(effective_dim),
            'embedding_range': {
                'min': np.min(embeddings, axis=0).tolist(),
                'max': np.max(embeddings, axis=0).tolist()
            }
        }
    
    def compare_with_pca(self, originals: np.ndarray, 
                        embeddings: np.ndarray) -> Dict[str, Any]:
        """Compare learned manifold with PCA."""
        if len(embeddings) == 0:
            return {'pca_mse': 0.0, 'ae_mse': 0.0, 'improvement_over_pca': 0.0}
        
        # Ensure PCA components don't exceed the minimum of samples and features
        max_components = min(embeddings.shape[1], originals.shape[1], len(originals))
        if max_components < 1:
            return {'pca_mse': 0.0, 'ae_mse': 0.0, 'improvement_over_pca': 0.0}
        
        # Fit PCA to original data with appropriate number of components
        pca = PCA(n_components=max_components)
        pca_embeddings = pca.fit_transform(originals)
        
        # Reconstruct from PCA embeddings
        pca_reconstructions = pca.inverse_transform(pca_embeddings)
        
        # Calculate PCA reconstruction error
        pca_mse = np.mean((originals - pca_reconstructions) ** 2)
        
        # Calculate autoencoder reconstruction error
        _, ae_reconstructions = self.get_reconstructions()
        if len(ae_reconstructions) > 0:
            ae_mse = np.mean((originals - ae_reconstructions) ** 2)
            improvement = ((pca_mse - ae_mse) / pca_mse) * 100
        else:
            # For models without reconstruction (like MLP), use PCA as baseline
            ae_mse = pca_mse
            improvement = 0.0
        
        return {
            'pca_mse': float(pca_mse),
            'ae_mse': float(ae_mse),
            'improvement_over_pca': float(improvement)
        }
    
    def plot_manifold_coordinates(self, embeddings: np.ndarray):
        """Plot manifold coordinates."""
        if len(embeddings) == 0:
            return
        
        fig, axes = plt.subplots(1, min(3, embeddings.shape[1]), figsize=(15, 5))
        if embeddings.shape[1] == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i < embeddings.shape[1]:
                ax.hist(embeddings[:, i], bins=50, alpha=0.7)
                ax.set_xlabel(f'Latent Dimension {i+1}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of Latent Dimension {i+1}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'manifold_coordinates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reconstruction_quality(self, originals: np.ndarray, reconstructions: np.ndarray):
        """Plot reconstruction quality."""
        if len(originals) == 0:
            return
        
        fig, axes = plt.subplots(1, self.state_dim, figsize=(15, 5))
        if self.state_dim == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.scatter(originals[:, i], reconstructions[:, i], alpha=0.5, s=1)
            ax.plot([originals[:, i].min(), originals[:, i].max()], 
                   [originals[:, i].min(), originals[:, i].max()], 'r--', linewidth=2)
            ax.set_xlabel(f'Original {self.state_names[i]}')
            ax.set_ylabel(f'Reconstructed {self.state_names[i]}')
            ax.set_title(f'Reconstruction Quality: {self.state_names[i]}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'reconstruction_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectories_in_manifold(self, embeddings: np.ndarray):
        """Plot trajectories in manifold space."""
        if len(embeddings) == 0 or embeddings.shape[1] < 2:
            return
        
        # Reshape embeddings back to trajectory format
        num_traj = 50  # Use last 50 trajectories as test
        time_points = len(embeddings) // num_traj
        
        if time_points < 2:
            return
        
        fig = plt.figure(figsize=(10, 8))
        
        if embeddings.shape[1] == 2:
            ax = fig.add_subplot(111)
            for i in range(num_traj):
                start_idx = i * time_points
                end_idx = (i + 1) * time_points
                traj_emb = embeddings[start_idx:end_idx]
                ax.plot(traj_emb[:, 0], traj_emb[:, 1], alpha=0.7, linewidth=1)
                ax.scatter(traj_emb[0, 0], traj_emb[0, 1], c='red', s=50, zorder=5)
                ax.scatter(traj_emb[-1, 0], traj_emb[-1, 1], c='blue', s=50, zorder=5)
            
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.set_title('Trajectories in Latent Space')
            ax.grid(True, alpha=0.3)
        
        elif embeddings.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            for i in range(num_traj):
                start_idx = i * time_points
                end_idx = (i + 1) * time_points
                traj_emb = embeddings[start_idx:end_idx]
                ax.plot(traj_emb[:, 0], traj_emb[:, 1], traj_emb[:, 2], alpha=0.7, linewidth=1)
                ax.scatter(traj_emb[0, 0], traj_emb[0, 1], traj_emb[0, 2], c='red', s=50, zorder=5)
                ax.scatter(traj_emb[-1, 0], traj_emb[-1, 1], traj_emb[-1, 2], c='blue', s=50, zorder=5)
            
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.set_zlabel('Latent Dimension 3')
            ax.set_title('Trajectories in Latent Space')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'trajectories_in_manifold.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latent_trajectories_for_monotonicity(self, embeddings: np.ndarray):
        """Plot latent trajectories to analyze monotonicity."""
        if len(embeddings) == 0:
            return
        
        # Reshape embeddings back to trajectory format
        num_traj = 50
        time_points = len(embeddings) // num_traj
        
        if time_points < 2:
            return
        
        fig, axes = plt.subplots(1, min(3, embeddings.shape[1]), figsize=(15, 5))
        if embeddings.shape[1] == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i < embeddings.shape[1]:
                for j in range(num_traj):
                    start_idx = j * time_points
                    end_idx = (j + 1) * time_points
                    traj_emb = embeddings[start_idx:end_idx, i]
                    time_steps = np.arange(len(traj_emb))
                    ax.plot(time_steps, traj_emb, alpha=0.7, linewidth=1)
                
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'Latent Dimension {i+1}')
                ax.set_title(f'Latent Trajectories: Dimension {i+1}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latent_trajectories_monotonicity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectory_reconstruction_comparison(self, originals: np.ndarray, reconstructions: np.ndarray):
        """Plot trajectory reconstruction comparison."""
        if len(originals) == 0:
            return
        
        # Use a few sample trajectories
        num_traj = min(5, len(originals) // 100)
        time_points = 100
        
        fig, axes = plt.subplots(num_traj, self.state_dim, figsize=(15, 3*num_traj))
        if num_traj == 1:
            axes = axes.reshape(1, -1)
        
        for traj_idx in range(num_traj):
            start_idx = traj_idx * time_points
            end_idx = (traj_idx + 1) * time_points
            
            for dim_idx in range(self.state_dim):
                ax = axes[traj_idx, dim_idx]
                time_steps = np.arange(time_points)
                
                ax.plot(time_steps, originals[start_idx:end_idx, dim_idx], 'b-', label='Original', linewidth=2)
                ax.plot(time_steps, reconstructions[start_idx:end_idx, dim_idx], 'r--', label='Reconstructed', linewidth=2)
                
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'{self.state_names[dim_idx]}')
                ax.set_title(f'Trajectory {traj_idx+1}: {self.state_names[dim_idx]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'trajectory_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON."""
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def plot_mlp_dynamics_analysis(self):
        """Generate plots specific to MLP dynamics models."""
        print("Generating MLP dynamics analysis plots...")
        
        # Get test data
        test_traj = self.dataset['trajectories'][-50:]  # Use last 50 trajectories as test
        test_states = test_traj.reshape(-1, test_traj.shape[-1])
        
        # Test single-step prediction
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_states)
            
            # Single step prediction
            predictions = self.model(test_tensor).cpu().numpy()
            
            # Plot prediction vs actual for next step
            fig, axes = plt.subplots(1, self.state_dim, figsize=(15, 5))
            if self.state_dim == 1:
                axes = [axes]
            
            for i, ax in enumerate(axes):
                # For single step, we compare current state with predicted next state
                # We'll use a subset for visualization
                subset_size = min(1000, len(test_states))
                ax.scatter(test_states[:subset_size, i], predictions[:subset_size, i], alpha=0.5, s=1)
                ax.plot([test_states[:, i].min(), test_states[:, i].max()], 
                       [test_states[:, i].min(), test_states[:, i].max()], 'r--', linewidth=2)
                ax.set_xlabel(f'Current {self.state_names[i]}')
                ax.set_ylabel(f'Predicted Next {self.state_names[i]}')
                ax.set_title(f'MLP Dynamics: {self.state_names[i]}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'mlp_dynamics_prediction.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("MLP dynamics analysis plots generated")
    
    def plot_trajectory_dynamics_prediction(self):
        """Plot trajectory predictions S_p steps ahead vs actual values."""
        print("Generating trajectory dynamics prediction plots...")
        
        try:
            # Determine prediction steps from training config
            s_p = self.config.get('training', {}).get('losses', {}).get('S_p', 1)
            print(f"Using S_p = {s_p}")
            
            # Use the dataset trajectories directly
            test_trajectories = self.dataset['trajectories'][-20:]  # Use last 20 trajectories
            num_traj, time_points, _ = test_trajectories.shape
            
            # For visualization, use fewer trajectories
            n_sample = min(3, num_traj)
            indices = np.random.choice(num_traj, n_sample, replace=False)
            
            import torch
            self.model.eval()
            
            # Create subplots for each coordinate
            fig, axes = plt.subplots(self.state_dim, 1, figsize=(15, 4*self.state_dim))
            if self.state_dim == 1:
                axes = [axes]
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_sample))
            
            for coord_idx, state_name in enumerate(self.state_names):
                ax = axes[coord_idx]
                
                for traj_idx, color in zip(indices, colors):
                    trajectory = test_trajectories[traj_idx]
                    
                    # Plot original trajectory
                    time_axis = np.arange(time_points)
                    ax.plot(time_axis, trajectory[:, coord_idx], 
                           color=color, alpha=0.8, linewidth=2, 
                           label=f'Original Traj {traj_idx}' if coord_idx == 0 else "")
                    
                    # Generate predicted trajectory by predicting S_p steps ahead from each timestep
                    predicted_trajectory = []
                    with torch.no_grad():
                        for start_step in range(time_points - s_p):
                            x_current = torch.FloatTensor(trajectory[start_step:start_step+1])
                            
                            if s_p == 1:
                                # Single step prediction
                                if hasattr(self.model, 'forward_prediction'):
                                    _, _, _, x_next_pred = self.model.forward_prediction(x_current)
                                else:
                                    x_next_pred = self.model(x_current)
                                predicted_trajectory.append(x_next_pred.cpu().numpy()[0, coord_idx])
                            else:
                                # Multi-step prediction
                                if hasattr(self.model, 'forward_multistep_prediction'):
                                    _, _, _, pred_states = self.model.forward_multistep_prediction(x_current, s_p)
                                    # Take the last predicted state (S_p steps ahead)
                                    x_s_p_ahead = pred_states[-1]
                                else:
                                    pred_states = self.model.forward_multistep_prediction(x_current, s_p)
                                    x_s_p_ahead = pred_states[-1]
                                predicted_trajectory.append(x_s_p_ahead.cpu().numpy()[0, coord_idx])
                    
                    # Plot predicted trajectory
                    pred_time_axis = np.arange(s_p, time_points)
                    ax.plot(pred_time_axis, predicted_trajectory, 
                           color=color, alpha=0.6, linewidth=1, linestyle='--',
                           label=f'Predicted Traj {traj_idx}' if coord_idx == 0 else "")
                
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'{state_name}')
                title_suffix = f"S_p={s_p}" if s_p > 1 else "S_p=1"
                ax.set_title(f'{state_name} Coordinate: Original vs {title_suffix} Step Ahead Predictions')
                ax.grid(True, alpha=0.3)
                
                if coord_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plot_path = self.plots_dir / 'trajectory_dynamics_prediction.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Trajectory dynamics prediction plots saved to: {plot_path}")
            
        except Exception as e:
            print(f"Error in plot_trajectory_dynamics_prediction: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report."""
        report_path = self.output_dir / 'evaluation_report.md'
        
        # Safely get values with defaults
        recon_error = results.get('reconstruction_error', {})
        manifold_quality = results.get('manifold_quality', {})
        pca_comparison = results.get('pca_comparison', {})
        model_type = results.get('model_type', 'autoencoder')
        
        if model_type == 'mlp_dynamics':
            report = f"""# MLP Dynamics Model Evaluation Report

## Overview
- **Model Type**: {type(self.model).__name__}
- **State Dimension**: {self.state_dim}
- **Model Architecture**: Direct state-space dynamics prediction

## Model Information
- **Number of Parameters**: {getattr(self.model, 'get_num_parameters', lambda: 'N/A')()}
- **Dynamics Layers**: {getattr(self.model, 'get_model_info', lambda: {})().get('dynamics_layers', 'N/A')}

## Evaluation Notes
This is an MLP dynamics model that operates directly in state space without a latent representation.
- **No Latent Space**: Model predicts next states directly from current states
- **No Reconstruction**: Model does not reconstruct states, only predicts dynamics
- **Dynamics Focus**: Evaluation focuses on prediction capabilities rather than manifold learning

## Generated Plots
- MLP dynamics prediction analysis
- Single-step prediction quality
"""
        else:
            report = f"""# Model Evaluation Report

## Overview
- **Model Type**: {type(self.model).__name__}
- **State Dimension**: {self.state_dim}
- **Latent Dimension**: {getattr(self.model, 'latent_dim', 'N/A')}

## Reconstruction Error
- **MSE**: {recon_error.get('mse', 0.0):.6f}
- **MAE**: {recon_error.get('mae', 0.0):.6f}
- **RMSE**: {recon_error.get('rmse', 0.0):.6f}
- **Mean Correlation**: {recon_error.get('mean_correlation', 0.0):.4f}

## Manifold Quality
- **Distance Correlation**: {manifold_quality.get('distance_correlation', 0.0):.4f}
- **Effective Dimensionality**: {manifold_quality.get('effective_dimensionality', 0.0):.2f}

## PCA Comparison
- **PCA MSE**: {pca_comparison.get('pca_mse', 0.0):.6f}
- **Autoencoder MSE**: {pca_comparison.get('ae_mse', 0.0):.6f}
- **Improvement over PCA**: {pca_comparison.get('improvement_over_pca', 0.0):.1f}%

## Generated Plots
- Manifold coordinates distribution
- Reconstruction quality scatter plots
- Trajectories in latent space
- Trajectory reconstruction comparison
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report


class StateAutoencoderEvaluator:
    """Evaluator for state-space autoencoder (manifold learning)."""
    
    def __init__(self, config: Config, dataset: Dict[str, Any], 
                 trainer, output_dir: Path):
        """Initialize evaluator.
        
        Args:
            config: Configuration object
            dataset: Original dataset
            trainer: Trained state autoencoder trainer
            output_dir: Output directory for plots and results
        """
        self.config = config
        self.dataset = dataset
        self.trainer = trainer
        self.output_dir = output_dir
        
        # Get data shapes
        self.trajectory_shape = dataset['trajectories'].shape
        self.state_dim = self.trajectory_shape[2]
        self.state_names = dataset['state_names']
        
        # Create plots directory
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Run evaluation for manifold learning."""
        print("Running manifold evaluation...")
        
        results = {}
        
        # Get embeddings and reconstructions
        embeddings = self.get_embeddings()
        originals, reconstructions = self.get_reconstructions()
        
        # Basic reconstruction metrics
        results['reconstruction_error'] = self.evaluate_reconstruction_error(
            originals, reconstructions
        )
        
        # Manifold quality metrics
        results['manifold_quality'] = self.evaluate_manifold_quality(
            originals, embeddings
        )
        
        # Compare with PCA
        results['pca_comparison'] = self.compare_with_pca(originals, embeddings)
        
        # Generate visualizations
        self.plot_manifold_coordinates(embeddings)
        self.plot_reconstruction_quality(originals, reconstructions)
        self.plot_trajectories_in_manifold(embeddings)
        
        # Add monotonicity analysis for monotonic dynamics
        if hasattr(self.model, 'latent_dynamics') and self.model.latent_dynamics is not None:
            if hasattr(self.model.latent_dynamics, 'weight_reparam'):
                print(f"\nDetected monotonic dynamics with {self.model.latent_dynamics.weight_reparam} reparameterization")
                print("Generating monotonicity analysis plots...")
                self.plot_latent_trajectories_for_monotonicity(embeddings)
        
        self.plot_trajectory_reconstruction_comparison(originals, reconstructions)
        
        # Note: Training curves are not available for loaded models
        
        # Plot latent dynamics effectiveness for prediction mode
        # Note: Mode is always 'prediction' for loaded models
        # Add dynamics prediction comparison
        self.plot_dynamics_prediction_comparison()
        
        # Add trajectory dynamics prediction plots
        self.plot_trajectory_dynamics_prediction()
        
        # Save results
        self.save_evaluation_results(results)
        
        print("Evaluation completed!")
        return results
    
    def evaluate_reconstruction_error(self, originals: np.ndarray, 
                                    reconstructions: np.ndarray) -> Dict[str, float]:
        """Evaluate reconstruction error for state points."""
        # Mean squared error
        mse = np.mean((originals - reconstructions) ** 2)
        
        # Mean absolute error
        mae = np.mean(np.abs(originals - reconstructions))
        
        # Per-dimension errors
        per_dim_mse = np.mean((originals - reconstructions) ** 2, axis=0)
        
        # Correlation per dimension
        correlations = []
        for i in range(originals.shape[1]):
            corr = np.corrcoef(originals[:, i], reconstructions[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        mean_correlation = np.mean(correlations)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'mean_correlation': float(mean_correlation),
            'per_dim_mse': per_dim_mse.tolist(),
            'correlations': correlations
        }
    
    def evaluate_manifold_quality(self, originals: np.ndarray, 
                                 embeddings: np.ndarray) -> Dict[str, Any]:
        """Evaluate quality of learned manifold coordinates."""
        # Distance preservation
        sample_size = min(5000, len(originals))
        orig_sample = originals[:sample_size]
        embed_sample = embeddings[:sample_size]
        
        orig_distances = pairwise_distances(orig_sample)
        embed_distances = pairwise_distances(embed_sample.reshape(-1, 1) if embeddings.shape[1] == 1 else embed_sample)
        
        # Flatten upper triangular matrices
        orig_flat = orig_distances[np.triu_indices_from(orig_distances, k=1)]
        embed_flat = embed_distances[np.triu_indices_from(embed_distances, k=1)]
        
        distance_correlation = np.corrcoef(orig_flat, embed_flat)[0, 1]
        if np.isnan(distance_correlation):
            distance_correlation = 0.0
        
        # Embedding statistics
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        # Effective dimensionality
        if embeddings.shape[1] == 1:
            # For 1D embeddings, variance is just the variance of the single dimension
            variance = np.var(embeddings[:, 0])
            eigenvalues = np.array([variance])
            explained_variance_ratio = np.array([1.0])
            effective_dim = 1.0
        else:
            cov_matrix = np.cov(embeddings.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove negative/zero
            
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            effective_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        return {
            'distance_correlation': float(distance_correlation),
            'mean_embedding': mean_embedding.tolist(),
            'std_embedding': std_embedding.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'effective_dimensionality': float(effective_dim),
            'embedding_range': {
                'min': np.min(embeddings, axis=0).tolist(),
                'max': np.max(embeddings, axis=0).tolist()
            }
        }
    
    def compare_with_pca(self, originals: np.ndarray, 
                        embeddings: np.ndarray) -> Dict[str, Any]:
        """Compare learned manifold with PCA."""
        if len(embeddings) == 0:
            return {'pca_mse': 0.0, 'ae_mse': 0.0, 'improvement_over_pca': 0.0}
        
        # Ensure PCA components don't exceed the minimum of samples and features
        max_components = min(embeddings.shape[1], originals.shape[1], len(originals))
        if max_components < 1:
            return {'pca_mse': 0.0, 'ae_mse': 0.0, 'improvement_over_pca': 0.0}
        
        # Fit PCA to original data with appropriate number of components
        pca = PCA(n_components=max_components)
        pca_embeddings = pca.fit_transform(originals)
        
        # Reconstruct from PCA embeddings
        pca_reconstructions = pca.inverse_transform(pca_embeddings)
        
        # Calculate PCA reconstruction error
        pca_mse = np.mean((originals - pca_reconstructions) ** 2)
        
        # Calculate autoencoder reconstruction error
        _, ae_reconstructions = self.get_reconstructions()
        if len(ae_reconstructions) > 0:
            ae_mse = np.mean((originals - ae_reconstructions) ** 2)
            improvement = ((pca_mse - ae_mse) / pca_mse) * 100
        else:
            # For models without reconstruction (like MLP), use PCA as baseline
            ae_mse = pca_mse
            improvement = 0.0
        
        return {
            'pca_mse': float(pca_mse),
            'ae_mse': float(ae_mse),
            'improvement_over_pca': float(improvement)
        }
    
    def plot_manifold_coordinates(self, embeddings: np.ndarray):
        """Plot the learned manifold coordinates."""
        latent_dim = embeddings.shape[1]
        
        if latent_dim == 1:
            # 1D histogram plot
            plt.figure(figsize=(10, 6))
            plt.hist(embeddings[:, 0], bins=100, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            plt.xlabel('Manifold Coordinate')
            plt.ylabel('Density')
            plt.title('Learned Manifold Coordinate Distribution (1D)')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / 'manifold_coordinates_1d.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 1D scatter plot over time
            plt.figure(figsize=(12, 6))
            plt.scatter(range(len(embeddings)), embeddings[:, 0], 
                       alpha=0.6, c=range(len(embeddings)), cmap='viridis', s=1)
            plt.colorbar(label='State Point Index')
            plt.xlabel('State Point Index')
            plt.ylabel('Manifold Coordinate')
            plt.title('Manifold Coordinate Evolution')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / 'manifold_evolution_1d.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        elif latent_dim >= 2:
            # 2D scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                                alpha=0.6, c=range(len(embeddings)), cmap='viridis', s=1)
            plt.colorbar(scatter, label='State Point Index')
            plt.xlabel('Manifold Coordinate 1')
            plt.ylabel('Manifold Coordinate 2')
            plt.title('Learned Manifold Coordinates (2D)')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / 'manifold_coordinates_2d.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        if latent_dim >= 3:
            # 3D scatter plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                               c=range(len(embeddings)), cmap='viridis', alpha=0.6, s=1)
            ax.set_xlabel('Manifold Coordinate 1')
            ax.set_ylabel('Manifold Coordinate 2')
            ax.set_zlabel('Manifold Coordinate 3')
            ax.set_title('Learned Manifold Coordinates (3D)')
            plt.colorbar(scatter, label='State Point Index')
            plt.savefig(self.plots_dir / 'manifold_coordinates_3d.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Distribution plots for each coordinate
        fig, axes = plt.subplots(1, latent_dim, figsize=(4*latent_dim, 4))
        if latent_dim == 1:
            axes = [axes]
        
        for i in range(latent_dim):
            axes[i].hist(embeddings[:, i], bins=50, alpha=0.7, density=True)
            axes[i].set_xlabel(f'Manifold Coordinate {i+1}')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Distribution of Coordinate {i+1}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'coordinate_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_reconstruction_quality(self, originals: np.ndarray, 
                                   reconstructions: np.ndarray):
        """Plot reconstruction quality for state points."""
        # Scatter plots: original vs reconstructed for each dimension
        fig, axes = plt.subplots(1, self.state_dim, figsize=(5*self.state_dim, 5))
        if self.state_dim == 1:
            axes = [axes]
        
        # Sample points for visualization
        n_sample = min(5000, len(originals))
        indices = np.random.choice(len(originals), n_sample, replace=False)
        
        for i, state_name in enumerate(self.state_names):
            orig_dim = originals[indices, i]
            recon_dim = reconstructions[indices, i]
            
            axes[i].scatter(orig_dim, recon_dim, alpha=0.5, s=1)
            
            # Perfect reconstruction line
            min_val = min(orig_dim.min(), recon_dim.min())
            max_val = max(orig_dim.max(), recon_dim.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[i].set_xlabel(f'Original {state_name}')
            axes[i].set_ylabel(f'Reconstructed {state_name}')
            axes[i].set_title(f'{state_name} Reconstruction')
            axes[i].grid(True, alpha=0.3)
            
            # Compute and display correlation
            corr = np.corrcoef(orig_dim, recon_dim)[0, 1]
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'reconstruction_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectories_in_manifold(self, embeddings: np.ndarray):
        """Plot how trajectories look in the manifold coordinates."""
        # Reshape embeddings back to trajectory format
        num_traj, time_points, _ = self.trajectory_shape
        # Only use test set embeddings, so calculate the actual number available
        available_points = len(embeddings)
        points_per_traj = available_points // num_traj
        actual_points = num_traj * points_per_traj
        traj_embeddings = embeddings[:actual_points].reshape(num_traj, points_per_traj, -1)
        
        latent_dim = embeddings.shape[1]
        
        if latent_dim == 1:
            # Plot trajectories in 1D manifold coordinate vs time
            plt.figure(figsize=(12, 8))
            
            # Plot a few sample trajectories
            n_sample = min(10, num_traj)
            indices = np.random.choice(num_traj, n_sample, replace=False)
            
            time_axis = np.arange(points_per_traj)
            colors = plt.cm.Set1(np.linspace(0, 1, n_sample))
            
            for idx, color in zip(indices, colors):
                traj = traj_embeddings[idx]
                plt.plot(time_axis, traj[:, 0], alpha=0.7, linewidth=2, 
                        color=color, label=f'Trajectory {idx}')
            
            plt.xlabel('Time Step')
            plt.ylabel('Manifold Coordinate')
            plt.title('Sample Trajectories in 1D Manifold Coordinate')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(self.plots_dir / 'trajectories_in_manifold_1d.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        elif latent_dim >= 2:
            plt.figure(figsize=(10, 8))
            
            # Plot a few sample trajectories
            n_sample = min(10, num_traj)
            indices = np.random.choice(num_traj, n_sample, replace=False)
            
            for idx in indices:
                traj = traj_embeddings[idx]
                plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=1)
            
            plt.xlabel('Manifold Coordinate 1')
            plt.ylabel('Manifold Coordinate 2')
            plt.title('Sample Trajectories in Manifold Coordinates')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / 'trajectories_in_manifold.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_latent_trajectories_for_monotonicity(self, embeddings: np.ndarray):
        """Plot latent trajectories to visualize monotonicity in the latent space.
        
        This method creates plots that help visualize whether the dynamics in the latent
        space are monotonic, which is enforced by the MonotonicNetwork architecture.
        """
        # Reshape embeddings back to trajectory format
        num_traj, time_points, _ = self.trajectory_shape
        available_points = len(embeddings)
        points_per_traj = available_points // num_traj
        actual_points = num_traj * points_per_traj
        traj_embeddings = embeddings[:actual_points].reshape(num_traj, points_per_traj, -1)
        
        latent_dim = embeddings.shape[1]
        
        # Sample a few trajectories for visualization
        n_sample = min(8, num_traj)
        indices = np.random.choice(num_traj, n_sample, replace=False)
        
        # Create time axis
        time_axis = np.arange(points_per_traj)
        
        # Plot 1: Individual latent coordinate evolution over time
        fig, axes = plt.subplots(latent_dim, 1, figsize=(14, 4*latent_dim))
        if latent_dim == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_sample))
        
        for coord_idx in range(latent_dim):
            ax = axes[coord_idx]
            
            for traj_idx, color in zip(indices, colors):
                traj = traj_embeddings[traj_idx]
                ax.plot(time_axis, traj[:, coord_idx], 
                       color=color, alpha=0.8, linewidth=2, 
                       label=f'Trajectory {traj_idx}')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Latent Coordinate {coord_idx+1}')
            ax.set_title(f'Latent Coordinate {coord_idx+1} Evolution Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latent_coordinate_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Phase space plots for pairs of latent coordinates
        if latent_dim >= 2:
            # Create subplots for coordinate pairs
            n_pairs = latent_dim // 2 + (latent_dim % 2)
            fig, axes = plt.subplots(2, n_pairs, figsize=(6*n_pairs, 10))
            if n_pairs == 1:
                axes = axes.reshape(2, 1)
            
            pair_idx = 0
            for i in range(0, latent_dim-1, 2):
                # Plot coordinate i vs coordinate i+1
                ax1 = axes[0, pair_idx] if n_pairs > 1 else axes[0]
                ax2 = axes[1, pair_idx] if n_pairs > 1 else axes[1]
                
                for traj_idx, color in zip(indices, colors):
                    traj = traj_embeddings[traj_idx]
                    # Phase space plot
                    ax1.plot(traj[:, i], traj[:, i+1], alpha=0.7, linewidth=1.5, color=color)
                    # Time evolution of the pair
                    ax2.plot(time_axis, traj[:, i], alpha=0.7, linewidth=1.5, color=color, 
                            label=f'Coord {i+1}' if traj_idx == 0 else "")
                    ax2.plot(time_axis, traj[:, i+1], alpha=0.7, linewidth=1.5, color=color, 
                            linestyle='--', label=f'Coord {i+2}' if traj_idx == 0 else "")
                
                ax1.set_xlabel(f'Latent Coordinate {i+1}')
                ax1.set_ylabel(f'Latent Coordinate {i+2}')
                ax1.set_title(f'Phase Space: Coord {i+1} vs Coord {i+2}')
                ax1.grid(True, alpha=0.3)
                
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Coordinate Value')
                ax2.set_title(f'Time Evolution: Coord {i+1} vs Coord {i+2}')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                pair_idx += 1
            
            # Handle odd number of coordinates
            if latent_dim % 2 == 1:
                ax = axes[1, -1] if n_pairs > 1 else axes[1]
                for traj_idx, color in zip(indices, colors):
                    traj = traj_embeddings[traj_idx]
                    ax.plot(time_axis, traj[:, -1], alpha=0.7, linewidth=1.5, color=color)
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'Latent Coordinate {latent_dim}')
                ax.set_title(f'Time Evolution: Coord {latent_dim}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'latent_phase_space_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Monotonicity analysis - check if trajectories are generally increasing/decreasing
        fig, axes = plt.subplots(latent_dim, 1, figsize=(14, 4*latent_dim))
        if latent_dim == 1:
            axes = [axes]
        
        for coord_idx in range(latent_dim):
            ax = axes[coord_idx]
            
            # Calculate trajectory trends (first derivative approximation)
            for traj_idx, color in zip(indices, colors):
                traj = traj_embeddings[traj_idx]
                coord_values = traj[:, coord_idx]
                
                # Calculate differences between consecutive time steps
                differences = np.diff(coord_values)
                
                # Plot the differences to show monotonicity
                ax.plot(time_axis[:-1], differences, 
                       color=color, alpha=0.8, linewidth=2, 
                       label=f'Trajectory {traj_idx}')
                
                # Add horizontal line at zero for reference
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Δ(Coord {coord_idx+1})')
            ax.set_title(f'Monotonicity Analysis: Changes in Latent Coordinate {coord_idx+1}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add text box with monotonicity statistics
            all_differences = []
            for traj_idx in indices:
                traj = traj_embeddings[traj_idx]
                coord_values = traj[:, coord_idx]
                differences = np.diff(coord_values)
                all_differences.extend(differences)
            
            all_differences = np.array(all_differences)
            positive_changes = np.sum(all_differences > 0)
            negative_changes = np.sum(all_differences < 0)
            zero_changes = np.sum(all_differences == 0)
            total_changes = len(all_differences)
            
            stats_text = f'Positive: {positive_changes}/{total_changes} ({100*positive_changes/total_changes:.1f}%)\n'
            stats_text += f'Negative: {negative_changes}/{total_changes} ({100*negative_changes/total_changes:.1f}%)\n'
            stats_text += f'Zero: {zero_changes}/{total_changes} ({100*zero_changes/total_changes:.1f}%)'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latent_monotonicity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Latent trajectory monotonicity plots saved to: {self.plots_dir}")
        print("Generated plots:")
        print("  - latent_coordinate_evolution.png")
        print("  - latent_phase_space_analysis.png")
        print("  - latent_monotonicity_analysis.png")
    
    def plot_trajectory_reconstruction_comparison(self, originals: np.ndarray, 
                                                reconstructions: np.ndarray):
        """Plot original vs reconstructed trajectories in real coordinates."""
        # Use the dataset trajectories directly
        test_trajectories = self.dataset['trajectories'][-50:]  # Use last 50 trajectories as test
        num_traj, time_points, _ = test_trajectories.shape
        
        # Reconstruct test trajectories by passing all points through the model
        # Flatten test trajectories to individual points
        test_orig_points = test_trajectories.reshape(-1, self.state_dim)
        
        # Get reconstructions for test points
        import torch
        self.model.eval()
        with torch.no_grad():
            test_orig_tensor = torch.FloatTensor(test_orig_points)
            _, test_recon_tensor = self.model(test_orig_tensor)
            test_recon_points = test_recon_tensor.cpu().numpy()
        
        # Reshape back to trajectory format
        orig_traj = test_orig_points.reshape(num_traj, time_points, self.state_dim)
        recon_traj = test_recon_points.reshape(num_traj, time_points, self.state_dim)
        
        # Create time axis
        time_axis = np.linspace(0, time_points-1, time_points)
        
        # Plot a few sample trajectories
        n_sample = min(5, num_traj)
        indices = np.random.choice(num_traj, n_sample, replace=False)
        
        # Create subplots for each coordinate
        fig, axes = plt.subplots(self.state_dim, 1, figsize=(12, 4*self.state_dim))
        if self.state_dim == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_sample))
        
        for coord_idx, state_name in enumerate(self.state_names):
            ax = axes[coord_idx]
            
            for traj_idx, color in zip(indices, colors):
                # Original trajectory
                ax.plot(time_axis, orig_traj[traj_idx, :, coord_idx], 
                       color=color, alpha=0.8, linewidth=2, 
                       label=f'Original Traj {traj_idx}' if coord_idx == 0 else "")
                
                # Reconstructed trajectory
                ax.plot(time_axis, recon_traj[traj_idx, :, coord_idx], 
                       color=color, alpha=0.6, linewidth=1, linestyle='--',
                       label=f'Reconstructed Traj {traj_idx}' if coord_idx == 0 else "")
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{state_name}')
            ax.set_title(f'{state_name} Coordinate: Original vs Reconstructed Trajectories')
            ax.grid(True, alpha=0.3)
            
            # Add legend only for the first subplot
            if coord_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'trajectory_reconstruction_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a single trajectory detailed comparison
        if n_sample > 0:
            fig, axes = plt.subplots(self.state_dim, 1, figsize=(15, 4*self.state_dim))
            if self.state_dim == 1:
                axes = [axes]
            
            # Pick one trajectory for detailed view
            traj_idx = indices[0]
            
            for coord_idx, state_name in enumerate(self.state_names):
                ax = axes[coord_idx]
                
                orig_coord = orig_traj[traj_idx, :, coord_idx]
                recon_coord = recon_traj[traj_idx, :, coord_idx]
                
                # Plot original and reconstructed
                ax.plot(time_axis, orig_coord, 'b-', linewidth=2, 
                       label='Original', alpha=0.8)
                ax.plot(time_axis, recon_coord, 'r--', linewidth=2, 
                       label='Reconstructed', alpha=0.8)
                
                # Plot difference (error)
                error = orig_coord - recon_coord
                ax2 = ax.twinx()
                ax2.plot(time_axis, error, 'g-', alpha=0.5, linewidth=1)
                ax2.set_ylabel('Reconstruction Error', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'{state_name}')
                ax.set_title(f'{state_name} Coordinate: Detailed Comparison (Trajectory {traj_idx})')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
    
    def plot_dynamics_prediction_comparison(self):
        """Plot dynamics prediction K steps ahead (K=1 defaults to one-step)."""
        # Note: Mode is always 'prediction' for loaded models
        
        # Determine horizon K from training config (fallback to 1)
        k_steps = self.config.get('training', {}).get('losses', {}).get('S_p', 1)

        # Use the dataset trajectories directly
        test_trajectories = self.dataset['trajectories'][-50:]  # Use last 50 trajectories as test
        num_traj, time_points, _ = test_trajectories.shape

        import torch
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for trajectory in test_trajectories:
                pred_traj = []

                if k_steps <= 1:
                    # One-step ahead for each timestep
                    for step in range(time_points - 1):
                        x_current = torch.FloatTensor(trajectory[step:step+1])
                        if hasattr(self.model, 'forward_prediction'):
                            _, _, _, x_next_pred = self.model.forward_prediction(x_current)
                        else:
                            # For MLP models, just use forward pass
                            x_next_pred = self.model(x_current)
                        pred_traj.append(x_next_pred.cpu().numpy())
                    predictions.append(np.vstack(pred_traj))
                else:
                    # K-step ahead: from each time t, predict t+K using multi-step latent rollout
                    for step in range(time_points - k_steps):
                        x_current = torch.FloatTensor(trajectory[step:step+1])
                        if hasattr(self.model, 'forward_multistep_prediction'):
                            _, _, _, pred_states = self.model.forward_multistep_prediction(x_current, k_steps)
                            x_k_plus_K = pred_states[-1]
                        else:
                            # For MLP models, use the built-in method
                            pred_states = self.model.forward_multistep_prediction(x_current, k_steps)
                            x_k_plus_K = pred_states[-1]
                        pred_traj.append(x_k_plus_K.cpu().numpy())
                    predictions.append(np.vstack(pred_traj))

        # Time axis alignment
        if k_steps <= 1:
            pred_time_axis = np.arange(1, time_points)
        else:
            pred_time_axis = np.arange(k_steps, time_points)

        # Plot a few sample trajectories
        n_sample = min(5, num_traj)
        indices = np.random.choice(num_traj, n_sample, replace=False)

        # Create subplots for each coordinate
        fig, axes = plt.subplots(self.state_dim, 1, figsize=(12, 4*self.state_dim))
        if self.state_dim == 1:
            axes = [axes]

        colors = plt.cm.Set1(np.linspace(0, 1, n_sample))

        for coord_idx, state_name in enumerate(self.state_names):
            ax = axes[coord_idx]

            for traj_idx, color in zip(indices, colors):
                # Original trajectory (full length)
                time_axis = np.arange(time_points)
                ax.plot(
                    time_axis,
                    test_trajectories[traj_idx, :, coord_idx],
                    color=color,
                    alpha=0.8,
                    linewidth=2,
                    label=f'Original Traj {traj_idx}' if coord_idx == 0 else "",
                )

                # K-step ahead predictions (aligned to t+K)
                label = (
                    f'{k_steps}-Step Prediction Traj {traj_idx}' if coord_idx == 0 else ""
                )
                ax.plot(
                    pred_time_axis,
                    predictions[traj_idx][:, coord_idx],
                    color=color,
                    alpha=0.6,
                    linewidth=1,
                    linestyle='--',
                    label=label,
                )

            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{state_name}')
            title_suffix = f"K={k_steps}" if k_steps > 1 else "One-Step"
            ax.set_title(f'{state_name} Coordinate: Original vs {title_suffix} Dynamics Predictions')
            ax.grid(True, alpha=0.3)

            if coord_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'dynamics_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Dynamics prediction comparison plot saved to: {self.plots_dir / 'dynamics_prediction_comparison.png'}"
        )
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file."""
        results_path = self.output_dir / 'evaluation_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
    
    def generate_report(self, results: Dict[str, Any] = None) -> str:
        """Generate evaluation report."""
        if results is None:
            results = self.evaluate_all()
        
        # Get model info safely
        model_info = {}
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
        
        # Get latent dimension safely
        latent_dim = getattr(self.model, 'latent_dim', 'N/A')
        
        report = f"""
# State-Space Autoencoder Evaluation Report

## System Configuration
- System: {self.dataset['system_type']}
- Parameters: {self.dataset['system_parameters']}
- State Dimension: {self.state_dim}
- Latent Dimension: {latent_dim}
- Total State Points: {len(self.dataset['trajectories'])}

## Manifold Learning Performance

### Reconstruction Quality
- MSE: {results['reconstruction_error']['mse']:.6f}
- RMSE: {results['reconstruction_error']['rmse']:.6f}
- Mean Correlation: {results['reconstruction_error']['mean_correlation']:.4f}

### Manifold Quality
- Distance Correlation: {results['manifold_quality']['distance_correlation']:.4f}
- Effective Dimensionality: {results['manifold_quality']['effective_dimensionality']:.2f}

### Comparison with PCA
- AE Reconstruction MSE: {results['pca_comparison']['ae_mse']:.6f}
- PCA Reconstruction MSE: {results['pca_comparison']['pca_mse']:.6f}
- Improvement over PCA: {results['pca_comparison']['improvement_over_pca']:.1f}%

## Model Information
- Model Type: {type(self.model).__name__}
- Number of Parameters: {model_info.get('num_parameters', 'N/A')}
"""
        
        # Add latent dynamics information if available
        if 'latent_dynamics_type' in model_info:
            report += f"""
## Latent Dynamics Analysis
### Latent Dynamics Type: {model_info['latent_dynamics_type'].title()}
- Dynamics Layers: {model_info.get('latent_dynamics_layers', 'N/A')}
"""
            
            if model_info.get('linear_dynamics', False) and 'A_matrix' in model_info:
                A_matrix = model_info['A_matrix']
                report += f"""
### Linear Dynamics Matrix A
The learned linear transformation matrix A (z_k+1 = A * z_k):

```
A = {A_matrix}
```

#### Matrix Properties:
- Shape: {A_matrix.shape}
- Determinant: {np.linalg.det(A_matrix):.6f}
- Trace: {np.trace(A_matrix):.6f}
- Eigenvalues: {np.linalg.eigvals(A_matrix)}
- Condition Number: {np.linalg.cond(A_matrix):.6f}
"""
        
        report += f"""
## Interpretation
The autoencoder learns a {latent_dim}-dimensional manifold coordinate system for the {self.state_dim}-dimensional state space. 
{'Good' if results['reconstruction_error']['mean_correlation'] > 0.95 else 'Moderate' if results['reconstruction_error']['mean_correlation'] > 0.8 else 'Poor'} reconstruction quality suggests the manifold coordinates {'well' if results['reconstruction_error']['mean_correlation'] > 0.95 else 'reasonably' if results['reconstruction_error']['mean_correlation'] > 0.8 else 'poorly'} capture the essential structure of the state space.
"""
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {report_path}")
        return report 
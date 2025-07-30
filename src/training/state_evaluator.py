"""Evaluation for state-space autoencoder (manifold learning)."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from typing import Dict, Any, Tuple
from pathlib import Path
import json

from ..utils.config import Config


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
        embeddings = self.trainer.get_embeddings()
        originals, reconstructions = self.trainer.get_reconstructions()
        
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
        self.plot_trajectory_reconstruction_comparison(originals, reconstructions)
        self.trainer.plot_training_curves(self.plots_dir / 'training_curves.png')
        
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
        """Compare autoencoder embeddings with PCA."""
        # Apply PCA with same dimensionality as embeddings
        n_components = embeddings.shape[1]
        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(originals)
        
        # Compare reconstruction errors
        pca_reconstruction = pca.inverse_transform(pca_embeddings)
        pca_mse = np.mean((originals - pca_reconstruction) ** 2)
        
        # Get AE reconstruction error
        _, ae_reconstructions = self.trainer.get_reconstructions()
        ae_mse = np.mean((originals - ae_reconstructions) ** 2)
        
        # Handle explained variance ratio for different dimensions
        if n_components == 1:
            explained_variance_total = float(pca.explained_variance_ratio_[0])
            explained_variance_components = [float(pca.explained_variance_ratio_[0])]
        else:
            explained_variance_total = float(np.sum(pca.explained_variance_ratio_))
            explained_variance_components = pca.explained_variance_ratio_.tolist()
        
        return {
            'pca_reconstruction_mse': float(pca_mse),
            'ae_reconstruction_mse': float(ae_mse),
            'improvement_over_pca': float((pca_mse - ae_mse) / pca_mse * 100),
            'pca_explained_variance': explained_variance_total,
            'pca_components': explained_variance_components
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
    
    def plot_trajectory_reconstruction_comparison(self, originals: np.ndarray, 
                                                reconstructions: np.ndarray):
        """Plot original vs reconstructed trajectories in real coordinates."""
        # Use the test trajectories that were kept separate during training
        test_trajectories = self.trainer.test_trajectories  # Shape: (test_traj, time_points, state_dim)
        num_traj, time_points, _ = test_trajectories.shape
        
        # Reconstruct test trajectories by passing all points through the model
        # Flatten test trajectories to individual points
        test_orig_points = test_trajectories.reshape(-1, self.state_dim)
        
        # Get reconstructions for test points
        import torch
        self.trainer.model.eval()
        with torch.no_grad():
            test_orig_tensor = torch.FloatTensor(test_orig_points).to(self.trainer.device)
            _, test_recon_tensor = self.trainer.model(test_orig_tensor)
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
                
                # Calculate and display statistics
                mse = np.mean(error**2)
                corr = np.corrcoef(orig_coord, recon_coord)[0, 1]
                ax.text(0.02, 0.98, f'MSE: {mse:.6f}\nCorr: {corr:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'detailed_trajectory_reconstruction.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
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
        
        report = f"""
# State-Space Autoencoder Evaluation Report

## System Configuration
- System: {self.dataset['system_type']}
- Parameters: {self.dataset['system_parameters']}
- State Dimension: {self.state_dim}
- Latent Dimension: {self.config['autoencoder']['latent_dim']}
 - Total State Points: {len(self.trainer.test_dataset) + len(self.trainer.train_dataset) + len(self.trainer.val_dataset)}

## Manifold Learning Performance

### Reconstruction Quality
- MSE: {results['reconstruction_error']['mse']:.6f}
- RMSE: {results['reconstruction_error']['rmse']:.6f}
- Mean Correlation: {results['reconstruction_error']['mean_correlation']:.4f}

### Manifold Quality
- Distance Correlation: {results['manifold_quality']['distance_correlation']:.4f}
- Effective Dimensionality: {results['manifold_quality']['effective_dimensionality']:.2f}

### Comparison with PCA
- AE Reconstruction MSE: {results['pca_comparison']['ae_reconstruction_mse']:.6f}
- PCA Reconstruction MSE: {results['pca_comparison']['pca_reconstruction_mse']:.6f}
- Improvement over PCA: {results['pca_comparison']['improvement_over_pca']:.1f}%

## Training Results
- Best Validation Loss: {self.trainer.best_val_loss:.6f}
- Epochs Trained: {len(self.trainer.training_history['epochs'])}

## Interpretation
The autoencoder learns a {self.config['autoencoder']['latent_dim']}-dimensional manifold coordinate system for the {self.state_dim}-dimensional state space. 
{'Good' if results['reconstruction_error']['mean_correlation'] > 0.95 else 'Moderate' if results['reconstruction_error']['mean_correlation'] > 0.8 else 'Poor'} reconstruction quality suggests the manifold coordinates {'well' if results['reconstruction_error']['mean_correlation'] > 0.95 else 'reasonably' if results['reconstruction_error']['mean_correlation'] > 0.8 else 'poorly'} capture the essential structure of the state space.
"""
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {report_path}")
        return report 
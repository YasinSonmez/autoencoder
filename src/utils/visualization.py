"""Visualization utilities for trajectory data inspection."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List


def plot_trajectory_data(dataset: Dict[str, Any], output_dir: Path, max_trajectories: int = 50):
    """Plot trajectory data for visual inspection.
    
    Args:
        dataset: Dataset dictionary containing trajectories
        output_dir: Directory to save plots
        max_trajectories: Maximum number of trajectories to plot
    """
    trajectories = dataset['trajectories']
    state_names = dataset['state_names']
    system_type = dataset['system_type']
    system_parameters = dataset['system_parameters']
    
    num_traj, time_points, state_dim = trajectories.shape
    
    # Create plots directory
    plots_dir = output_dir / 'data_inspection'
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TRAJECTORY DATA INSPECTION")
    print(f"{'='*60}")
    print(f"System: {system_type}")
    print(f"Parameters: {system_parameters}")
    print(f"Number of trajectories: {num_traj}")
    print(f"Time points per trajectory: {time_points}")
    print(f"State dimension: {state_dim}")
    print(f"State variables: {state_names}")
    
    # Sample trajectories for plotting
    n_plot = min(max_trajectories, num_traj)
    indices = np.random.choice(num_traj, n_plot, replace=False)
    
    if state_dim <= 3:
        # Plot trajectories in phase space
        _plot_phase_space(trajectories[indices], state_names, plots_dir, system_type)
    
    # Always plot time series for each dimension
    _plot_time_series(trajectories[indices], state_names, plots_dir, system_type)
    
    # Plot initial conditions
    _plot_initial_conditions(trajectories, state_names, plots_dir, system_type)
    
    # Plot statistics
    _plot_trajectory_statistics(trajectories, state_names, plots_dir, system_type)
    
    print(f"\nTrajectory plots saved to: {plots_dir}")
    print(f"Generated plots:")
    for plot_file in plots_dir.glob("*.png"):
        print(f"  - {plot_file.name}")


def _plot_phase_space(trajectories: np.ndarray, state_names: List[str], 
                     plots_dir: Path, system_type: str):
    """Plot trajectories in phase space (for 1D, 2D, or 3D systems)."""
    n_traj, time_points, state_dim = trajectories.shape
    
    if state_dim == 1:
        # 1D: Plot trajectory vs time
        plt.figure(figsize=(10, 6))
        for i, traj in enumerate(trajectories):
            time_axis = np.arange(time_points)
            plt.plot(time_axis, traj[:, 0], alpha=0.7, linewidth=1, 
                    label=f'Trajectory {i+1}' if i < 5 else "")
        
        plt.xlabel('Time Step')
        plt.ylabel(state_names[0])
        plt.title(f'{system_type.title()} System - 1D Trajectories')
        plt.grid(True, alpha=0.3)
        if n_traj <= 5:
            plt.legend()
        plt.savefig(plots_dir / 'phase_space_1d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    elif state_dim == 2:
        # 2D: Phase portrait
        plt.figure(figsize=(12, 10))
        
        # Plot all trajectories with low alpha
        for i, traj in enumerate(trajectories):
            alpha = 0.8 if i < 10 else 0.3  # Highlight first 10 trajectories
            linewidth = 1.5 if i < 10 else 0.8
            plt.plot(traj[:, 0], traj[:, 1], alpha=alpha, linewidth=linewidth,
                    label=f'Trajectory {i+1}' if i < 5 else "")
        
        # Mark all starting points
        start_points = trajectories[:, 0, :]
        plt.scatter(start_points[:, 0], start_points[:, 1], 
                   color='green', s=30, alpha=0.7, zorder=5, 
                   label='Start points', edgecolors='darkgreen', linewidth=0.5)
        
        # Mark all ending points
        end_points = trajectories[:, -1, :]
        plt.scatter(end_points[:, 0], end_points[:, 1], 
                   color='red', s=30, alpha=0.7, zorder=5,
                   label='End points', edgecolors='darkred', linewidth=0.5)
        
        plt.xlabel(state_names[0])
        plt.ylabel(state_names[1])
        plt.title(f'{system_type.title()} System - 2D Phase Portrait ({n_traj} trajectories)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plots_dir / 'phase_space_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    elif state_dim == 3:
        # 3D: 3D phase portrait
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all trajectories with varying alpha
        for i, traj in enumerate(trajectories):
            alpha = 0.8 if i < 10 else 0.3  # Highlight first 10 trajectories
            linewidth = 1.2 if i < 10 else 0.6
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=alpha, linewidth=linewidth,
                   label=f'Trajectory {i+1}' if i < 5 else "")
        
        # Mark all starting points
        start_points = trajectories[:, 0, :]
        ax.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2], 
                  color='green', s=40, alpha=0.8, label='Start points',
                  edgecolors='darkgreen', linewidth=0.5)
        
        # Mark all ending points
        end_points = trajectories[:, -1, :]
        ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], 
                  color='red', s=40, alpha=0.8, label='End points',
                  edgecolors='darkred', linewidth=0.5)
        
        ax.set_xlabel(state_names[0])
        ax.set_ylabel(state_names[1])
        ax.set_zlabel(state_names[2])
        ax.set_title(f'{system_type.title()} System - 3D Phase Portrait ({n_traj} trajectories)')
        ax.legend()
        plt.savefig(plots_dir / 'phase_space_3d.png', dpi=300, bbox_inches='tight')
        plt.close()


def _plot_initial_conditions(trajectories: np.ndarray, state_names: List[str], 
                            plots_dir: Path, system_type: str):
    """Plot initial conditions distribution."""
    n_traj, time_points, state_dim = trajectories.shape
    initial_conditions = trajectories[:, 0, :]  # Extract initial points
    
    if state_dim == 1:
        # 1D: Histogram of initial conditions
        plt.figure(figsize=(10, 6))
        plt.hist(initial_conditions[:, 0], bins=30, alpha=0.7, color='green', 
                edgecolor='darkgreen', density=True)
        plt.xlabel(state_names[0])
        plt.ylabel('Density')
        plt.title(f'{system_type.title()} System - Initial Conditions Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'initial_conditions_1d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    elif state_dim == 2:
        # 2D: Scatter plot of initial conditions
        plt.figure(figsize=(10, 8))
        plt.scatter(initial_conditions[:, 0], initial_conditions[:, 1], 
                   alpha=0.7, s=50, c='green', edgecolors='darkgreen', linewidth=0.5)
        plt.xlabel(state_names[0])
        plt.ylabel(state_names[1])
        plt.title(f'{system_type.title()} System - Initial Conditions ({n_traj} points)')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'initial_conditions_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    elif state_dim == 3:
        # 3D: 3D scatter plot of initial conditions
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(initial_conditions[:, 0], initial_conditions[:, 1], initial_conditions[:, 2],
                  alpha=0.7, s=50, c='green', edgecolors='darkgreen', linewidth=0.5)
        ax.set_xlabel(state_names[0])
        ax.set_ylabel(state_names[1])
        ax.set_zlabel(state_names[2])
        ax.set_title(f'{system_type.title()} System - Initial Conditions ({n_traj} points)')
        plt.savefig(plots_dir / 'initial_conditions_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # For higher dimensions, create pairwise plots
    if state_dim > 3:
        # Create pairwise scatter plots
        n_pairs = min(6, state_dim * (state_dim - 1) // 2)  # Limit to 6 pairs
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        pair_idx = 0
        for i in range(state_dim):
            for j in range(i + 1, state_dim):
                if pair_idx >= n_pairs:
                    break
                ax = axes[pair_idx]
                ax.scatter(initial_conditions[:, i], initial_conditions[:, j],
                          alpha=0.7, s=30, c='green', edgecolors='darkgreen', linewidth=0.3)
                ax.set_xlabel(state_names[i])
                ax.set_ylabel(state_names[j])
                ax.set_title(f'{state_names[i]} vs {state_names[j]}')
                ax.grid(True, alpha=0.3)
                pair_idx += 1
            if pair_idx >= n_pairs:
                break
        
        # Hide unused subplots
        for idx in range(pair_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{system_type.title()} System - Initial Conditions Pairwise Plots', fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / 'initial_conditions_pairwise.png', dpi=300, bbox_inches='tight')
        plt.close()


def _plot_time_series(trajectories: np.ndarray, state_names: List[str], 
                     plots_dir: Path, system_type: str):
    """Plot time series for each dimension in separate subplots."""
    n_traj, time_points, state_dim = trajectories.shape
    
    # Create subplots for each dimension
    fig, axes = plt.subplots(state_dim, 1, figsize=(12, 4*state_dim))
    if state_dim == 1:
        axes = [axes]
    
    time_axis = np.arange(time_points)
    colors = plt.cm.Set1(np.linspace(0, 1, min(n_traj, 10)))
    
    for dim in range(state_dim):
        ax = axes[dim]
        
        for i, traj in enumerate(trajectories):
            color = colors[i % len(colors)]
            ax.plot(time_axis, traj[:, dim], alpha=0.7, linewidth=1, color=color,
                   label=f'Trajectory {i+1}' if i < 5 else "")
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(state_names[dim])
        ax.set_title(f'{state_names[dim]} vs Time')
        ax.grid(True, alpha=0.3)
        
        if dim == 0 and n_traj <= 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'{system_type.title()} System - Time Series', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / 'time_series.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_trajectory_statistics(trajectories: np.ndarray, state_names: List[str], 
                               plots_dir: Path, system_type: str):
    """Plot statistical properties of trajectories."""
    n_traj, time_points, state_dim = trajectories.shape
    
    # Calculate statistics
    means = np.mean(trajectories, axis=(0, 1))  # Mean across all trajectories and time
    stds = np.std(trajectories, axis=(0, 1))    # Std across all trajectories and time
    mins = np.min(trajectories, axis=(0, 1))    # Min across all trajectories and time
    maxs = np.max(trajectories, axis=(0, 1))    # Max across all trajectories and time
    
    # Plot distribution for each dimension
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Box plots
    ax = axes[0]
    data_for_box = [trajectories[:, :, dim].flatten() for dim in range(state_dim)]
    box_plot = ax.boxplot(data_for_box, labels=state_names, patch_artist=True)
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_title('Distribution of State Variables')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # 2. Histograms
    ax = axes[1]
    for dim in range(state_dim):
        data = trajectories[:, :, dim].flatten()
        ax.hist(data, bins=50, alpha=0.6, label=state_names[dim], density=True)
    ax.set_title('Probability Density')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Range plot
    ax = axes[2]
    x_pos = np.arange(state_dim)
    ax.bar(x_pos, maxs - mins, bottom=mins, alpha=0.7, color='skyblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(state_names)
    ax.set_title('Value Ranges')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, (mean, std, min_val, max_val) in enumerate(zip(means, stds, mins, maxs)):
        ax.text(i, max_val + 0.1*(max_val - min_val), 
               f'μ={mean:.2f}\nσ={std:.2f}', 
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 4. Trajectory lengths (in state space)
    ax = axes[3]
    lengths = []
    for traj in trajectories:
        # Calculate cumulative distance along trajectory
        diffs = np.diff(traj, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        total_length = np.sum(distances)
        lengths.append(total_length)
    
    ax.hist(lengths, bins=20, alpha=0.7, color='orange')
    ax.set_title('Trajectory Lengths in State Space')
    ax.set_xlabel('Total Distance')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{system_type.title()} System - Statistical Properties', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / 'statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nStatistical Summary:")
    print(f"{'Variable':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    for i, name in enumerate(state_names):
        print(f"{name:<10} {means[i]:<10.3f} {stds[i]:<10.3f} {mins[i]:<10.3f} {maxs[i]:<10.3f}")


def ask_user_approval() -> bool:
    """Ask user if they want to continue with training."""
    print(f"\n{'='*60}")
    print("DATA INSPECTION COMPLETE")
    print(f"{'='*60}")
    print("Please check the generated plots to verify the trajectory data.")
    print("The plots show:")
    print("  - Phase space visualization (if ≤3D)")
    print("  - Time series for each dimension")
    print("  - Statistical properties")
    print()
    
    while True:
        response = input("Do you want to continue with autoencoder training? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("Training cancelled. You can modify the configuration and run again.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.") 
#!/usr/bin/env python3
"""Main orchestration script for the dynamics autoencoder workspace."""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.utils.visualization import plot_trajectory_data, ask_user_approval
from src.dynamics.simulator import generate_dataset, load_dataset
from src.training.state_trainer import StateAutoencoderTrainer
from src.training.state_evaluator import StateAutoencoderEvaluator


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description='Dynamics Autoencoder Workspace',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--skip-data-generation',
        action='store_true',
        help='Skip data generation and use existing dataset'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to existing dataset (if skipping data generation)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create output directory
    output_dir = config.create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Save configuration to output directory
    config.save(output_dir / 'config.yaml')
    
    start_time = time.time()
    
    try:
        # Step 1: Data Generation
        if args.skip_data_generation:
            if args.dataset_path:
                dataset_path = Path(args.dataset_path)
            else:
                # Look for existing dataset in output directory
                dataset_filename = f"{config['system_name']}_dataset.pkl"
                dataset_path = output_dir / 'data' / dataset_filename
            
            print(f"Loading existing dataset from {dataset_path}")
            if not dataset_path.exists():
                print(f"Error: Dataset file not found at {dataset_path}")
                return 1
            
            dataset = load_dataset(dataset_path)
        else:
            print("\n" + "="*60)
            print("STEP 1: GENERATING TRAJECTORY DATASET")
            print("="*60)
            
            dataset_path = generate_dataset(config)
            dataset = load_dataset(dataset_path)
        
        print(f"Dataset loaded: {dataset['num_successful']} trajectories")
        print(f"Trajectory shape: {dataset['trajectories'].shape}")
        
        # Step 1.5: Visualize trajectory data and ask for approval
        plot_trajectory_data(dataset, output_dir, max_trajectories=50)
        
        if not ask_user_approval():
            print("Exiting without training.")
            return 0
        
        # Step 2: Train State-Space Autoencoder
        print("\n" + "="*60)
        print("STEP 2: TRAINING STATE-SPACE AUTOENCODER")
        print("="*60)
        
        # Determine device
        device = args.device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create trainer
        trainer = StateAutoencoderTrainer(config, dataset, device=device)
        
        # Train model
        training_results = trainer.train()
        
        print(f"Training completed!")
        print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
        print(f"Test loss: {training_results['test_loss']:.6f}")
        
        # Step 3: Manifold Evaluation
        print("\n" + "="*60)
        print("STEP 3: MANIFOLD EVALUATION")
        print("="*60)
        
        # Create evaluator
        evaluator = StateAutoencoderEvaluator(config, dataset, trainer, output_dir)
        
        # Run evaluation
        evaluation_results = evaluator.evaluate_all()
        
        # Generate report
        if config.get('output.generate_report', True):
            report = evaluator.generate_report(evaluation_results)
            print("\nEvaluation report generated!")
        
        # Step 4: Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Configuration: {output_dir / 'config.yaml'}")
        print(f"- Dataset: {dataset_path}")
        print(f"- Best model: {training_results['best_model_path']}")
        print(f"- Evaluation results: {output_dir / 'evaluation_results.json'}")
        print(f"- Plots: {output_dir / 'plots'}")
        
        # Print key metrics
        print(f"\nKey Performance Metrics:")
        print(f"- Reconstruction MSE: {evaluation_results['reconstruction_error']['mse']:.6f}")
        print(f"- Mean Correlation: {evaluation_results['reconstruction_error']['mean_correlation']:.4f}")
        print(f"- Distance Correlation: {evaluation_results['manifold_quality']['distance_correlation']:.4f}")
        print(f"- Improvement over PCA: {evaluation_results['pca_comparison']['improvement_over_pca']:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
#!/usr/bin/env python3
"""Main orchestration script for the dynamics autoencoder workspace."""

import argparse
import sys
from pathlib import Path
import time
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.utils.visualization import plot_trajectory_data, ask_user_approval
from src.dynamics.simulator import generate_dataset, load_dataset
from src.training.state_trainer import create_trainer
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
        trainer = create_trainer(config, dataset, device=device)
        
        # Train model(s) - now always uses the unified trainer
        print("Running training pipeline...")
        training_results = trainer.train()
        print("Training completed!")
        
        # Check if we have multiple models (for comparison)
        num_models = len(training_results)
        if num_models > 1:
            print(f"Trained {num_models} models - comparison plots generated automatically")
        else:
            print(f"Trained {num_models} model")
        
        # Step 3: Manifold Evaluation (now runs for all models)
        print("\n" + "="*60)
        print("STEP 3: MANIFOLD EVALUATION")
        print("="*60)
        
        if num_models == 1:
            # Single model training - run evaluation normally
            model_name = list(training_results.keys())[0]
            if hasattr(trainer, 'model'):
                evaluator = StateAutoencoderEvaluator(config, dataset, trainer, output_dir)
                evaluation_results = evaluator.evaluate_all()
                if config.get('output.generate_report', True):
                    evaluator.generate_report(evaluation_results)
                    print("\nEvaluation report generated!")
            else:
                print("Warning: Trainer has no single model for evaluation")
        else:
            # Multi-model training - run evaluation for each model
            print(f"Running manifold evaluation for {num_models} models...")
            evaluate_all_models(training_results, config, dataset, output_dir)
            print(f"\nManifold evaluation completed for all models!")
        
        # Step 4: Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"- Configuration: {output_dir / 'config.yaml'}")
        print(f"- Dataset: {dataset_path}")
        
        if num_models == 1:
            # Single model training
            model_name = list(training_results.keys())[0]
            print(f"- Evaluation results: {output_dir / 'evaluation_results.json'}")
            print(f"- Plots: {output_dir / 'plots'}")
        else:
            # Multi-model training
            print(f"- Models: {', '.join(training_results.keys())}")
            print(f"- Comparison plots: {output_dir / 'comprehensive_prediction_loss_comparison.png'}")
            print(f"- Individual evaluations: {output_dir / 'evaluations'}")
        
        # Print key metrics
        if num_models == 1:
            print(f"\nKey Performance Metrics:")
            # Note: evaluation_results not available here for single model case
            print(f"- Training completed successfully")
        else:
            print(f"\nMulti-Model Training Results:")
            for model_name, results in training_results.items():
                if 'best_test_loss_components' in results:
                    pred_loss = results['best_test_loss_components'].get('prediction', 'N/A')
                    print(f"- {model_name}: Prediction Loss = {pred_loss:.6f}")
                elif isinstance(results, list) and len(results) > 0:
                    # Sweep results
                    for sweep_result in results:
                        if 'prediction_loss' in sweep_result:
                            d = sweep_result['latent_dim']
                            pred_loss = sweep_result['prediction_loss']
                            print(f"- {model_name} (d={d}): Prediction Loss = {pred_loss:.6f}")
            print(f"- Comparison plots saved to: {output_dir / 'comprehensive_prediction_loss_comparison.png'}")
        
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


def evaluate_all_models(training_results, config, dataset, output_dir):
    """Evaluate all models in a clean, modular way."""
    for model_name, results in training_results.items():
        print(f"\nEvaluating {model_name}...")
        model_output_dir = output_dir / 'evaluations' / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name.endswith('_sweep'):
            evaluate_sweep_model(model_name, results, config, dataset, model_output_dir)
        else:
            evaluate_single_model(model_name, config, dataset, model_output_dir)


def evaluate_sweep_model(model_name, sweep_results, config, dataset, model_output_dir):
    """Evaluate a model with latent dimension sweep."""
    for sweep_result in sweep_results:
        d = sweep_result['latent_dim']
        sweep_dir = model_output_dir / f'd{d}'
        sweep_dir.mkdir(exist_ok=True)
        
        # Create model and load weights
        model = create_model_for_evaluation(model_name, config, dataset, d)
        # Load the trained weights
        model_path = model_output_dir.parent.parent / 'models' / f'{model_name.replace("_sweep", "")}_d{d}_best_model.pth'
        if not load_model_weights(model, model_path, f"d={d}"):
            continue
        
        # Run evaluation
        run_model_evaluation(model, config, dataset, sweep_dir, f"d={d}")


def evaluate_single_model(model_name, config, dataset, model_output_dir):
    """Evaluate a single model."""
    model = create_model_for_evaluation(model_name, config, dataset)
    model_path = model_output_dir.parent.parent / 'models' / f'{model_name}_best_model.pth'
    
    if not load_model_weights(model, model_path):
        return
    
    run_model_evaluation(model, config, dataset, model_output_dir)


def create_model_for_evaluation(model_name, config, dataset, latent_dim=None):
    """Create a model instance for evaluation."""
    if model_name == 'state_mlp':
        from src.models.state_autoencoder import create_mlp_dynamics_model
        return create_mlp_dynamics_model(config.to_dict(), dataset['trajectories'].shape[2])
    else:
        from src.models.state_autoencoder import create_state_autoencoder
        if latent_dim:
            temp_config = config.to_dict()
            temp_config['autoencoder']['latent_dim'] = latent_dim
            return create_state_autoencoder(temp_config['autoencoder'], dataset['trajectories'].shape[2], 'prediction')
        else:
            return create_state_autoencoder(config['autoencoder'], dataset['trajectories'].shape[2], 'prediction')


def load_model_weights(model, model_path, label=""):
    """Load trained weights into a model."""
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print(f"  {label}: Model loaded from {model_path.name}")
        return True
    else:
        print(f"  {label}: Warning - model file not found at {model_path}")
        return False


def run_model_evaluation(model, config, dataset, output_dir, label=""):
    """Run evaluation for a single model."""
    try:
        from src.training.state_evaluator import ModelEvaluator
        evaluator = ModelEvaluator(config, dataset, model, output_dir)
        evaluation_results = evaluator.evaluate_all()
        print(f"  {label}: Evaluation completed")
        
        if config.get('output.generate_report', True):
            evaluator.generate_report(evaluation_results)
            print(f"  {label}: Report generated")
    except Exception as e:
        print(f"  {label}: Evaluation failed - {e}")
        # Remove args.verbose reference since it's not available here
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
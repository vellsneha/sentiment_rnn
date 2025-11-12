"""
Evaluation Module
Runs all experiments and generates results.
"""

import os
import pandas as pd
import itertools
from train import run_experiment
from utils import set_seed
import torch

# Set seeds for reproducibility
set_seed(42)


def run_all_experiments(device='cpu', epochs=5):
    """
    Run all experiment combinations systematically.
    
    Args:
        device: Device to train on
        epochs: Number of epochs for each experiment
        
    Returns:
        List of all results
    """
    # Define experiment configurations
    architectures = ['rnn', 'lstm', 'bilstm']
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['adam', 'sgd', 'rmsprop']
    sequence_lengths = [25, 50, 100]
    gradient_clipping_options = [False, True]
    
    all_results = []
    
    # We'll test each factor systematically:
    # 1. Test all architectures with baseline config
    # 2. Test all activations with best architecture
    # 3. Test all optimizers with best architecture and activation
    # 4. Test all sequence lengths
    # 5. Test gradient clipping
    
    print("="*80)
    print("RUNNING SYSTEMATIC EXPERIMENTS")
    print("="*80)
    
    # Baseline configuration
    baseline = {
        'model_type': 'lstm',
        'activation': 'relu',
        'optimizer_name': 'adam',
        'seq_length': 50,
        'use_gradient_clipping': True
    }
    
    # Experiment 1: Test all architectures
    print("\n" + "="*80)
    print("EXPERIMENT 1: Testing Architectures")
    print("="*80)
    for arch in architectures:
        config = baseline.copy()
        config['model_type'] = arch
        result = run_experiment(**config, epochs=epochs, device=device)
        all_results.append(result)
    
    # Experiment 2: Test all activations
    print("\n" + "="*80)
    print("EXPERIMENT 2: Testing Activation Functions")
    print("="*80)
    for activation in activations:
        config = baseline.copy()
        config['activation'] = activation
        result = run_experiment(**config, epochs=epochs, device=device)
        all_results.append(result)
    
    # Experiment 3: Test all optimizers
    print("\n" + "="*80)
    print("EXPERIMENT 3: Testing Optimizers")
    print("="*80)
    for optimizer in optimizers:
        config = baseline.copy()
        config['optimizer_name'] = optimizer
        result = run_experiment(**config, epochs=epochs, device=device)
        all_results.append(result)
    
    # Experiment 4: Test all sequence lengths
    print("\n" + "="*80)
    print("EXPERIMENT 4: Testing Sequence Lengths")
    print("="*80)
    for seq_len in sequence_lengths:
        config = baseline.copy()
        config['seq_length'] = seq_len
        result = run_experiment(**config, epochs=epochs, device=device)
        all_results.append(result)
    
    # Experiment 5: Test gradient clipping
    print("\n" + "="*80)
    print("EXPERIMENT 5: Testing Gradient Clipping")
    print("="*80)
    for use_clipping in gradient_clipping_options:
        config = baseline.copy()
        config['use_gradient_clipping'] = use_clipping
        result = run_experiment(**config, epochs=epochs, device=device)
        all_results.append(result)
    
    return all_results


def save_results(results, output_dir='results'):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df_data = []
    for result in results:
        row = {
            'Model': result['model'],
            'Activation': result['activation'],
            'Optimizer': result['optimizer'],
            'Seq_Length': result['seq_length'],
            'Grad_Clipping': 'Yes' if result['grad_clipping'] else 'No',
            'Accuracy': result['accuracy'],
            'F1_Score': result['f1_score'],
            'Avg_Epoch_Time_s': result['avg_epoch_time']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'metrics.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Find best configuration
    best_idx = df['Accuracy'].idxmax()
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    print(df.iloc[best_idx].to_string())
    
    return df


def generate_comparison_data(results):
    """
    Generate data for comparison plots.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary containing comparison data
    """
    comparison_data = {
        'architectures': [],
        'activations': [],
        'optimizers': [],
        'sequence_lengths': [],
        'gradient_clipping': [],
        'best_model_history': None,
        'worst_model_history': None
    }
    
    # Group results by category
    for result in results:
        if result['activation'] == 'relu' and result['optimizer'] == 'adam' and \
           result['seq_length'] == 50 and result['grad_clipping']:
            comparison_data['architectures'].append({
                'model': result['model'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score']
            })
        
        if result['model'] == 'lstm' and result['optimizer'] == 'adam' and \
           result['seq_length'] == 50 and result['grad_clipping']:
            comparison_data['activations'].append({
                'activation': result['activation'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score']
            })
        
        if result['model'] == 'lstm' and result['activation'] == 'relu' and \
           result['seq_length'] == 50 and result['grad_clipping']:
            comparison_data['optimizers'].append({
                'optimizer': result['optimizer'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score']
            })
        
        if result['model'] == 'lstm' and result['activation'] == 'relu' and \
           result['optimizer'] == 'adam' and result['grad_clipping']:
            comparison_data['sequence_lengths'].append({
                'seq_length': result['seq_length'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score']
            })
        
        if result['model'] == 'lstm' and result['activation'] == 'relu' and \
           result['optimizer'] == 'adam' and result['seq_length'] == 50:
            comparison_data['gradient_clipping'].append({
                'grad_clipping': result['grad_clipping'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score']
            })
    
    # Find best and worst models
    accuracies = [r['accuracy'] for r in results]
    best_idx = accuracies.index(max(accuracies))
    worst_idx = accuracies.index(min(accuracies))
    
    comparison_data['best_model_history'] = results[best_idx]['history']
    comparison_data['worst_model_history'] = results[worst_idx]['history']
    comparison_data['best_model_config'] = {
        'model': results[best_idx]['model'],
        'activation': results[best_idx]['activation'],
        'optimizer': results[best_idx]['optimizer'],
        'seq_length': results[best_idx]['seq_length']
    }
    comparison_data['worst_model_config'] = {
        'model': results[worst_idx]['model'],
        'activation': results[worst_idx]['activation'],
        'optimizer': results[worst_idx]['optimizer'],
        'seq_length': results[worst_idx]['seq_length']
    }
    
    return comparison_data


if __name__ == "__main__":
    print("Starting experiments...")
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run all experiments
    results = run_all_experiments(device=device, epochs=5)
    
    # Save results
    df = save_results(results)
    
    # Generate comparison data for plotting
    comparison_data = generate_comparison_data(results)
    
    # Save comparison data for plotting
    import pickle
    with open('results/comparison_data.pkl', 'wb') as f:
        pickle.dump(comparison_data, f)
    print("\nComparison data saved to results/comparison_data.pkl")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
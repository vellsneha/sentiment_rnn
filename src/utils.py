"""
Utility Functions
Helper functions for reproducibility and common operations.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_training_history(history, title, save_path=None):
    """
    Plot training and validation loss/accuracy.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['test_loss'], label='Test Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['test_acc'], label='Test Accuracy', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_sequence_length_comparison(comparison_data, save_path=None):
    """
    Plot accuracy and F1-score vs sequence length.
    
    Args:
        comparison_data: Dictionary containing comparison data
        save_path: Path to save the plot
    """
    seq_data = comparison_data['sequence_lengths']
    
    if not seq_data:
        print("No sequence length data available")
        return
    
    seq_lengths = [d['seq_length'] for d in seq_data]
    accuracies = [d['accuracy'] for d in seq_data]
    f1_scores = [d['f1_score'] for d in seq_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(seq_lengths, accuracies, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Sequence Length')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(seq_lengths)
    
    # Plot F1-score
    axes[1].plot(seq_lengths, f1_scores, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score vs Sequence Length')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(seq_lengths)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_best_worst_comparison(comparison_data, save_path=None):
    """
    Plot training loss comparison between best and worst models.
    
    Args:
        comparison_data: Dictionary containing comparison data
        save_path: Path to save the plot
    """
    best_history = comparison_data['best_model_history']
    worst_history = comparison_data['worst_model_history']
    best_config = comparison_data['best_model_config']
    worst_config = comparison_data['worst_model_config']
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(best_history['train_loss']) + 1)
    
    plt.plot(epochs, best_history['train_loss'], 
             label=f"Best Model Train ({best_config['model']}, {best_config['activation']})", 
             marker='o', linewidth=2)
    plt.plot(epochs, worst_history['train_loss'], 
             label=f"Worst Model Train ({worst_config['model']}, {worst_config['activation']})", 
             marker='s', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss: Best vs Worst Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_architecture_comparison(comparison_data, save_path=None):
    """
    Plot comparison of different architectures.
    
    Args:
        comparison_data: Dictionary containing comparison data
        save_path: Path to save the plot
    """
    arch_data = comparison_data['architectures']
    
    if not arch_data:
        print("No architecture data available")
        return
    
    models = [d['model'].upper() for d in arch_data]
    accuracies = [d['accuracy'] for d in arch_data]
    f1_scores = [d['f1_score'] for d in arch_data]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Different Architectures')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def create_all_plots(comparison_data, output_dir='results/plots'):
    """
    Create all visualization plots.
    
    Args:
        comparison_data: Dictionary containing comparison data
        output_dir: Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # Plot 1: Sequence length comparison
    plot_sequence_length_comparison(
        comparison_data, 
        save_path=os.path.join(output_dir, 'sequence_length_comparison.png')
    )
    
    # Plot 2: Best vs Worst model
    plot_best_worst_comparison(
        comparison_data,
        save_path=os.path.join(output_dir, 'best_worst_comparison.png')
    )
    
    # Plot 3: Architecture comparison
    plot_architecture_comparison(
        comparison_data,
        save_path=os.path.join(output_dir, 'architecture_comparison.png')
    )
    
    # Plot 4: Training history for best model
    plot_training_history(
        comparison_data['best_model_history'],
        f"Best Model: {comparison_data['best_model_config']['model'].upper()}",
        save_path=os.path.join(output_dir, 'best_model_history.png')
    )
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    # Load comparison data and create plots
    if os.path.exists('results/comparison_data.pkl'):
        with open('results/comparison_data.pkl', 'rb') as f:
            comparison_data = pickle.load(f)
        
        create_all_plots(comparison_data)
    else:
        print("No comparison data found. Run evaluate.py first.")

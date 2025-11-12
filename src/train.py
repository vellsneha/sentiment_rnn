"""
Training Module
Handles model training with various configurations.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import get_model
from utils import set_seed

# Set seeds for reproducibility
set_seed(42)


def get_optimizer(model, optimizer_name, learning_rate=0.001):
    """
    Get optimizer based on name.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, clip_gradient=None):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        clip_gradient: Gradient clipping value (None for no clipping)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for batch_idx, (text, labels) in enumerate(dataloader):
        # Move data to device
        text = text.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(text)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if specified
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, predictions, true_labels)
    """
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, labels in dataloader:
            # Move data to device
            text = text.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            # Forward pass
            predictions = model(text)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            # Store predictions and labels
            preds = torch.sigmoid(predictions).round()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return epoch_loss / len(dataloader), np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, test_loader, optimizer, criterion, 
                device, epochs=10, clip_gradient=None, verbose=True):
    """
    Train model for multiple epochs.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs
        clip_gradient: Gradient clipping value
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                device, clip_gradient)
        
        # Evaluate on test set
        test_loss, test_preds, test_labels = evaluate(model, test_loader, 
                                                       criterion, device)
        
        # Calculate accuracy
        test_acc = np.mean(test_preds == test_labels)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Store history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Test Loss: {test_loss:.4f}')
            print(f'  Test Accuracy: {test_acc:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
    
    return history


def run_experiment(model_type, activation, optimizer_name, seq_length, 
                   use_gradient_clipping, vocab_size=10000, batch_size=32, 
                   epochs=5, device='cpu'):
    """
    Run a single experiment with given configuration.
    
    Args:
        model_type: Type of model ('rnn', 'lstm', 'bilstm')
        activation: Activation function
        optimizer_name: Optimizer name
        seq_length: Sequence length
        use_gradient_clipping: Whether to use gradient clipping
        vocab_size: Vocabulary size
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to train on
        
    Returns:
        Dictionary containing results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment:")
    print(f"  Model: {model_type}")
    print(f"  Activation: {activation}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Sequence Length: {seq_length}")
    print(f"  Gradient Clipping: {use_gradient_clipping}")
    print(f"{'='*60}")
    
    # Get the project root directory (parent of src directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Load data
    print("Loading data...")
    train_sequences = np.load(os.path.join(data_dir, f'train_sequences_{seq_length}.npy'))
    train_labels = np.load(os.path.join(data_dir, f'train_labels_{seq_length}.npy'))
    test_sequences = np.load(os.path.join(data_dir, f'test_sequences_{seq_length}.npy'))
    test_labels = np.load(os.path.join(data_dir, f'test_labels_{seq_length}.npy'))
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.LongTensor(train_sequences), 
                                  torch.LongTensor(train_labels))
    test_dataset = TensorDataset(torch.LongTensor(test_sequences), 
                                 torch.LongTensor(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    print("Creating model...")
    model = get_model(model_type, vocab_size, embedding_dim=100, hidden_dim=64,
                     output_dim=1, n_layers=2, dropout=0.3, activation=activation)
    model = model.to(device)
    
    # Get optimizer and criterion
    optimizer = get_optimizer(model, optimizer_name)
    criterion = nn.BCEWithLogitsLoss()
    
    # Set gradient clipping
    clip_gradient = 1.0 if use_gradient_clipping else None
    
    # Train model
    print("Training model...")
    history = train_model(model, train_loader, test_loader, optimizer, criterion,
                         device, epochs=epochs, clip_gradient=clip_gradient)
    
    # Get final predictions for detailed metrics
    _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='macro')
    avg_epoch_time = np.mean(history['epoch_times'])
    
    results = {
        'model': model_type,
        'activation': activation,
        'optimizer': optimizer_name,
        'seq_length': seq_length,
        'grad_clipping': use_gradient_clipping,
        'accuracy': accuracy,
        'f1_score': f1,
        'avg_epoch_time': avg_epoch_time,
        'history': history
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Avg Epoch Time: {avg_epoch_time:.2f}s")
    
    return results


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Run a single experiment
    results = run_experiment(
        model_type='lstm',
        activation='relu',
        optimizer_name='adam',
        seq_length=50,
        use_gradient_clipping=True,
        epochs=5,
        device=device
    )

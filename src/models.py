"""
Model Architecture Module
Defines RNN, LSTM, and Bidirectional LSTM models for sentiment classification.
"""

import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    """
    Basic RNN model for sentiment classification.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, activation='relu'):
        """
        Initialize RNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Number of hidden units
            output_dim: Number of output classes (1 for binary)
            n_layers: Number of RNN layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(SentimentRNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, text):
        """
        Forward pass.
        
        Args:
            text: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output predictions
        """
        # Embedding: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(text)
        
        # Apply activation after embedding
        embedded = self.activation(embedded)
        
        # RNN output: (batch_size, seq_length, hidden_dim)
        output, hidden = self.rnn(embedded)
        
        # Take the last hidden state
        # hidden: (n_layers, batch_size, hidden_dim)
        hidden = hidden[-1, :, :]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layer: (batch_size, output_dim)
        output = self.fc(hidden)
        
        return output


class SentimentLSTM(nn.Module):
    """
    LSTM model for sentiment classification.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, activation='relu'):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Number of hidden units
            output_dim: Number of output classes (1 for binary)
            n_layers: Number of LSTM layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(SentimentLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, text):
        """
        Forward pass.
        
        Args:
            text: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output predictions
        """
        # Embedding: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(text)
        
        # Apply activation after embedding
        embedded = self.activation(embedded)
        
        # LSTM output: (batch_size, seq_length, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take the last hidden state
        # hidden: (n_layers, batch_size, hidden_dim)
        hidden = hidden[-1, :, :]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layer: (batch_size, output_dim)
        output = self.fc(hidden)
        
        return output


class SentimentBiLSTM(nn.Module):
    """
    Bidirectional LSTM model for sentiment classification.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, activation='relu'):
        """
        Initialize Bidirectional LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Number of hidden units
            output_dim: Number of output classes (1 for binary)
            n_layers: Number of LSTM layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(SentimentBiLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
                           bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer (hidden_dim * 2 for bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, text):
        """
        Forward pass.
        
        Args:
            text: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output predictions
        """
        # Embedding: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(text)
        
        # Apply activation after embedding
        embedded = self.activation(embedded)
        
        # LSTM output: (batch_size, seq_length, hidden_dim * 2)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        # hidden: (n_layers * 2, batch_size, hidden_dim)
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layer: (batch_size, output_dim)
        output = self.fc(hidden)
        
        return output


def get_model(model_type, vocab_size, embedding_dim=100, hidden_dim=64, 
              output_dim=1, n_layers=2, dropout=0.3, activation='relu'):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('rnn', 'lstm', 'bilstm')
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_dim: Number of hidden units
        output_dim: Number of output classes
        n_layers: Number of layers
        dropout: Dropout rate
        activation: Activation function
        
    Returns:
        Model instance
    """
    if model_type.lower() == 'rnn':
        return SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim, 
                           n_layers, dropout, activation)
    elif model_type.lower() == 'lstm':
        return SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, 
                            n_layers, dropout, activation)
    elif model_type.lower() == 'bilstm':
        return SentimentBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, 
                              n_layers, dropout, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

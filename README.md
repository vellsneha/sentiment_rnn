# RNN Sentiment Classification on IMDb Dataset

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures for sentiment classification on the IMDb Movie Review Dataset.

## Project Structure
```
├── data/                          # Data directory (created automatically)
│   ├── train_sequences_*.npy     # Preprocessed training sequences
│   ├── test_sequences_*.npy      # Preprocessed test sequences
│   ├── train_labels_*.npy        # Training labels
│   ├── test_labels_*.npy         # Test labels
│   └── tokenizer.pkl             # Saved tokenizer
├── src/
│   ├── preprocess.py             # Data preprocessing
│   ├── models.py                 # RNN model architectures
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation and experiments
│   └── utils.py                  # Utility functions
├── results/                       # Results directory (created automatically)
│   ├── metrics.csv               # Performance metrics
│   ├── comparison_data.pkl       # Data for plotting
│   └── plots/                    # Generated plots
├── report.pdf                    # Project report
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone or download this repository:
```bash
git clone https://github.com/vellsneha/sentiment_rnn.git
cd sentiment_rnn
```

2. Create and activate the virtual environment
```bash
python3.12 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Preprocess Data

Navigate to the src directory and run preprocessing:
```bash
python src/preprocess.py
```

This will:
- Download the IMDb dataset (50,000 reviews)
- Clean and tokenize the text
- Create sequences for lengths 25, 50, and 100
- Save preprocessed data to `data/` directory

**Expected Output:**
- Dataset statistics (average review length, vocabulary size)
- Preprocessed sequences saved as `.npy` files

**Runtime:** ~5-10 minutes on CPU

### Step 2: Run All Experiments
```bash
python src/evaluate.py
```

This will:
- Test all architectures (RNN, LSTM, Bidirectional LSTM)
- Test all activation functions (Sigmoid, ReLU, Tanh)
- Test all optimizers (Adam, SGD, RMSProp)
- Test all sequence lengths (25, 50, 100)
- Test gradient clipping (Yes/No)
- Save results to `../results/metrics.csv`

**Expected Runtime:** ~3-5 hours on CPU (5 epochs per configuration)

**To run with GPU:**
The code automatically detects and uses CUDA if available.

### Step 3: Generate Plots
```bash
python src/utils.py
```

This creates visualization plots in `../results/plots/`:
- Accuracy/F1 vs Sequence Length
- Training Loss: Best vs Worst Model
- Architecture Comparison
- Best Model Training History

**Runtime:** ~1-2 minutes

### Alternative: Run Single Experiment

To test a specific configuration, edit `train.py` and run:
```bash
python train.py
```

Modify the configuration at the bottom of the file:
```python
results = run_experiment(
    model_type='lstm',           # 'rnn', 'lstm', or 'bilstm'
    activation='relu',            # 'sigmoid', 'relu', or 'tanh'
    optimizer_name='adam',        # 'adam', 'sgd', or 'rmsprop'
    seq_length=50,               # 25, 50, or 100
    use_gradient_clipping=True,  # True or False
    epochs=5,
    device=device
)
```

**Runtime:** ~10-20 minutes per experiment on CPU

## Model Configurations

### Fixed Hyperparameters
- **Embedding dimension:** 100
- **Hidden layer size:** 64
- **Number of hidden layers:** 2
- **Dropout rate:** 0.3
- **Batch size:** 32
- **Loss function:** Binary Cross-Entropy
- **Output activation:** Sigmoid

### Variable Parameters Tested
1. **Architecture:** RNN, LSTM, Bidirectional LSTM
2. **Activation Function:** Sigmoid, ReLU, Tanh
3. **Optimizer:** Adam, SGD, RMSProp
4. **Sequence Length:** 25, 50, 100 words
5. **Gradient Clipping:** Yes (value=1.0) or No

## Results

After running experiments, results are saved in `results/metrics.csv` with columns:
- Model: Architecture type
- Activation: Activation function
- Optimizer: Optimization algorithm
- Seq_Length: Sequence length
- Grad_Clipping: Whether gradient clipping was used
- Accuracy: Test accuracy
- F1_Score: Macro F1-score
- Avg_Epoch_Time_s: Average training time per epoch

## Hardware Requirements

### Minimum (CPU only)
- RAM: 8GB minimum
- Storage: 2GB
- Expected runtime: 3-5 hours for all experiments

### Recommended (with GPU)
- RAM: 8GB
- GPU: Any CUDA-compatible GPU
- Expected runtime: 30-60 minutes for all experiments

The code automatically detects and uses GPU if available.

## Reproducibility

All random seeds are fixed to 42 for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

## Dataset

**Dataset:** IMDb Movie Review Dataset
- 50,000 movie reviews
- Binary sentiment classification (positive/negative)
- 25,000 training samples, 25,000 test samples
- Predefined train/test split

## File Descriptions

### src/preprocess.py
- Loads IMDb dataset
- Cleans text (lowercase, remove punctuation)
- Tokenizes and creates sequences
- Pads/truncates to fixed lengths

### src/models.py
- Implements three architectures:
  - `SentimentRNN`: Basic RNN
  - `SentimentLSTM`: LSTM
  - `SentimentBiLSTM`: Bidirectional LSTM
- Factory function `get_model()` for easy instantiation

### src/train.py
- Training loop implementation
- Evaluation functions
- `run_experiment()`: Runs single configuration
- Supports gradient clipping

### src/evaluate.py
- Runs all experiment combinations systematically
- Saves results to CSV
- Generates comparison data for plotting

### src/utils.py
- Seed setting for reproducibility
- Plotting functions
- Visualization generation

## Troubleshooting

**Issue:** Out of memory error
- **Solution:** Reduce batch size in `train.py` (line with `batch_size=32`)

**Issue:** Slow training
- **Solution:** Use GPU if available, or reduce number of epochs

**Issue:** Import errors
- **Solution:** Make sure the correct version of python and packages are installed

**Issue:** Dataset download fails
- **Solution:** Check internet connection; Keras automatically downloads IMDb dataset

## Dependencies

All dependencies are listed in `requirements.txt`:
- torch>=1.9.0
- numpy>=1.19.5
- pandas>=1.3.0
- matplotlib>=3.3.4
- seaborn>=0.11.1
- scikit-learn>=0.24.2
- tensorflow>=2.6.0


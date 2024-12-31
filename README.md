# Transformer from Scratch: Language Translation (English to Italian)

This project implements a transformer model from scratch for English-to-Italian language translation, inspired by Umar Jamil's tutorial video on transformers. The goal is to understand transformers by building and training one without relying on pre-built libraries.

## Learning Source

This project is based on Umar Jamil's tutorial: [Building Transformer from Scratch](https://www.youtube.com/watch?v=ISNdQcPhsts&t=9601s).

## Project Structure

- `config.py`: Contains all configuration parameters for the model, training, and dataset setup.
- `dataset.py`: Includes the `BilingualDataset` class for processing English-Italian sentence pairs, creating tokenized inputs, and applying necessary masking.
- `model.py`: Defines the transformer model, including encoder, decoder, multi-head attention, and feed-forward layers.
- `train.py`: Handles the training process, including data loading, model training, and validation.
- `My_notes.pdf`: Notes explaining the concepts of transformers, including encoder-decoder structure, attention mechanisms, and training vs. inference processes.

---


## Components

### 1. **Configuration (`config.py`)**
This file defines:
- Model hyperparameters (`d_model`, `n_heads`, `num_layers`, etc.)
- Training parameters (`batch_size`, `learning_rate`, `num_epochs`, etc.)
- Dataset information (`source language`, `target language`, etc.)

### 2. **Dataset Preparation (`dataset.py`)**
The `BilingualDataset` class:
- Loads the dataset (English-Italian translation pairs) using Hugging Face's `datasets` library.
- Tokenizes sentences into numerical IDs.
- Adds special tokens ([SOS], [EOS]) and applies padding for uniform sequence lengths.
- Creates encoder and decoder masks for attention mechanisms.

### 3. **Model Architecture (`model.py`)**
The transformer is implemented from scratch and includes:
- **Input Embeddings**: Convert tokenized inputs into dense vectors scaled by √(d_model).
- **Positional Encoding**: Adds position-based information to embeddings.
- **Multi-Head Attention**: Computes self-attention across multiple attention heads.
- **Feed-Forward Networks**: Applies linear layers with ReLU activation for dimensional transformations.
- **Residual Connections & Layer Normalization**: Stabilizes and accelerates training.
- **Encoder**: Encodes the source sequence into contextualized embeddings.
- **Decoder**: Decodes the target sequence while attending to encoder outputs.

### 4. **Training (`train.py`)**
- **Training Mode**:
  - Teacher forcing is used, where the decoder's input is the ground truth sequence shifted by one token.
  - Cross-entropy loss is computed between predictions and actual tokens.
- **Validation Mode**:
  - Greedy decoding generates translations by predicting tokens sequentially without access to ground truth.
- **Optimization**:
  - Adam optimizer with learning rate scheduling is used.

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- Required Python libraries:
  ```bash
  pip install torch torchtext datasets tokenizers tqdm

Steps to Run
	1.	Clone the repository.
	2.	Set up your environment and install dependencies.
	3.	Run the training script:

python train.py

Example Output

During training, the model generates translations for validation examples. A typical output includes:
	•	Source Sentence
	•	Target Sentence (Ground Truth)
	•	Model Prediction


Feel free to contribute or suggest improvements!

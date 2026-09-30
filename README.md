# Toxix

Neural network models for predicting therapy outcomes with interaction effects between diagnostics and treatments.

## Setup

```bash
pip install -r requirements.txt
```

### Dependency installation time

Installing dependencies typically takes **1–3 minutes** on a standard machine with a working Python environment.
If PyTorch is not already installed, download time may increase slightly depending on network speed.

## Requirements
- Python 3.9.6

## Usage

```bash
python Main.py
```

## Configuration

Edit `settings.py` to configure:

* `location`: Set to `'M'` for mock/simulated data
* `mode`: Output folder name (e.g., `'mock'`)
* `model_type`: `'interaction'`
* `device`: `'cpu'` or `'cuda:0'`
* `n_splits`: Number of cross-validation splits
* `n_epochs`: Training epochs per split

## Expected Output and Results

Running the demo (with mock data) produces the following outputs:

* **Model performance metrics** for each cross-validation split (e.g., training/validation loss and prediction metrics, depending on task configuration).
* **Saved model checkpoints** for each fold.
* **Layer-wise Relevance Propagation (LRP) scores** providing interpretability of the trained model.

### LRP Scores

LRP scores quantify the contribution of each input component to the model’s predictions, including:

* Individual diagnostic features
* Individual treatment variables
* Interaction effects between diagnostics and treatments

These scores are intended to highlight which features and interactions most strongly influence predicted therapy outcomes. When using mock or simulated data, relevance patterns should be stable and interpretable, but they are **not clinically meaningful**.

## Expected Runtime

* **End-to-end demo runtime:** typically **under 10 minutes** on CPU with default settings and mock data.
* Runtime depends primarily on:

  * Number of cross-validation splits (`n_splits`)
  * Number of training epochs (`n_epochs`)
  * Hardware (`cpu` vs `cuda`)

Using a GPU or reducing epochs/splits can further reduce runtime.

## Project Structure

* `Main.py` – Main training loop
* `NN.py` – Neural network model definitions
* `training.py` – Training and evaluation functions
* `data.py` – Data loading and dataset classes
* `LRP.py` – Layer-wise Relevance Propagation for interpretability
* `settings.py` – Configuration parameters

## Time Summary

* **Dependency installation:** ~1–3 minutes
* **Demo execution (mock data):** typically < 10 minutes

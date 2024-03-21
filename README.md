# Project: Incident Classification 

This project aims to classify incidents using different machine learning models, including Feedforward Neural Networks, LSTM, and Transformer models (BERT and GPT-2). The project is implemented in Python and uses libraries such as PyTorch, Transformers, and Matplotlib.

## Project Structure

The project is divided into three main tasks, each implemented in a separate Jupyter notebook:

1. `feedforward.ipynb`: This notebook implements a Feedforward Neural Network for incident classification. The model is trained and evaluated on a dataset of incidents, and the results are visualized using Matplotlib.

2. `lstm.ipynb`: This notebook implements a LSTM model for incident classification. The model is trained and evaluated on the same dataset of incidents.

3. `transformers.ipynb`: This notebook implements two Transformer models (BERT and GPT-2) for incident classification. The models are trained and evaluated on the same dataset of incidents, and the results are compared with the Feedforward Neural Network and LSTM models.

## Setup

To run the notebooks, you need to have Python installed on your system. The project also requires the following Python libraries:

- PyTorch
- Transformers
- Matplotlib
- Pandas
- Numpy
- Spacy

You can install these libraries using pip:

```bash
pip install torch transformers matplotlib pandas numpy spacy
```

## Usage

To run a notebook, open it in Jupyter and execute the cells in order. Each notebook includes detailed comments explaining each step of the process.
You can have look on the dataset used in the project [here](https://drive.google.com/drive/folders/12CBUYGICyc40DVGil6QmcKp8YAUNr_t9?usp=sharing).

## Results

The project includes a detailed analysis of the results obtained by each model. The analysis includes accuracy, precision, recall, and F1 score metrics, as well as loss curves for the training and validation sets. The results show that the Transformer models (BERT and GPT-2) outperform the Feedforward Neural Network and LSTM models in terms of accuracy and F1 score.

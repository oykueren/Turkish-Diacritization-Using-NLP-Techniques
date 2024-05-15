# Turkish Diacritization Using NLP Techniques

This project aims to perform diacritization for the Turkish language using various natural language processing (NLP) techniques. The main goal is to predict the correct diacritics for Turkish characters in a given text.

## Project Structure

- **final.ipynb**: Contains the main code for data preprocessing, model training, and evaluation.
- **metrics.ipynb**: Contains the code for calculating and evaluating different performance metrics for the diacritization model.

## Dataset

The dataset used in this project includes Turkish sentences with and without diacritics. The dataset is split into training and testing sets:
- **train.csv**: Training data
- **test.csv**: Testing data

## Preprocessing

Preprocessing steps include:
1. Converting Turkish characters to their ASCII equivalents.
2. Lowercasing characters.
3. Mapping each character to a unique integer.
4. Padding sequences to a maximum length of 200.

## Model

The model is implemented using PyTorch and includes:
- Transformer-based embeddings to provide rich contextual representations.
- An LSTM layer to capture sequential dependencies.
- A CRF layer to model the sequence labeling task.

## Training

The training process involves:
1. Loading the data and converting it to the required format.
2. Defining the model architecture.
3. Training the model on the training data.
4. Evaluating the model on the testing data.

## Evaluation Metrics

Several metrics are calculated to evaluate the performance of the diacritization model, including:
- Word-level accuracy
- Diacritization Error Rate (DER)
- Precision
- Recall
- F1 Score

## Results

The results of the model are saved and can be viewed in the `metrics.ipynb` notebook. Key metrics are printed and visualized to understand the model's performance.

## Usage

To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/turkish-diacritization.git


2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt

3. Run the Jupyter notebooks to preprocess the data, train the model, and evaluate the performance:
jupyter notebook final.ipynb
jupyter notebook metrics.ipynb

# Dependencies
The project requires the following libraries:

- PyTorch
- Transformers
- zemberek-python
- pandas
- numpy
- unidecode
- Install the dependencies using the requirements.txt file provided in the repository.

## Authors
- Öykü Eren
- Mustafa Bayrak

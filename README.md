# LSTM Text Classification with TensorFlow

This project implements a text classification model using TensorFlow's Keras API. The model employs an LSTM-based architecture to analyze sentiment in movie reviews from the IMDB dataset.

## Dataset

The IMDB movie review dataset is used for training and testing. Each review is labeled with either positive (1) or negative (0) sentiment.

## Key Features

1. **Data Preprocessing**:
   - The dataset is shuffled and batched for efficient processing.
   - Prefetching is used to optimize input pipeline performance.

2. **Text Vectorization**:
   - A `TextVectorization` layer is used to tokenize and encode the text data.
   - The vocabulary size is limited to 1000 tokens for simplicity.

3. **Model Architecture**:
   - **Embedding Layer**: Maps words into dense vectors of fixed size.
   - **Bidirectional LSTM Layers**: Processes the text data in both forward and backward directions for better context understanding.
   - **Dense Layers**: Fully connected layers for classification.

4. **Model Compilation**:
   - Loss Function: Binary cross-entropy (suitable for binary classification tasks).
   - Optimizer: Adam optimizer with a learning rate of 1e-4.
   - Metrics: Accuracy.

5. **Training and Validation**:
   - The model is trained for 3 epochs with validation data.
   - Validation steps are set to 30 for monitoring performance during training.

## Code Structure

### Files
- **`592ML-tf-keras-LSTM.ipynb`**: Main notebook containing the code and outputs.

### Main Components
1. **Data Loading and Exploration**:
   - Loads the IMDB dataset using TensorFlow Datasets (TFDS).
   - Explores text samples and labels.

2. **Text Vectorization**:
   - Adapts the vectorizer to the training data.
   - Encodes sample text into integer sequences.

3. **Model Definition**:
   - Constructs a sequential model with embedding, LSTM, and dense layers.

4. **Model Training**:
   - Trains the model on the prepared dataset.
   - Validates the performance during training.

5. **Inference**:
   - Demonstrates predictions on custom sample texts.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd lstm-text-classification
   ```

3. Open the notebook in Jupyter or Google Colab:
   ```bash
   jupyter notebook 592ML-tf-keras-LSTM.ipynb
   ```

4. Run all cells to train the model and perform predictions.

## Example Predictions

### Input: Positive Review
```
The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.
```
### Output:
```
Positive Sentiment
```

### Input: Negative Review
```
The movie was not good. The animation and the graphics were terrible. I would not recommend this movie.
```
### Output:
```
Negative Sentiment
```

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow
  - TensorFlow Datasets
  - NumPy

## Future Enhancements

- Implement padding and truncation for sequence length consistency.
- Fine-tune hyperparameters for improved accuracy.
- Add a web interface to input reviews and display sentiment predictions.
---
Happy Coding!

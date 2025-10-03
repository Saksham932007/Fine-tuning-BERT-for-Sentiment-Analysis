# Fine-Tuning BERT for Sentiment Analysis 🎬

This repository contains the code for fine-tuning a pre-trained BERT model for sentiment analysis on the IMDb movie review dataset. The goal is to classify movie reviews as either positive or negative.

-----

## 📋 Table of Contents

  * [Project Overview](https://www.google.com/search?q=%23project-overview)
  * [Dataset](https://www.google.com/search?q=%23dataset)
  * [Methodology](https://www.google.com/search?q=%23methodology)
  * [Results](https://www.google.com/search?q=%23results)
  * [How to Run](https://www.google.com/search?q=%23how-to-run)
  * [Dependencies](https://www.google.com/search?q=%23dependencies)

-----

## Project Overview

This project demonstrates the process of using transfer learning to adapt a large language model, **BERT (Bidirectional Encoder Representations from Transformers)**, for a specific natural language processing task. We fine-tune the `bert-base-uncased` model on the IMDb dataset to perform binary sentiment classification.

-----

## Dataset

We use the **IMDb Large Movie Review Dataset**, which is available through the Hugging Face `datasets` library.

  * **Description**: This dataset consists of 50,000 movie reviews, split evenly into 25,000 for training and 25,000 for testing. The sentiment is binary, labeled as `0` for negative and `1` for positive. An additional 50,000 unlabeled reviews are also included for unsupervised learning, though they are not used in this specific project.
  * **Splits**:
      * **Train**: 25,000 reviews
      * **Test**: 25,000 reviews
      * **Unsupervised**: 50,000 reviews

For our training process, the initial training set of 25,000 reviews was further split into a training subset (20,000 reviews) and a validation subset (5,000 reviews).

-----

## Methodology

The process involves several key steps, from data preparation to model training and evaluation.

### 1\. Data Loading and Preprocessing

The IMDb dataset is loaded, and a `BertTokenizer` is used to convert the raw text of the movie reviews into a format suitable for the BERT model.

The preprocessing function `tokenize_function` performs the following actions:

  * Tokenizes the input text.
  * Pads shorter sequences to a maximum length.
  * Truncates longer sequences to ensure uniform input size.

<!-- end list -->

```python
from datasets import load_dataset
from transformers import BertTokenizer

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

### 2\. Model Configuration

We use the `BertForSequenceClassification` model from the Hugging Face `transformers` library. The model is initialized with weights from the `bert-base-uncased` pre-trained model and configured for a binary classification task (`num_labels=2`).

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

### 3\. Training

The model is trained using the `Trainer` API, which simplifies the training loop. The training arguments are configured as follows:

  * **Epochs**: 3
  * **Batch Size**: 8
  * **Learning Rate**: $2 \times 10^{-5}$
  * **Evaluation Strategy**: Performed at the end of each epoch.

<!-- end list -->

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()
```

-----

## Results

After three epochs of training, the model's performance on the validation set was evaluated.

| Epoch | Training Loss | Validation Loss |
| :---: | :-----------: | :-------------: |
|   1   |    0.2492     |     0.2436      |
|   2   |    0.1547     |     0.2905      |
|   3   |    0.0820     |     0.3401      |

The final evaluation on the validation set yielded an **evaluation loss of 0.3401**. This indicates that the model has learned to effectively classify the sentiment of movie reviews.

-----

## How to Run

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    (Or install them directly as shown below).

3.  **Run the Jupyter Notebook**:
    Open and run the `Untitled14.ipynb` notebook in a Jupyter environment or Google Colab.

-----

## Dependencies

The project relies on the following major Python libraries:

  * `transformers`: For accessing pre-trained models like BERT and the `Trainer` API.
  * `torch`: The deep learning framework used for model training.
  * `datasets`: For easily loading and preprocessing the IMDb dataset.

You can install these packages using pip:

```bash
pip install transformers torch datasets
```

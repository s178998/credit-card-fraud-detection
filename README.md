# Credit Card Fraud Detection Using PyTorch

## 1. Project Overview

This project explores data preprocessing, class imbalance handling, and neural network architecture development for credit card fraud detection using the European Credit Card Fraud Detection dataset.

The objective is to prepare transaction data for a machine learning workflow and implement a feedforward neural network using PyTorch to distinguish between legitimate and fraudulent transactions.

Because fraudulent transactions represent a very small percentage of the dataset, particular attention is given to data quality, feature preprocessing, and class imbalance.

The current implementation focuses on data preparation, resampling techniques, and neural network architecture. Model training and evaluation are planned as future development.

## 2. Dataset Description

The project uses the [European Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

The original dataset contains:

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features
- One target column (`Class`)

The target variable identifies each transaction:

| Class | Description |
|---|---|
| 0 | Legitimate transaction |
| 1 | Fraudulent transaction |

The features include anonymized PCA-transformed variables (`V1` through `V28`), transaction time (`Time`), and transaction amount (`Amount`).

## 3. Dataset Setup

The raw dataset is not included in this repository. It must be downloaded separately.

### Step 1: Download the Dataset

Download the [European Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and extract `creditcard.csv`.

### Step 2: Set Up the Project Directory

Place `creditcard.csv` inside the `data/raw/` directory.

Your project should initially look like this:

```text
credit-card-fraud-detection/
├── data/
│   └── raw/
│       └── creditcard.csv
├── README.md
└── credit_card_fraud_detection.ipynb
```

The notebook automatically creates the remaining project directories when executed.

### Step 3: Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn torch
```

You will also need Jupyter Notebook or another environment that supports `.ipynb` files.

### Step 4: Run the Notebook

Open `credit_card_fraud_detection.ipynb` in Jupyter Notebook and execute the cells in order.

The notebook expects the raw dataset at:

```text
data/raw/creditcard.csv
```

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Jupyter Notebook | Development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Data preprocessing and dataset splitting |
| Imbalanced-learn | Class imbalance techniques |
| PyTorch | Neural network implementation |

## 5. Project Workflow

The project follows these stages:

1. Data loading and initial inspection. **(Complete)**
2. Data cleaning and quality validation. **(Complete)**
3. Exploratory data analysis (EDA). **(Complete)**
4. Feature preprocessing and scaling. **(Complete)**
5. Training and testing dataset preparation. **(Complete)**
6. Class imbalance handling using undersampling, oversampling, and SMOTE. **(Complete)**
7. PyTorch tensor preparation. **(Complete)**
8. Neural network architecture development. **(Complete)**
9. Model instantiation and forward-pass testing. **(Complete)**
10. Model training and optimization. **(Future Development)**
11. Model evaluation and threshold selection. **(Future Development)**
12. Results analysis and conclusions. **(Future Development)**

## 6. Data Preprocessing

The preprocessing workflow includes:

- Inspecting the original dataset.
- Identifying and removing duplicate transactions.
- Checking for missing values and invalid data.
- Validating transaction class labels.
- Generating a data quality report and audit log.
- Exploring transaction distributions and amounts.
- Separating input features from target labels.
- Splitting the data into training and testing sets using stratification.
- Scaling features using `StandardScaler`.

The scaler is fitted using the training data and then applied to the testing data to prevent data leakage.

### Class Imbalance Handling

The dataset contains significantly more legitimate transactions than fraudulent transactions.

Three resampling techniques are implemented to explore different approaches to this imbalance:

- **Random Undersampling:** Reduces the number of legitimate transactions.
- **Random Oversampling:** Increases the number of fraudulent transactions by duplicating existing examples.
- **SMOTE:** Generates synthetic examples of the minority class.

Resampling is performed on the training data only. The original testing dataset remains unchanged.

## 7. Neural Network Architecture

The project implements a feedforward neural network using PyTorch for binary classification.

| Layer | Configuration |
|---|---|
| Input layer | 30 features |
| Hidden layer 1 | 64 neurons |
| Hidden layer 2 | 32 neurons |
| Output layer | 1 neuron |
| Activation function | ReLU |

**Total learnable parameters: 4,097**

The network uses fully connected layers to process transaction features and produce a single raw output (logit) for binary classification.

The output can later be converted into a probability using the sigmoid function.

## 8. Expected Outputs

The project currently produces:

- Cleaned transaction dataset.
- Data quality report.
- Data cleaning audit log.
- Exploratory data analysis visualizations.
- Preprocessed training and testing datasets.
- Resampled training datasets using multiple techniques.
- PyTorch tensors prepared for model input.
- Feedforward neural network architecture.
- Forward-pass output tests.

Future development may include:

- A trained PyTorch neural network.
- Model evaluation metrics, including precision, recall, F1-score, and Average Precision.
- Confusion matrix and performance visualizations.
- Analysis of false positives and false negatives.
- Final project findings and conclusions.

## 9. Current Project Status

The current implementation establishes the data preprocessing pipeline and initial PyTorch neural network architecture.

The model has been instantiated and its forward pass tested using the original and resampled training datasets. These tests confirm that the network accepts the prepared input tensors and produces outputs with the expected dimensions.

**The model has not been trained or evaluated.** Therefore, its current outputs are not meaningful fraud predictions, and no predictive performance claims are made.

The next development stage will focus on model training, optimization, and evaluation.

## 10. Repository Contents

```text
credit-card-fraud-detection/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── clean/
│       └── cleaned_creditcard.csv
├── models/
├── results/
│   ├── rejects.log
│   └── quality_report.md
├── README.md
└── credit_card_fraud_detection.ipynb
```

The notebook automatically creates the project directories. The raw dataset must be downloaded separately. The cleaned dataset, audit log, and quality report are generated when the notebook is executed.

## Author

**Ayodeji Osungbohun**

Computer Information Systems Student  
Texas A&M University–Central Texas

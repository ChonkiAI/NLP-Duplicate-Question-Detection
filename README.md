# NLP Duplicate Question Detection

This project uses Natural Language Processing (NLP) and Machine Learning to determine if a pair of questions from Quora are duplicates. The goal is to build a model that can accurately identify similar questions, which is a crucial task for improving user experience on platforms like Quora.

## Table of Contents
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [File Structure](#file-structure)

## Project Workflow
The project follows a standard machine learning pipeline:
1.  **Data Loading**: The `train.csv` from the Quora Question Pairs dataset is loaded.
2.  **Text Preprocessing**: A comprehensive cleaning process is applied to the questions, which includes:
    - Lowercasing text
    - Removing HTML tags and URLs
    - Replacing special characters and currency symbols
    - Expanding contractions (e.g., "don't" to "do not")
    - Removing punctuation and stopwords
    - Lemmatization to reduce words to their root form.
3.  **Feature Engineering**: To help the model understand the nuances between questions, the following features were engineered:
    - **Basic Features**: Length of questions, word count.
    - **Word-Share Features**: Number of common words, total words, and the ratio of common words.
    - **Token-Based Features**: Ratios of common non-stopwords, stopwords, and all tokens.
    - **Fuzzy Features**: FuzzyWuzzy ratios (`fuzz_ratio`, `fuzz_partial_ratio`, etc.) to measure string similarity.
4.  **Vectorization**: The cleaned text data was converted into numerical vectors using `CountVectorizer`.
5.  **Model Training**: A `RandomForestClassifier` was trained on the combination of engineered features and vectorized text data.
6.  **Model Evaluation**: The model's performance was evaluated using Accuracy, a Confusion Matrix, and a Classification Report (Precision, Recall, F1-Score).
7.  **Model Saving**: The trained `RandomForestClassifier` and `CountVectorizer` were saved as `.pkl` files for future use.

## Dataset
The dataset used is the **Quora Question Pairs** dataset, which can be found on Kaggle. It contains pairs of questions and a label indicating if they are duplicates.

- `train.csv`: The dataset used for training and evaluation.

## Technologies Used
- **Programming Language**: Python 3
- **Libraries**:
  - `pandas` & `numpy` for data manipulation
  - `scikit-learn` for machine learning (`RandomForestClassifier`, `CountVectorizer`)
  - `nltk`, `spacy`, `beautifulsoup4`, `emoji`, `textblob` for text preprocessing
  - `fuzzywuzzy` for string similarity features
  - `tqdm` for progress bars
  - `Jupyter Notebook` for development

## Setup and Installation
To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ChonkiAI/NLP-Duplicate-Question-Detection.git](https://github.com/ChonkiAI/NLP-Duplicate-Question-Detection.git)
    cd NLP-Duplicate-Question-Detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download necessary NLTK and spaCy data:**
    Run the following commands in your terminal:
    ```bash
    python -m spacy download en_core_web_sm
    ```
    Then, run this Python script to download the stopwords from NLTK:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## How to Run
Once the setup is complete, you can open and run the `MAIN.ipynb` notebook in a Jupyter environment to see the complete data processing, model training, and evaluation pipeline.

```bash
jupyter notebook MAIN.ipynb

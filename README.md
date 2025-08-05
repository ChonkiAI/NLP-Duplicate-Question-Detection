# NLP Duplicate Question Detection

This project uses Natural Language Processing (NLP) and Machine Learning to determine if a pair of questions from Quora are duplicates. The goal is to build a model that can accurately identify similar questions, which is a crucial task for improving user experience on platforms like Quora.

## Table of Contents
- [Project Workflow](#project-workflow)
- [Dataset and Pre-trained Models](#dataset-and-pre-trained-models)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [File Structure](#file-structure)

## Project Workflow
The project follows a standard machine learning pipeline:
1.  **Data Loading**: The `train.csv` from the Quora Question Pairs dataset is loaded.
2.  **Text Preprocessing**: A comprehensive cleaning process is applied to the questions, which includes lowercasing, removing HTML/URLs, expanding contractions, removing punctuation/stopwords, and lemmatization.
3.  **Feature Engineering**: A rich set of features were engineered, including basic text stats, word-share ratios, token-based features, and FuzzyWuzzy string similarity scores.
4.  **Vectorization**: The cleaned text was converted into numerical vectors using `CountVectorizer`.
5.  **Model Training**: A `RandomForestClassifier` was trained on the combination of engineered features and vectorized text.
6.  **Model Evaluation**: The model's performance was evaluated using Accuracy, a Confusion Matrix, and a Classification Report.
7.  **Model Saving**: The trained `RandomForestClassifier` and `CountVectorizer` were saved as `.pkl` files.

## Dataset and Pre-trained Models
Due to GitHub's file size limitations, the dataset and the final trained models are hosted on Google Drive.

* **[Download the required files from Google Drive](https://drive.google.com/drive/folders/1W5wwyEipz7DmF5qSpH1ec8BUNs1IRjJK?usp=sharing)**

The link contains:
-   `train.csv`: The dataset used for training and evaluation.
-   `model_final.pkl`: The trained `RandomForestClassifier`.
-   `cv_final.pkl`: The fitted `CountVectorizer`.

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

2.  **Download the data and models:**
    - Visit the **[Google Drive Folder](https://drive.google.com/drive/folders/1W5wwyEipz7DmF5qSpH1ec8BUNs1IRjJK?usp=sharing)**.
    - Download the `data` and `saved_models` folders.
    - Place both folders in the root directory of this project.

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download necessary NLTK and spaCy data:**
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

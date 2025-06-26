# AI vs Human Detection Project

This Streamlit application can distinguish AI-generated text from human-written text with three machine-learning models—Decision Tree, AdaBoost, and SVM—powered by a shared TF-IDF feature extractor.

## Requirements

- **Python 3.8+**
- Virtual environment (recommended)
- **Dependencies** listed in `requirements.txt`

## Project Structure

```
ai_human_detection_project/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Project dependencies
├── models/                  # Trained model artifacts
│   ├── svm_model.pkl
│   ├── decision_tree_model.pkl
│   ├── adaboost_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/                    # Processed data
│   ├── training_data/       # Pickled training DataFrame
│   └── test_data/           # Pickled test DataFrame or raw docs
├── notebooks/               # Jupyter notebooks for development
│   └── Your code.ipynb      # Experimentation, EDA, model training
└── README.md                # Project documentation
```

## Setup Instructions (linux based)

1. **Clone the repository** and `cd` into it:

   ```bash
   git clone https://github.com/henry-rob17/I2LLMS_Project1
   cd ai_human_detection_project
   ```

2. **Create & activate** a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Install SpaCy’s English model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Ensure your **``** directory** contains the following files:

   ```text
   svm_model.pkl
   decision_tree_model.pkl
   adaboost_model.pkl
   tfidf_vectorizer.pkl
   ```

6. **Run the application**:

   ```bash
   streamlit run app.py
   ```

Open the URL shown in your browser (typically `http://localhost:8501`).

## Usage Overview

- **Single Prediction**: Paste or type text, choose a model, and view prediction with confidence.
- **Batch Processing**: Upload `.txt` or `.csv`, select model, and download a CSV report of results.
- **Model Comparison**: Compare outputs from Decision Tree, AdaBoost, and SVM on the same input.

## Author

Henry Robinson


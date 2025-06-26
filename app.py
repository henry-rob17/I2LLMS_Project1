# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    """
    Load TF-IDF vectorizer + Decision Tree, AdaBoost, and SVM
    from './models/'—all files ending in .pkl.
    """
    models = {}
    models_dir = os.path.join(os.getcwd(), "models")

    # TF-IDF vectorizer
    vect_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(vect_path):
        models['vectorizer'] = joblib.load(vect_path)
        models['vectorizer_available'] = True
    else:
        models['vectorizer_available'] = False

    # Decision Tree
    dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
    if os.path.exists(dt_path):
        models['dt'] = joblib.load(dt_path)
        models['dt_available'] = True
    else:
        models['dt_available'] = False

    # AdaBoost
    ada_path = os.path.join(models_dir, "adaboost_model.pkl")
    if os.path.exists(ada_path):
        models['ada'] = joblib.load(ada_path)
        models['ada_available'] = True
    else:
        models['ada_available'] = False

    # SVM
    svm_path = os.path.join(models_dir, "svm_model.pkl")
    if os.path.exists(svm_path):
        models['svm'] = joblib.load(svm_path)
        models['svm_available'] = True
    else:
        models['svm_available'] = False

    # Sanity check: need at least one classifier + vectorizer
    if not (models.get('vectorizer_available', False) and
            (models.get('dt_available', False) or
             models.get('ada_available', False) or
             models.get('svm_available', False))):
        st.error("❌ Could not find a valid TF-IDF + model combo in ./models/")
        return None

    return models

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================


# Text‐processing function (spaCy‐based)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
_alphanum_re = re.compile(r"(?=.*\d)(?=.*[A-Za-z])")

# text_process definition
def text_process(text: str) -> str:
    """
    - lowercase
    - remove stopwords, punctuation, whitespace
    - drop tokens <2 chars, numeric, alphanumeric
    - keep only alphabetic tokens
    """
    if not isinstance(text, str):
        return ""
    tokens = []
    for token in nlp(text.lower()):
        t = token.text
        if token.is_stop:            continue
        if token.is_punct or token.is_space:  continue
        if len(t) < 2:               continue
        if token.like_num:           continue
        if _alphanum_re.search(t):   continue
        if not t.isalpha():          continue
        tokens.append(t)
    return " ".join(tokens)






from sklearn.pipeline import Pipeline

def make_prediction(text, model_choice, models):
    """
    Choose the right object (pipeline or estimator), then
    either call pipeline.predict_proba on [text], or
    classifier.predict_proba on X = vectorizer.transform([text]).
    """
    if models is None:
        return None, None

    # Map display name → (dict key, availability flag)
    mapping = {
        "Decision Tree": ("dt",  models.get("dt_available", False)),
        "AdaBoost":      ("ada", models.get("ada_available", False)),
        "SVM":           ("svm", models.get("svm_available", False)),
    }
    key, available = mapping.get(model_choice, (None, False))
    if not available:
        return None, None

    clf = models[key]

    # If this object is a Pipeline, it already wraps vectorizer → classifier
    if isinstance(clf, Pipeline):
        # feed it raw text
        docs = [text]
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(docs)[0]
        else:
            p0 = clf.predict(docs)[0]
            proba = [0.0, 0.0]
            proba[p0] = 1.0
        pred = clf.predict(docs)[0]

    else:
        # fallback: you loaded a bare classifier + separate vectorizer
        if not models.get("vectorizer_available", False):
            return None, None
        X = models["vectorizer"].transform([text])
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
        else:
            p0 = clf.predict(X)[0]
            proba = [0.0, 0.0]
            proba[p0] = 1.0
        pred = clf.predict(X)[0]

    label = "AI-generated" if pred == 1 else "Human-written"
    return label, proba


# ============================================================================ 
#GET_AVAILABLE_MODELS
def get_available_models(models):
    opts = []
    if not models:
        return opts
    if models.get('dt_available'):
        opts.append("Decision Tree")
    if models.get('ada_available'):
        opts.append("AdaBoost")
    if models.get('svm_available'):
        opts.append("SVM")
    return opts




# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    # Main header
    st.markdown(
        '<h1 class="main-header">🤖 AI vs Human Text Classifier</h1>',
        unsafe_allow_html=True
    )
    
    # Welcome text
    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates classification
    of AI-generated vs. human-written text using three trained models: **Decision Tree**, **AdaBoost**, and **SVM**.
    """)
    
    # App overview cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually  
        - Choose between models  
        - Get instant predictions  
        - See confidence scores  
        """)
    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text files  
        - Process multiple texts  
        - Compare model performance  
        - Download results  
        """)
    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare different models  
        - Side-by-side results  
        - Agreement analysis  
        - Performance metrics  
        """)
    
    # Model status section
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        col1, col2, col3 = st.columns(3)

        with col1:
            if models.get('dt_available'):
                st.info("🌳 Decision Tree — ✅ Available")
            else:
                st.warning("🌳 Decision Tree — ❌ Not Found")

        with col2:
            if models.get('ada_available'):
                st.info("✨ AdaBoost — ✅ Available")
            else:
                st.warning("✨ AdaBoost — ❌ Not Found")

        with col3:
            if models.get('svm_available'):
                st.info("🖋️ SVM — ✅ Available")
            else:
                st.warning("🖋️ SVM — ❌ Not Found")

    else:
        st.error("❌ Models not loaded. Please check the model files.")


# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to classify it as AI-generated or human-written.")
    
    if models:
        # Build a simple list of names
        model_options = get_available_models(models)
        
        if model_options:
            # Model selector
            model_choice = st.selectbox(
                "Choose a model:",
                options=model_options
            )
            
            # Text input area
            user_input = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Type or paste your text (e.g., an essay, article, or snippet)…"
            )
            
            # Show counts
            if user_input:
                st.caption(f"Characters: {len(user_input)}   Words: {len(user_input.split())}")
            
            # Predict button
            if st.button("🚀 Predict"):
                if not user_input.strip():
                    st.warning("Please enter some text to classify!")
                else:
                    with st.spinner("Analyzing..."):
                        label, proba = make_prediction(user_input, model_choice, models)
                    if label is not None:
                        # Display result
                        col1, col2 = st.columns([3,1])
                        with col1:
                            if label == "AI-generated":
                                st.error(f"🤖 Prediction: **{label}**")
                            else:
                                st.success(f"👤 Prediction: **{label}**")
                        with col2:
                            conf = max(proba)
                            st.metric("Confidence", f"{conf:.1%}")
                        
                        # Probability breakdown
                        st.subheader("📊 Class Probabilities")
                        prob_df = pd.DataFrame({
                            "Class": ["Human-written", "AI-generated"],
                            "Probability": proba
                        }).set_index("Class")
                        st.bar_chart(prob_df)
                    else:
                        st.error("❌ Prediction failed. Check that your model files exist and are valid.")
        else:
            st.error("No models available for single prediction.")
    else:
        st.warning("Models not loaded. Please revisit the Home page to load your models first.")

# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text file or CSV to process multiple texts at once.")
    
    if models:
        available_models = get_available_models(models)
        
        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv'],
                help="Upload a .txt file (one text per line) or .csv file (text in first column)"
            )
            
            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                # Process file
                if st.button("📊 Process File"):
                    try:
                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]
                        else:  # CSV
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).tolist()
                        
                        if not texts:
                            st.error("No text found in file")
                        else:
                            st.info(f"Processing {len(texts)} texts...")
                            
                            # Process all texts
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(texts):
                                if text.strip():
                                    prediction, probabilities = make_prediction(text, model_choice, models)
                                    
                                    if prediction and probabilities is not None:
                                        results.append({
                                            'Text': text[:100] + "..." if len(text) > 100 else text,
                                            'Full_Text': text,
                                            'Prediction': prediction,
                                            'Confidence': f"{max(probabilities):.1%}",
                                            'Negative_Prob': f"{probabilities[0]:.1%}",
                                            'Positive_Prob': f"{probabilities[1]:.1%}"
                                        })
                                
                                progress_bar.progress((i + 1) / len(texts))
                            
                            if results:
                                # Display results
                                st.success(f"✅ Processed {len(results)} texts successfully!")
                                
                                results_df = pd.DataFrame(results)
                                
                                # Summary statistics
                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                positive_count = sum(1 for r in results if r['Prediction'] == 'Positive')
                                negative_count = len(results) - positive_count
                                avg_confidence = np.mean([float(r['Confidence'].strip('%')) for r in results])
                                
                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("😊 Positive", positive_count)
                                with col3:
                                    st.metric("😞 Negative", negative_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                                
                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(
                                    results_df[['Text', 'Prediction', 'Confidence']],
                                    use_container_width=True
                                )
                                
                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed")
                                
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a file to get started.")
                
                # Show example file formats
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    This product is amazing!
                    Terrible quality, very disappointed
                    Great service and fast delivery
                    ```
                    
                    **CSV File (.csv):**
                    ```
                    text,category
                    "Amazing product, love it!",review
                    "Poor quality, not satisfied",review
                    ```
                    """)
        else:
            st.error("No models available for batch processing.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")
    
    if models:
        available_models = get_available_models(models)
        
        if len(available_models) >= 2:
            # Text input for comparison
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Enter text to see how different models perform...",
                height=100
            )
            
            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")
                
                # Get predictions from all available models
                comparison_results = []
                
                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)
                    
                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Negative %': f"{probabilities[0]:.1%}",
                            'Positive %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })
                
                if comparison_results:
                    # Comparison table
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Negative %', 'Positive %']])
                    
                    # Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]} Sentiment**")
                    else:
                        st.warning("⚠️ Models disagree on prediction")
                        for result in comparison_results:
                            model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                            st.write(f"- {model_name}: {result['Prediction']}")
                    
                    # Side-by-side probability charts
                    st.subheader("📊 Detailed Probability Comparison")
                    
                    cols = st.columns(len(comparison_results))
                    
                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            model_name = result['Model']
                            st.write(f"**{model_name}**")
                            
                            chart_data = pd.DataFrame({
                                'Sentiment': ['Negative', 'Positive'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('Sentiment'))
                    
                else:
                    st.error("Failed to get predictions from models")
        
        elif len(available_models) == 1:
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")
            
        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================
elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    if models:
        st.success("✅ Models are loaded and ready!")
        
        # Model details
        st.subheader("🔧 Available Models")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🌳 Decision Tree
            **Type:** Tree-Based Classifier  
            **Features:** TF-IDF vectors (unigrams + bigrams)  
            **Strengths:**  
            - Captures non-linear patterns  
            - Easily interpretable decision rules  
            - Fast inference  
            """)
        
        with col2:
            st.markdown("""
            ### ✨ AdaBoost
            **Type:** Ensemble of Decision Trees  
            **Features:** Same TF-IDF + boosting  
            **Strengths:**  
            - Reduced variance via boosting  
            - Good generalization on noisy data  
            - Handles class imbalance well  
            """)
        
        with col3:
            st.markdown("""
            ### 🖋️ SVM
            **Type:** Margin-Based Classifier  
            **Features:** TF-IDF vectors in high-dimensional space  
            **Strengths:**  
            - Effective when feature space is large  
            - Robust to overfitting with proper regularization  
            - Can model non-linear boundaries with kernels  
            """)
        
        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)  
        - **Max Features:** 5,000 most important terms  
        - **N-grams:** Unigrams & Bigrams  
        - **Min Document Frequency:** 5 documents  
        - **Normalization:** L2 norm  
        """)
        
        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        files_to_check = [
            ("tfidf_vectorizer.pkl",      "TF-IDF Vectorizer",     models.get('vectorizer_available', False)),
            ("decision_tree_best.pkl",    "Decision Tree Model",   models.get('dt_available', False)),
            ("adaboost_model.pkl",        "AdaBoost Model",        models.get('ada_available', False)),
            ("svm_model.pkl",             "SVM Model",             models.get('svm_available', False)),
        ]
        for filename, desc, ok in files_to_check:
            file_status.append({
                "File":        filename,
                "Description": desc,
                "Status":      "✅ Loaded" if ok else "❌ Not Found"
            })
        st.table(pd.DataFrame(file_status))
        
        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** AI vs. Human Text Corpus  
        - **Classes:** AI-generated (1) and Human-written (0)  
        - **Preprocessing:** Lowercasing, stopword removal, lemmatization  
        - **Training:** 5-fold cross-validation for hyperparameter tuning  
        """)
    else:
        st.warning("Models not loaded. Please check the model files in the 'models/' directory.")
# ============================================================================
# HELP PAGE
# ============================================================================
elif page == "❓ Help":
    st.header("❓ How to Use This App")
    
    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model**: Decision Tree, AdaBoost, or SVM.  
        2. **Enter your text** in the text area (paste or type).  
        3. **Click “Predict”** to classify your text.  
        4. **View results**: label (AI-generated or Human-written) and confidence score.  
        """)
    
    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file**:  
           - **.txt**: one text sample per line  
           - **.csv**: text in the first column  
        2. **Upload the file**.  
        3. **Choose a model**.  
        4. **Click “Process File”** to analyze all entries.  
        5. **Download results** as a CSV with predictions and probabilities.  
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want compared.  
        2. **Click “Compare All Models”** to run Decision Tree, AdaBoost, and SVM on the same input.  
        3. **See side-by-side results**: each model’s prediction and confidence.  
        4. **Check agreement**: identify consensus or disagreements.  
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        - **Models not loading?**  
          Ensure these files exist in `./models/`:  
          `tfidf_vectorizer.pkl`, `decision_tree_best.pkl`, `adaboost_model.pkl`, `svm_model.pkl`.  
        - **spaCy model error?**  
          Activate your virtual environment and run:  
          ```bash
          python -m spacy download en_core_web_sm
          ```  
        - **Identical outputs?**  
          Confirm that `text_process` is applied before prediction and that your models were trained on the cleaned data.  
        - **File upload issues?**  
          Use UTF-8 encoded `.txt` or `.csv` files with text in the correct format.  
        """)



# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.info("""
**AI vs Human Text Classifier App**  
Built with Streamlit

**Models:**  
- 🌳 Decision Tree  
- ✨ AdaBoost  
- 🖋️ SVM  

**Framework:** scikit-learn  
**Deployment:** Streamlit Cloud Ready
""")


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with CAFFEINE using Streamlit | Machine Learning Text Classification Demo | By Henry Robinson<br>
</div>
""", unsafe_allow_html=True)
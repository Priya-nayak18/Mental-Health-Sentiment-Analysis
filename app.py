import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠"
)



BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Combined Data.csv"
BUNDLE_PATH = BASE_DIR / "mental_health_model_bundle.joblib"

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f8f7ff 0%, #f5fbff 100%); }
[data-testid="stSidebar"] { background: #f0efff; }
.hero {
    padding: 1.6rem; border-radius: 18px;
    background: linear-gradient(120deg, #e9e7ff, #e5f7f4);
    border: 1px solid #e0defb; margin-bottom: 1rem;
}
.hero h1 { color: #25275f; }
div.stButton > button {
    border-radius: 10px; font-weight: 600;
    background: #625cf6; color: white; border: 0;
}
</style>
""", unsafe_allow_html=True)


# Keep this function consistent with the notebook's preprocessing.
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Make sure data/Combined Data.csv is included in the project."
        )
    raw = pd.read_csv(DATA_PATH)
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")

    required = {"statement", "status"}
    if not required.issubset(raw.columns):
        raise ValueError(
            "The CSV must contain 'statement' and 'status' columns."
        )

    raw = raw.dropna(subset=["statement", "status"]).copy()
    raw["statement"] = raw["statement"].astype(str).str.strip()
    raw["status"] = raw["status"].astype(str).str.strip()
    raw = raw[(raw["statement"] != "") & (raw["status"] != "")]
    raw = raw.drop_duplicates(subset=["statement", "status"]).copy()
    raw["clean_statement"] = raw["statement"].apply(clean_text)
    raw = raw[raw["clean_statement"].str.len() > 0].copy()
    raw["word_count"] = raw["clean_statement"].str.split().str.len()
    return raw.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    # Reuse the saved model if it is already available.
    if BUNDLE_PATH.exists():
        try:
            bundle = joblib.load(BUNDLE_PATH)
            needed = {"model", "tfidf_vectorizer", "feature_type"}
            if needed.issubset(bundle.keys()):
                return bundle
        except Exception:
            # If the saved file is invalid, retrain below.
            pass

    df = load_dataset()
    if df.empty or df["status"].nunique() < 2:
        raise ValueError("The dataset must contain text and at least two categories.")

    # Match the notebook's TF-IDF configuration.
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(df["clean_statement"])
    y = df["status"]

    # The notebook selected Logistic Regression by Macro F1-score.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "model_name": "Logistic Regression",
        "feature_type": "tfidf",
        "tfidf_vectorizer": vectorizer,
        "svd": None,
        "labels": sorted(y.unique()),
    }
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


# ---------------- SIDEBAR / NAVIGATION ----------------
st.sidebar.markdown("# 🧠 Mental Health Sentiment Analysis")
st.sidebar.caption("Text Analysis Using Machine Learning and NLP")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "Dataset Overview", "Model Information", "Model Evaluation", "About"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Educational use only — not a medical diagnosis.")


# ---------------- LOAD DATA ----------------
try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# ---------------- HOME ----------------
if page == "Home":
    st.markdown("""
    <div class="hero">
      <h1>🧠 Welcome to Mental Health Sentiment Analysis</h1>
<p>Text Analysis Using Machine Learning and NLP</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records after cleaning", f"{len(df):,}")
    c2.metric("Dataset categories", f"{df['status'].nunique():,}")
    c3.metric("Average words per statement", f"{df['word_count'].mean():.1f}")

    st.subheader("How it works")
    st.markdown("""
    1. Enter a text statement on the **Predict** page.
    2. The text is cleaned using the same function as the notebook.
    3. TF-IDF converts the text into numerical features.
    4. Logistic Regression predicts one of the dataset's existing categories.
    """)
    st.info(
        "The first prediction may take some time because the model is trained "
        "and saved automatically. Later predictions reuse the saved model."
    )


# ---------------- PREDICT ----------------
elif page == "Predict":
    st.markdown("""
    <div class="hero">
      <h1>💬 Enter a text statement</h1>
     <p>Analyze text using Machine Learning and NLP.</p>
    </div>
    """, unsafe_allow_html=True)

    statement = st.text_area(
        "Your statement",
        placeholder="Type a statement here...",
        height=180,
        max_chars=5000,
    )

    if st.button("Predict category", use_container_width=True):
        if not statement.strip():
            st.warning("Please enter a statement first.")
        else:
            try:
                with st.spinner("Loading or training the model. The first run may take a little while..."):
                    bundle = load_or_train_model()
                    cleaned = clean_text(statement)
                    features = bundle["tfidf_vectorizer"].transform([cleaned])
                    prediction = bundle["model"].predict(features)[0]

                st.success(f"Predicted dataset category: **{prediction}**")
                st.caption(f"Model used: {bundle.get('model_name', 'Logistic Regression')}")
                st.warning(
                    "This is a text-classification prediction, not a diagnosis. "
                    "It may be incorrect and should not be used to make health decisions."
                )
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")


# ---------------- DATASET OVERVIEW ----------------
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records after cleaning", f"{len(df):,}")
    c2.metric("Categories", f"{df['status'].nunique():,}")
    c3.metric("Unique statements", f"{df['statement'].nunique():,}")
    c4.metric("Average words", f"{df['word_count'].mean():.1f}")

    st.subheader("Category distribution")
    counts = df["status"].value_counts().rename_axis("Category").reset_index(name="Count")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=counts, x="Count", y="Category", ax=ax)
    ax.set_xlabel("Number of statements")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sample records")
    st.dataframe(
        df[["statement", "status", "word_count"]].head(15),
        use_container_width=True,
        hide_index=True,
    )


# ---------------- MODEL INFORMATION ----------------
elif page == "Model Information":
    st.title("📈 Model Information")
    st.markdown("""
    **Selected model:** Logistic Regression

    **Text representation:** TF-IDF (unigrams and bigrams)

    **Selection basis:** The uploaded notebook selected Logistic Regression
    as the best model using Macro F1-score on its held-out test set.

    The notebook's recorded classification report shows approximately:
    - Accuracy: **0.77**
    - Macro F1-score: **0.71**
    - Weighted F1-score: **0.77**

    


    """)


# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")
    st.caption(
        "These are the recorded held-out test results from the Jupyter Notebook. "
        "They are displayed here for reference; this page does not retrain or re-evaluate the models."
    )

    comparison = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.7696, "Precision (Macro)": 0.6895, "Recall (Macro)": 0.7448, "F1 Score (Macro)": 0.7108, "F1 Score (Weighted)": 0.7695, "Training Time (seconds)": 14.53},
        {"Model": "Linear SVM", "Accuracy": 0.7633, "Precision (Macro)": 0.7021, "Recall (Macro)": 0.6993, "F1 Score (Macro)": 0.6993, "F1 Score (Weighted)": 0.7610, "Training Time (seconds)": 10.06},
        {"Model": "Random Forest", "Accuracy": 0.7274, "Precision (Macro)": 0.7433, "Recall (Macro)": 0.5964, "F1 Score (Macro)": 0.6282, "F1 Score (Weighted)": 0.7209, "Training Time (seconds)": 34.78},
        {"Model": "HistGradientBoosting", "Accuracy": 0.7019, "Precision (Macro)": 0.5894, "Recall (Macro)": 0.5230, "F1 Score (Macro)": 0.5441, "F1 Score (Weighted)": 0.6910, "Training Time (seconds)": 3.79},
        {"Model": "Multinomial Naive Bayes", "Accuracy": 0.6951, "Precision (Macro)": 0.6316, "Recall (Macro)": 0.4720, "F1 Score (Macro)": 0.4949, "F1 Score (Weighted)": 0.6748, "Training Time (seconds)": 0.21},
        {"Model": "KNN", "Accuracy": 0.6491, "Precision (Macro)": 0.4980, "Recall (Macro)": 0.4676, "F1 Score (Macro)": 0.4704, "F1 Score (Weighted)": 0.6334, "Training Time (seconds)": 0.06},
    ])
    st.subheader("Comparison of six machine-learning models")
    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.4f}", "Precision (Macro)": "{:.4f}",
            "Recall (Macro)": "{:.4f}", "F1 Score (Macro)": "{:.4f}",
            "F1 Score (Weighted)": "{:.4f}", "Training Time (seconds)": "{:.2f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.subheader("Macro F1-score comparison")
    chart_df = comparison[["Model", "F1 Score (Macro)"]].set_index("Model")
    st.bar_chart(chart_df)

    st.subheader("Classification report — Logistic Regression")
    report = pd.DataFrame([
        {"Category": "Anxiety", "Precision": 0.75, "Recall": 0.82, "F1-score": 0.78, "Support": 725},
        {"Category": "Bipolar", "Precision": 0.73, "Recall": 0.78, "F1-score": 0.75, "Support": 500},
        {"Category": "Depression", "Precision": 0.80, "Recall": 0.62, "F1-score": 0.70, "Support": 3019},
        {"Category": "Normal", "Precision": 0.90, "Recall": 0.93, "F1-score": 0.91, "Support": 3208},
        {"Category": "Personality disorder", "Precision": 0.48, "Recall": 0.60, "F1-score": 0.53, "Support": 179},
        {"Category": "Stress", "Precision": 0.49, "Recall": 0.70, "F1-score": 0.58, "Support": 459},
        {"Category": "Suicidal", "Precision": 0.68, "Recall": 0.76, "F1-score": 0.72, "Support": 2128},
    ])
    st.dataframe(
        report.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}),
        use_container_width=True, hide_index=True
    )
    st.markdown(
        "**Overall test results:** Accuracy 0.77 · Macro F1-score 0.71 · "
        "Weighted F1-score 0.77 · Test samples 10,218"
    )

    st.subheader("Confusion matrix — Logistic Regression")
    cm_path = BASE_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix from the notebook's held-out test set", use_container_width=True)
    else:
        st.info("The confusion matrix image is not in the project folder. Add confusion_matrix.png beside app.py to display it.")

    st.warning(
        "These values are the notebook's saved evaluation results. They are not a medical assessment, "
        "and predictions must not be used to diagnose mental health conditions."
    )



# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About Mental Health Sentiment Analysis")
    st.markdown("""
    **Project:** Mental Health Sentiment Analysis

    **Description:** Text Analysis Using Machine Learning and NLP

    **Technologies:** Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Streamlit.

    **Purpose:** Demonstrate supervised text classification using the labels
    present in the dataset.
    """)
    st.warning(
        "This project is for educational demonstration only. It does not diagnose "
        "mental health conditions or replace assessment by a qualified professional."
    )


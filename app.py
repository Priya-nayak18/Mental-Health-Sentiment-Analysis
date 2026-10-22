
import re
from pathlib import Path

import gdown
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
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Combined Data.csv"
BUNDLE_PATH = BASE_DIR / "mental_health_model_bundle.joblib"

# Replace this with the ID from your Google Drive sharing link.
# Example link: https://drive.google.com/file/d/FILE_ID/view
GOOGLE_DRIVE_FILE_ID = "1bo432nn1YkbnS7G8QChCtJgZcueyUQE9"


# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8f7ff 0%, #f5fbff 100%);
}
[data-testid="stSidebar"] {
    background: #f0efff;
}
.hero {
    padding: 1.6rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #e9e7ff, #e5f7f4);
    border: 1px solid #e0defb;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #25275f;
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    background: #625cf6;
    color: white;
    border: 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- DATASET LOADING ----------------
@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        if (
            not GOOGLE_DRIVE_FILE_ID
            or GOOGLE_DRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE"
        ):
            raise RuntimeError(
                "Please set GOOGLE_DRIVE_FILE_ID in app.py "
                "to your Google Drive CSV file ID."
            )

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        download_url = (
            f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        )

        with st.spinner("Downloading the dataset from Google Drive..."):
            result = gdown.download(
                url=download_url,
                output=str(DATA_PATH),
                quiet=False,
            )

        if result is None or not DATA_PATH.exists():
            raise RuntimeError(
                "Dataset download failed. Check the Google Drive "
                "file ID and sharing permissions."
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

    raw = raw[
        (raw["statement"] != "") & (raw["status"] != "")
    ].copy()

    raw = raw.drop_duplicates(
        subset=["statement", "status"]
    ).copy()

    raw["clean_statement"] = raw["statement"].apply(clean_text)
    raw = raw[raw["clean_statement"].str.len() > 0].copy()
    raw["word_count"] = raw["clean_statement"].str.split().str.len()

    return raw.reset_index(drop=True)


# ---------------- MODEL LOADING / TRAINING ----------------
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    # Try the saved model bundle first.
    if BUNDLE_PATH.exists():
        try:
            bundle = joblib.load(BUNDLE_PATH)
            required_keys = {
                "model",
                "tfidf_vectorizer",
                "feature_type",
            }
            if required_keys.issubset(bundle.keys()):
                return bundle
        except Exception:
            pass

    df = load_dataset()

    if df.empty or df["status"].nunique() < 2:
        raise ValueError(
            "The dataset must contain text and at least two categories."
        )

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(df["clean_statement"])
    y = df["status"]

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
    [
        "Home",
        "Predict",
        "Dataset Overview",
        "Model Information",
        "Model Evaluation",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Educational use only — not a medical diagnosis."
)


# ---------------- LOAD DATA ----------------
try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info(
        "Check your Google Drive file ID, sharing settings, "
        "and internet access. Confirm that the CSV contains "
        "'statement' and 'status' columns."
    )
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
    c3.metric(
        "Average words per statement",
        f"{df['word_count'].mean():.1f}",
    )

    st.subheader("How it works")
    st.markdown("""
    1. Enter a text statement on the **Predict** page.
    2. The text is cleaned using the same function as the notebook.
    3. TF-IDF converts the text into numerical features.
    4. Logistic Regression predicts one of the dataset's categories.
    """)

    st.info(
        "The first prediction may take some time because the model "
        "may need to be trained. Later predictions can reuse the "
        "saved model while that file remains available."
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
                with st.spinner(
                    "Loading or training the model. "
                    "The first run may take a while..."
                ):
                    bundle = load_or_train_model()
                    cleaned = clean_text(statement)
                    features = bundle["tfidf_vectorizer"].transform(
                        [cleaned]
                    )
                    prediction = bundle["model"].predict(features)[0]

                st.success(
                    f"Predicted dataset category: **{prediction}**"
                )
                st.caption(
                    "Model used: "
                    f"{bundle.get('model_name', 'Logistic Regression')}"
                )
                st.warning(
                    "This is a text-classification prediction, not a "
                    "diagnosis. It may be incorrect and should not be "
                    "used to make health decisions."
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

    counts = (
        df["status"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Count")
    )

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

    **Selection basis:** The original notebook selected Logistic
    Regression as the best model using Macro F1-score on its held-out
    test set.

    The notebook's recorded classification results are approximately:

    - Accuracy: **0.77**
    - Macro F1-score: **0.71**
    - Weighted F1-score: **0.77**
    """)

    st.info(
        "The model trained by this app uses the full cleaned dataset. "
        "The metrics above are recorded notebook results and are not "
        "a new evaluation of this app-trained model."
    )


# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")

    st.caption(
        "These are the recorded held-out test results from the notebook. "
        "They are displayed for reference; this page does not retrain "
        "or re-evaluate the models."
    )

    comparison = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.7696,
            "Precision (Macro)": 0.6895,
            "Recall (Macro)": 0.7448,
            "F1 Score (Macro)": 0.7108,
            "F1 Score (Weighted)": 0.7695,
            "Training Time (seconds)": 14.53,
        },
        {
            "Model": "Linear SVM",
            "Accuracy": 0.7633,
            "Precision (Macro)": 0.7021,
            "Recall (Macro)": 0.6993,
            "F1 Score (Macro)": 0.6993,
            "F1 Score (Weighted)": 0.7610,
            "Training Time (seconds)": 10.06,
        },
        {
            "Model": "Random Forest",
            "Accuracy": 0.7274,
            "Precision (Macro)": 0.7433,
            "Recall (Macro)": 0.5964,
            "F1 Score (Macro)": 0.6282,
            "F1 Score (Weighted)": 0.7209,
            "Training Time (seconds)": 34.78,
        },
        {
            "Model": "HistGradientBoosting",
            "Accuracy": 0.7019,
            "Precision (Macro)": 0.5894,
            "Recall (Macro)": 0.5230,
            "F1 Score (Macro)": 0.5441,
            "F1 Score (Weighted)": 0.6910,
            "Training Time (seconds)": 3.79,
        },
        {
            "Model": "Multinomial Naive Bayes",
            "Accuracy": 0.6951,
            "Precision (Macro)": 0.6316,
            "Recall (Macro)": 0.4720,
            "F1 Score (Macro)": 0.4949,
            "F1 Score (Weighted)": 0.6748,
            "Training Time (seconds)": 0.21,
        },
        {
            "Model": "KNN",
            "Accuracy": 0.6491,
            "Precision (Macro)": 0.4980,
            "Recall (Macro)": 0.4676,
            "F1 Score (Macro)": 0.4704,
            "F1 Score (Weighted)": 0.6334,
            "Training Time (seconds)": 0.06,
        },
    ])

    st.subheader("Comparison of six machine-learning models")
    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.4f}",
            "Precision (Macro)": "{:.4f}",
            "Recall (Macro)": "{:.4f}",
            "F1 Score (Macro)": "{:.4f}",
            "F1 Score (Weighted)": "{:.4f}",
            "Training Time (seconds)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Macro F1-score comparison")
    chart_df = comparison[
        ["Model", "F1 Score (Macro)"]
    ].set_index("Model")
    st.bar_chart(chart_df)

    st.subheader("Classification report — Logistic Regression")

    report = pd.DataFrame([
        {
            "Category": "Anxiety",
            "Precision": 0.75,
            "Recall": 0.82,
            "F1-score": 0.78,
            "Support": 725,
        },
        {
            "Category": "Bipolar",
            "Precision": 0.73,
            "Recall": 0.78,
            "F1-score": 0.75,
            "Support": 500,
        },
        {
            "Category": "Depression",
            "Precision": 0.80,
            "Recall": 0.62,
            "F1-score": 0.70,
            "Support": 3019,
        },
        {
            "Category": "Normal",
            "Precision": 0.90,
            "Recall": 0.93,
            "F1-score": 0.91,
            "Support": 3208,
        },
        {
            "Category": "Personality disorder",
            "Precision": 0.48,
            "Recall": 0.60,
            "F1-score": 0.53,
            "Support": 179,
        },
        {
            "Category": "Stress",
            "Precision": 0.49,
            "Recall": 0.70,
            "F1-score": 0.58,
            "Support": 459,
        },
        {
            "Category": "Suicidal",
            "Precision": 0.68,
            "Recall": 0.76,
            "F1-score": 0.72,
            "Support": 2128,
        },
    ])

    st.dataframe(
        report.style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-score": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**Overall test results:** Accuracy 0.77 · Macro F1-score 0.71 · "
        "Weighted F1-score 0.77 · Test samples 10,218"
    )

    st.subheader("Confusion matrix — Logistic Regression")

    cm_path = BASE_DIR / "confusion_matrix.png"

    if cm_path.exists():
        st.image(
            str(cm_path),
            caption="Confusion matrix from the notebook's held-out test set",
            use_container_width=True,
        )
    else:
        st.info(
            "The confusion matrix image is not in the project folder. "
            "Add confusion_matrix.png beside app.py to display it."
        )

    st.warning(
        "These values are recorded notebook results, not a medical "
        "assessment. Predictions must not be used to diagnose "
        "mental health conditions."
    )


# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About Mental Health Sentiment Analysis")

    st.markdown("""
    **Project:** Mental Health Sentiment Analysis

    **Description:** Text Analysis Using Machine Learning and NLP

    **Technologies:** Python, Pandas, Scikit-learn, TF-IDF,
    Logistic Regression, Streamlit.

    **Purpose:** Demonstrate supervised text classification using
    the labels present in the dataset.
    """)

    st.warning(
        "This project is for educational demonstration only. It does "
        "not diagnose mental health conditions or replace assessment "
        "by a qualified professional."
    )

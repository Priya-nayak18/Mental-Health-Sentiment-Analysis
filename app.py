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


# ============================================================
# SETTINGS
# ============================================================

st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Combined Data.csv"
BUNDLE_PATH = BASE_DIR / "mental_health_model_bundle.joblib"

# Google Drive CSV file ID
GOOGLE_DRIVE_FILE_ID = "1bo432nn1YkbnS7G8QChCtJgZcueyUQE9"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------------- MAIN APP ---------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f7ff 0%,
            #f5fbff 100%
        );
    }

    /* ---------------- SIDEBAR ---------------- */

    [data-testid="stSidebar"] {
        background: #f0efff;
    }

    [data-testid="stSidebar"] h1 {
        color: #25275f;
        font-size: 1.7rem;
        line-height: 1.25;
    }

    /* ---------------- MAIN TITLE ---------------- */

    .main-title {
        color: #292b70;
        font-size: 3.1rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }

    .main-subtitle {
        color: #626783;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }

    /* ---------------- HERO ---------------- */

    .hero-box {
        background: linear-gradient(
            120deg,
            #e9e7ff,
            #e5f7f4
        );
        border: 1px solid #dedcf8;
        border-radius: 22px;
        padding: 2.5rem 2.5rem;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 8px 25px rgba(60, 60, 120, 0.05);
    }

    .hero-title {
        color: #292b70;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
    }

    .hero-subtitle {
        color: #565b78;
        font-size: 1.15rem;
        margin-top: 1rem;
        margin-bottom: 0;
    }

    /* ---------------- CARDS ---------------- */

    .info-card {
        background: white;
        border: 1px solid #e1e0f4;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 20px rgba(50, 50, 100, 0.04);
    }

    .info-card h3 {
        color: #292b70;
        margin-top: 0;
    }

    /* ---------------- BUTTON ---------------- */

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        background: #625cf6;
        color: white;
        border: 0;
        padding: 0.65rem 1rem;
    }

    div.stButton > button:hover {
        background: #514be4;
        color: white;
    }

    /* ---------------- HEADINGS ---------------- */

    h1, h2, h3 {
        color: #292b70;
    }

    /* ---------------- METRICS ---------------- */

    [data-testid="stMetricValue"] {
        color: #292b70;
        font-weight: 700;
    }

    /* ---------------- TEXT AREA ---------------- */

    textarea {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Download dataset if it does not exist
    # --------------------------------------------------------

    if (
        not DATA_PATH.exists()
        or DATA_PATH.stat().st_size == 0
    ):

        if not GOOGLE_DRIVE_FILE_ID:
            raise RuntimeError(
                "Google Drive file ID is missing."
            )

        download_url = (
            f"https://drive.google.com/uc?id="
            f"{GOOGLE_DRIVE_FILE_ID}"
        )

        try:

            with st.spinner(
                "Downloading dataset from Google Drive..."
            ):

                # IMPORTANT:
                # Do NOT use fuzzy=True.
                # It causes an error with some gdown versions.
                result = gdown.download(
                    url=download_url,
                    output=str(DATA_PATH),
                    quiet=False,
                )

        except Exception as exc:

            raise RuntimeError(
                f"Dataset download failed: {exc}"
            ) from exc

        if (
            result is None
            or not DATA_PATH.exists()
            or DATA_PATH.stat().st_size == 0
        ):

            raise RuntimeError(
                "Dataset download failed. "
                "Check the Google Drive file ID and "
                "sharing permissions."
            )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        raw = pd.read_csv(DATA_PATH)

    except Exception as exc:

        raise RuntimeError(
            "Could not read the downloaded file as CSV. "
            "Make sure the Google Drive file is actually "
            "Combined Data.csv."
        ) from exc

    # --------------------------------------------------------
    # Remove unwanted index column
    # --------------------------------------------------------

    raw = raw.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {
        "statement",
        "status",
    }

    if not required_columns.issubset(raw.columns):

        raise ValueError(
            "The CSV must contain 'statement' and 'status' "
            f"columns.\n\nColumns found: {list(raw.columns)}"
        )

    # --------------------------------------------------------
    # Clean rows
    # --------------------------------------------------------

    raw = raw.dropna(
        subset=[
            "statement",
            "status",
        ]
    ).copy()

    raw["statement"] = (
        raw["statement"]
        .astype(str)
        .str.strip()
    )

    raw["status"] = (
        raw["status"]
        .astype(str)
        .str.strip()
    )

    raw = raw[
        (raw["statement"] != "")
        & (raw["status"] != "")
    ].copy()

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    raw = raw.drop_duplicates(
        subset=[
            "statement",
            "status",
        ]
    ).copy()

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    raw["clean_statement"] = (
        raw["statement"]
        .apply(clean_text)
    )

    raw = raw[
        raw["clean_statement"].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # Word count
    # --------------------------------------------------------

    raw["word_count"] = (
        raw["clean_statement"]
        .str.split()
        .str.len()
    )

    if raw.empty:
        raise ValueError(
            "The dataset contains no valid records."
        )

    return raw.reset_index(drop=True)


# ============================================================
# MODEL LOADING / TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_or_train_model():

    # --------------------------------------------------------
    # Try saved model first
    # --------------------------------------------------------

    if BUNDLE_PATH.exists():

        try:

            bundle = joblib.load(
                BUNDLE_PATH
            )

            required_keys = {
                "model",
                "tfidf_vectorizer",
                "feature_type",
            }

            if (
                isinstance(bundle, dict)
                and required_keys.issubset(
                    bundle.keys()
                )
            ):

                return bundle

        except Exception:
            pass

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    if (
        df.empty
        or df["status"].nunique() < 2
    ):

        raise ValueError(
            "The dataset must contain text "
            "and at least two categories."
        )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(
        df["clean_statement"]
    )

    y = df["status"]

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    # --------------------------------------------------------
    # Model bundle
    # --------------------------------------------------------

    bundle = {
        "model": model,
        "model_name": "Logistic Regression",
        "feature_type": "tfidf",
        "tfidf_vectorizer": vectorizer,
        "svd": None,
        "labels": sorted(
            y.unique()
        ),
    }

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    try:

        joblib.dump(
            bundle,
            BUNDLE_PATH,
        )

    except Exception:
        pass

    return bundle


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "# 🧠 Mental Health"
)

st.sidebar.markdown(
    "## Sentiment Analysis"
)

st.sidebar.caption(
    "Text Analysis Using Machine Learning and NLP"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**Navigate**"
)

page = st.sidebar.radio(
    "",
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


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_dataset()

except Exception as exc:

    st.error(
        f"Could not load the dataset: {exc}"
    )

    st.info(
        "Please check the Google Drive file ID, "
        "sharing permissions, and CSV columns."
    )

    st.stop()


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🧠 Mental Health Sentiment Analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-subtitle">
            Text Analysis Using Machine Learning and NLP
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    with st.container():

        st.markdown(
            """
            <div class="hero-box">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### 🧠 Welcome to Mental Health Sentiment Analysis
            """
        )

        st.markdown(
            """
            This application demonstrates text classification
            using **TF-IDF** and **Logistic Regression**.
            """
        )

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records after cleaning",
        f"{len(df):,}",
    )

    c2.metric(
        "Dataset categories",
        f"{df['status'].nunique():,}",
    )

    c3.metric(
        "Average words per statement",
        f"{df['word_count'].mean():.1f}",
    )

    st.markdown("")

    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.header("How it works")

    st.markdown(
        """
        **1. Enter a statement**

        Go to the **Predict** page and enter a text statement.

        **2. Text preprocessing**

        The application converts the text to lowercase,
        removes URLs and normalizes whitespace.

        **3. TF-IDF**

        TF-IDF converts the statement into numerical
        features that the machine-learning model can use.

        **4. Logistic Regression**

        The trained model predicts one of the categories
        contained in the dataset.
        """
    )

    st.info(
        "The first prediction may take some time because "
        "the model may need to be trained. A saved model "
        "will be reused when available."
    )


# ============================================================
# PREDICT PAGE
# ============================================================

elif page == "Predict":

    st.title("💬 Predict Mental Health Category")

    st.caption(
        "Enter a statement and the machine-learning model "
        "will classify it."
    )

    statement = st.text_area(
        "Your statement",
        placeholder=(
            "Example: I have been feeling very stressed "
            "and unable to sleep lately..."
        ),
        height=180,
        max_chars=5000,
    )

    st.markdown("")

    if st.button(
        "🔍 Predict Category",
        use_container_width=True,
    ):

        if not statement.strip():

            st.warning(
                "Please enter a statement first."
            )

        else:

            try:

                with st.spinner(
                    "Loading or training the model..."
                ):

                    bundle = load_or_train_model()

                    cleaned = clean_text(
                        statement
                    )

                    features = (
                        bundle[
                            "tfidf_vectorizer"
                        ].transform(
                            [cleaned]
                        )
                    )

                    prediction = (
                        bundle[
                            "model"
                        ].predict(
                            features
                        )[0]
                    )

                st.success(
                    f"Predicted dataset category: **{prediction}**"
                )

                st.caption(
                    "Model used: "
                    f"{bundle.get(
                        'model_name',
                        'Logistic Regression'
                    )}"
                )

                st.warning(
                    "This is a text-classification prediction, "
                    "not a medical diagnosis. It may be incorrect "
                    "and must not be used to make health decisions."
                )

            except Exception as exc:

                st.error(
                    f"Prediction could not be completed: {exc}"
                )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.caption(
        "Summary of the cleaned dataset used by the application."
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records",
        f"{len(df):,}",
    )

    c2.metric(
        "Categories",
        f"{df['status'].nunique():,}",
    )

    c3.metric(
        "Unique statements",
        f"{df['statement'].nunique():,}",
    )

    c4.metric(
        "Average words",
        f"{df['word_count'].mean():.1f}",
    )

    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    st.header("Category Distribution")

    counts = (
        df["status"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(
            name="Count"
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=counts,
        x="Count",
        y="Category",
        ax=ax,
    )

    ax.set_xlabel(
        "Number of statements"
    )

    ax.set_ylabel("")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # Sample records
    # --------------------------------------------------------

    st.header("Sample Records")

    st.dataframe(
        df[
            [
                "statement",
                "status",
                "word_count",
            ]
        ].head(15),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.title("📈 Model Information")

    st.markdown(
        """
        ### Selected Model

        **Logistic Regression**

        ### Text Representation

        **TF-IDF**

        The application uses:

        - Unigrams
        - Bigrams
        - Maximum 12,000 features
        - Sublinear TF scaling
        - Unicode accent stripping

        ### Model Training

        Logistic Regression is trained using the cleaned
        dataset and balanced class weights.

        ### Recorded Notebook Results

        - Accuracy: **0.77**
        - Macro F1-score: **0.71**
        - Weighted F1-score: **0.77**
        """
    )

    st.info(
        "The model trained by this application uses the full "
        "cleaned dataset. The metrics above are recorded "
        "notebook results and are not a new evaluation of "
        "the application-trained model."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.caption(
        "Recorded held-out test results from the original notebook."
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
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
        ]
    )

    st.header(
        "Comparison of Six Machine-Learning Models"
    )

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision (Macro)": "{:.4f}",
                "Recall (Macro)": "{:.4f}",
                "F1 Score (Macro)": "{:.4f}",
                "F1 Score (Weighted)": "{:.4f}",
                "Training Time (seconds)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # F1 chart
    # --------------------------------------------------------

    st.header("Macro F1-Score Comparison")

    chart_df = (
        comparison[
            [
                "Model",
                "F1 Score (Macro)",
            ]
        ]
        .set_index("Model")
    )

    st.bar_chart(
        chart_df
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    st.header(
        "Classification Report — Logistic Regression"
    )

    report = pd.DataFrame(
        [
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
        ]
    )

    st.dataframe(
        report.style.format(
            {
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-score": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    st.info(
        "Overall test results: "
        "Accuracy 0.77 · "
        "Macro F1-score 0.71 · "
        "Weighted F1-score 0.77 · "
        "Test samples 10,218"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.header(
        "Confusion Matrix — Logistic Regression"
    )

    cm_path = (
        BASE_DIR / "confusion_matrix.png"
    )

    if cm_path.exists():

        st.image(
            str(cm_path),
            caption=(
                "Confusion matrix from the notebook's "
                "held-out test set"
            ),
            use_container_width=True,
        )

    else:

        st.info(
            "confusion_matrix.png was not found in the "
            "project folder. Add it beside app.py if you "
            "want it displayed here."
        )

    st.warning(
        "These values are recorded notebook results, not "
        "a medical assessment. Predictions must not be "
        "used to diagnose mental health conditions."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title(
        "ℹ️ About Mental Health Sentiment Analysis"
    )

    st.markdown(
        """
        ### Project

        **Mental Health Sentiment Analysis**

        ### Description

        Text Analysis Using Machine Learning and NLP.

        ### Technologies

        - Python
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Streamlit

        ### Purpose

        This project demonstrates supervised text
        classification using the labels contained in
        the dataset.

        ### Dataset

        The application uses the cleaned
        **Combined Data.csv** dataset.

        ### Machine Learning

        TF-IDF is used to convert text into numerical
        features, followed by Logistic Regression for
        classification.
        """
    )

    st.warning(
        "This project is for educational demonstration "
        "only. It does not diagnose mental health conditions "
        "and does not replace assessment by a qualified "
        "professional."
    )

    st.markdown("### ⚠️ Important")

    st.write(
        "If someone is experiencing a mental health crisis "
        "or is in immediate danger, they should seek help "
        "from an appropriate qualified professional or "
        "local emergency service."
    )

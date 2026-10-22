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
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "Combined Data.csv"

BUNDLE_PATH = BASE_DIR / "mental_health_model_bundle.joblib"

# Google Drive file ID
GOOGLE_DRIVE_FILE_ID = "1bo432nn1YkbnS7G8QChCtJgZcueyUQE9"


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f7ff 0%,
            #f5fbff 100%
        );
    }

    [data-testid="stSidebar"] {
        background: #f0efff;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #25275f;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 19px;
        color: #666b85;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 20px;
        border-radius: 15px;
        background: #ffffff;
        border: 1px solid #e0defb;
        margin-bottom: 20px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: #625cf6;
        color: white;
        border: 0;
    }

    div.stButton > button:hover {
        background: #514be5;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean input text before sending it to the model.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download dataset if it does not exist
    # --------------------------------------------------------

    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:

        if not GOOGLE_DRIVE_FILE_ID:
            raise RuntimeError(
                "Google Drive file ID is missing."
            )

        try:

            with st.spinner(
                "Downloading dataset from Google Drive..."
            ):

                result = gdown.download(
                    id=GOOGLE_DRIVE_FILE_ID,
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
                "Please check the Google Drive file ID "
                "and sharing permissions."
            )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        raw = pd.read_csv(
            DATA_PATH
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not read the downloaded file as a CSV. "
            "Make sure the Google Drive file is the correct CSV."
        ) from exc

    # --------------------------------------------------------
    # Remove unnecessary index column
    # --------------------------------------------------------

    raw = raw.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {
        "statement",
        "status"
    }

    if not required_columns.issubset(
        raw.columns
    ):

        raise ValueError(
            "The CSV must contain the columns "
            "'statement' and 'status'. "
            f"Columns found: {list(raw.columns)}"
        )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    raw = raw.dropna(
        subset=[
            "statement",
            "status"
        ]
    ).copy()

    # Convert to strings

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

    # --------------------------------------------------------
    # Remove empty rows
    # --------------------------------------------------------

    raw = raw[
        (raw["statement"] != "")
        &
        (raw["status"] != "")
    ].copy()

    # --------------------------------------------------------
    # Remove duplicate records
    # --------------------------------------------------------

    raw = raw.drop_duplicates(
        subset=[
            "statement",
            "status"
        ]
    ).copy()

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    raw["clean_statement"] = (
        raw["statement"]
        .apply(clean_text)
    )

    # Remove empty cleaned statements

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

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if raw.empty:

        raise ValueError(
            "The dataset contains no valid records."
        )

    return raw.reset_index(
        drop=True
    )


# ============================================================
# MODEL LOADING / TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_or_train_model():

    # --------------------------------------------------------
    # Try loading saved model
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

            # If saved model cannot be loaded,
            # train a new model.
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
        y
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
            BUNDLE_PATH
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
        "Please check your Google Drive file ID, "
        "sharing permissions, and confirm that "
        "the CSV contains 'statement' and 'status' columns."
    )

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">'
        '🧠 Mental Health Sentiment Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Text Analysis Using Machine Learning and NLP'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        <b>Welcome!</b>

        <br><br>

        This application demonstrates text classification
        using TF-IDF and Logistic Regression.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records after cleaning",
        f"{len(df):,}"
    )

    c2.metric(
        "Dataset categories",
        f"{df['status'].nunique():,}"
    )

    c3.metric(
        "Average words per statement",
        f"{df['word_count'].mean():.1f}"
    )

    st.header(
        "How it works"
    )

    st.markdown(
        """
        **1. Enter a statement**

        Go to the **Predict** page and enter a text statement.

        **2. Text preprocessing**

        The text is converted to lowercase and URLs and
        unnecessary whitespace are removed.

        **3. TF-IDF**

        TF-IDF converts the text into numerical features.

        **4. Machine Learning**

        Logistic Regression predicts one of the categories
        found in the dataset.
        """
    )

    st.info(
        "The first prediction may take some time because "
        "the model may need to be trained. After training, "
        "the saved model can be reused."
    )


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    st.title(
        "💬 Predict Mental Health Category"
    )

    st.write(
        "Enter a text statement below and the machine-learning "
        "model will predict its dataset category."
    )

    st.markdown("---")

    statement = st.text_area(
        "Your statement",
        placeholder=(
            "Example: I have been feeling worried and "
            "unable to concentrate recently."
        ),
        height=180,
        max_chars=5000,
    )

    if st.button(
        "🔍 Predict Category",
        use_container_width=True
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
                        ]
                        .transform(
                            [cleaned]
                        )
                    )

                    prediction = (
                        bundle["model"]
                        .predict(features)[0]
                    )

                st.success(
                    f"Predicted dataset category: **{prediction}**"
                )

                st.caption(
                    "Model used: "
                    f"{bundle.get('model_name', 'Logistic Regression')}"
                )

                # ------------------------------------------------
                # Prediction probabilities
                # ------------------------------------------------

                model = bundle["model"]

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model
                        .predict_proba(
                            features
                        )[0]
                    )

                    classes = model.classes_

                    probability_df = pd.DataFrame(
                        {
                            "Category": classes,
                            "Probability": probabilities,
                        }
                    )

                    probability_df = (
                        probability_df
                        .sort_values(
                            "Probability",
                            ascending=False
                        )
                        .reset_index(
                            drop=True
                        )
                    )

                    probability_df[
                        "Probability"
                    ] = probability_df[
                        "Probability"
                    ].map(
                        lambda x: f"{x:.2%}"
                    )

                    st.subheader(
                        "Prediction probabilities"
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True,
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

    st.title(
        "📊 Dataset Overview"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records after cleaning",
        f"{len(df):,}"
    )

    c2.metric(
        "Categories",
        f"{df['status'].nunique():,}"
    )

    c3.metric(
        "Unique statements",
        f"{df['statement'].nunique():,}"
    )

    c4.metric(
        "Average words",
        f"{df['word_count'].mean():.1f}"
    )

    st.subheader(
        "Category Distribution"
    )

    # --------------------------------------------------------
    # Category counts
    # --------------------------------------------------------

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
        ax=ax
    )

    ax.set_xlabel(
        "Number of statements"
    )

    ax.set_ylabel("")

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Sample data
    # --------------------------------------------------------

    st.subheader(
        "Sample Records"
    )

    st.dataframe(
        df[
            [
                "statement",
                "status",
                "word_count"
            ]
        ].head(15),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.title(
        "📈 Model Information"
    )

    st.subheader(
        "Selected Model"
    )

    st.write(
        "Logistic Regression"
    )

    st.subheader(
        "Text Representation"
    )

    st.write(
        "TF-IDF using unigrams and bigrams."
    )

    st.subheader(
        "Model Configuration"
    )

    model_config = pd.DataFrame(
        [
            {
                "Parameter": "Algorithm",
                "Value": "Logistic Regression",
            },
            {
                "Parameter": "Text representation",
                "Value": "TF-IDF",
            },
            {
                "Parameter": "N-gram range",
                "Value": "(1, 2)",
            },
            {
                "Parameter": "Maximum features",
                "Value": "12,000",
            },
            {
                "Parameter": "Minimum document frequency",
                "Value": "2",
            },
            {
                "Parameter": "Class weight",
                "Value": "balanced",
            },
            {
                "Parameter": "Maximum iterations",
                "Value": "1,000",
            },
        ]
    )

    st.dataframe(
        model_config,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Recorded Notebook Results"
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Accuracy",
        "0.77"
    )

    r2.metric(
        "Macro F1-score",
        "0.71"
    )

    r3.metric(
        "Weighted F1-score",
        "0.77"
    )

    st.info(
        "These metrics are the recorded held-out test results "
        "from the original notebook. The model trained by this "
        "Streamlit app uses the full cleaned dataset, so these "
        "values should not be interpreted as a fresh evaluation "
        "of the deployed model."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title(
        "📊 Model Evaluation"
    )

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

    st.subheader(
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
    # Macro F1 chart
    # --------------------------------------------------------

    st.subheader(
        "Macro F1-score Comparison"
    )

    chart_df = (
        comparison[
            [
                "Model",
                "F1 Score (Macro)"
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

    st.subheader(
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

    st.markdown(
        """
        **Overall test results**

        - Accuracy: **0.77**
        - Macro F1-score: **0.71**
        - Weighted F1-score: **0.77**
        - Test samples: **10,218**
        """
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix — Logistic Regression"
    )

    cm_path = (
        BASE_DIR /
        "confusion_matrix.png"
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
            "The confusion matrix image was not found. "
            "Add 'confusion_matrix.png' beside app.py "
            "if you want to display it."
        )

    st.warning(
        "These are machine-learning evaluation results, "
        "not a medical assessment. Predictions must not "
        "be used to diagnose mental health conditions."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title(
        "ℹ️ About Mental Health Sentiment Analysis"
    )

    st.subheader(
        "Project"
    )

    st.write(
        "Mental Health Sentiment Analysis"
    )

    st.subheader(
        "Description"
    )

    st.write(
        "Text Analysis Using Machine Learning and NLP"
    )

    st.subheader(
        "Technologies"
    )

    st.markdown(
        """
        - Python
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Streamlit
        - Matplotlib
        - Seaborn
        """
    )

    st.subheader(
        "Purpose"
    )

    st.write(
        "The purpose of this project is to demonstrate "
        "supervised text classification using the labels "
        "present in the dataset."
    )

    st.warning(
        "This project is for educational demonstration only. "
        "It does not diagnose mental health conditions and "
        "does not replace assessment by a qualified professional."
    )

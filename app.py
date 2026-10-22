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

# Your Google Drive file ID
GOOGLE_DRIVE_FILE_ID = "1bo432nn1YkbnS7G8QChCtJgZcueyUQE9"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       MAIN APP
    ----------------------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f7ff 0%,
            #f5fbff 100%
        );
    }


    /* -----------------------------
       SIDEBAR
    ----------------------------- */

    [data-testid="stSidebar"] {
        background: #f0efff;
    }

    [data-testid="stSidebar"] h1 {
        color: #25275f;
        font-size: 1.55rem;
    }


    /* -----------------------------
       HERO
    ----------------------------- */

    .hero {
        padding: 2.7rem 2.8rem;
        border-radius: 24px;

        background: linear-gradient(
            120deg,
            #e9e7ff,
            #e5f7f4
        );

        border: 1px solid #dedcf5;

        margin-bottom: 2rem;

        box-shadow: 0 8px 25px rgba(60, 60, 120, 0.06);
    }


    .hero-title {
        font-size: 3.15rem;
        font-weight: 800;
        line-height: 1.15;
        color: #25275f;
        margin-bottom: 1rem;
    }


    .hero-icon {
        font-size: 3.5rem;
        margin-right: 0.45rem;
    }


    .hero-subtitle {
        font-size: 1.25rem;
        color: #5f6280;
        margin-top: 0.8rem;
    }


    /* -----------------------------
       SECTION TITLE
    ----------------------------- */

    .section-title {
        font-size: 2rem;
        font-weight: 750;
        color: #25275f;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }


    /* -----------------------------
       STEPS
    ----------------------------- */

    .steps {
        font-size: 1.1rem;
        line-height: 1.9;
        color: #34364a;
        padding-left: 1.5rem;
    }


    .steps li {
        margin-bottom: 0.45rem;
    }


    /* -----------------------------
       INFO CARDS
    ----------------------------- */

    .info-card {
        padding: 1.4rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e2e0f5;
        margin-bottom: 1rem;
    }


    .info-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #25275f;
        margin-bottom: 0.5rem;
    }


    .info-card-text {
        color: #55586f;
        line-height: 1.6;
    }


    /* -----------------------------
       PREDICTION RESULT
    ----------------------------- */

    .prediction-card {
        padding: 1.7rem;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            #eeeaff,
            #e9faf7
        );
        border: 1px solid #dddaf4;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }


    .prediction-title {
        font-size: 1rem;
        color: #666985;
        margin-bottom: 0.3rem;
    }


    .prediction-value {
        font-size: 2rem;
        font-weight: 800;
        color: #25275f;
    }


    /* -----------------------------
       BUTTON
    ----------------------------- */

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: #625cf6;
        color: white;
        border: 0;
        min-height: 45px;
    }


    div.stButton > button:hover {
        background: #514be4;
        color: white;
    }


    /* -----------------------------
       GENERAL HEADINGS
    ----------------------------- */

    h1, h2, h3 {
        color: #25275f;
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
    Clean a text statement before sending it to the model.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize whitespace
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

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DOWNLOAD DATASET IF IT DOES NOT EXIST
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

                # IMPORTANT:
                # Do NOT use fuzzy=True.
                #
                # This avoids:
                # download() got an unexpected keyword argument 'fuzzy'
                #
                result = gdown.download(
                    id=GOOGLE_DRIVE_FILE_ID,
                    output=str(DATA_PATH),
                    quiet=False
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
                "Check that the Google Drive file is shared "
                "and that the file ID is correct."
            )


    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    try:

        raw = pd.read_csv(
            DATA_PATH
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not read the downloaded file as a CSV. "
            "Make sure the Google Drive file is the correct "
            "Combined Data.csv file."
        ) from exc


    # --------------------------------------------------------
    # REMOVE UNNAMED INDEX COLUMN
    # --------------------------------------------------------

    raw = raw.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = {
        "statement",
        "status"
    }

    if not required_columns.issubset(
        raw.columns
    ):

        raise ValueError(
            "The CSV must contain 'statement' and 'status' columns. "
            f"Columns found: {list(raw.columns)}"
        )


    # --------------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------------

    raw = raw.dropna(
        subset=[
            "statement",
            "status"
        ]
    ).copy()


    # --------------------------------------------------------
    # CONVERT TO STRING
    # --------------------------------------------------------

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
    # REMOVE EMPTY VALUES
    # --------------------------------------------------------

    raw = raw[
        (raw["statement"] != "")
        &
        (raw["status"] != "")
    ].copy()


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    raw = raw.drop_duplicates(
        subset=[
            "statement",
            "status"
        ]
    ).copy()


    # --------------------------------------------------------
    # CLEAN STATEMENTS
    # --------------------------------------------------------

    raw["clean_statement"] = (
        raw["statement"]
        .apply(clean_text)
    )


    # Remove statements that became empty
    raw = raw[
        raw["clean_statement"].str.len() > 0
    ].copy()


    # --------------------------------------------------------
    # WORD COUNT
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


    return raw.reset_index(
        drop=True
    )


# ============================================================
# MODEL LOADING / TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_or_train_model():

    # --------------------------------------------------------
    # TRY SAVED MODEL FIRST
    # --------------------------------------------------------

    if BUNDLE_PATH.exists():

        try:

            bundle = joblib.load(
                BUNDLE_PATH
            )

            required_keys = {
                "model",
                "tfidf_vectorizer",
                "feature_type"
            }

            if (
                isinstance(bundle, dict)
                and required_keys.issubset(
                    bundle.keys()
                )
            ):

                return bundle

        except Exception:

            # If saved model is corrupted,
            # train a new one.
            pass


    # --------------------------------------------------------
    # LOAD DATASET
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
        strip_accents="unicode"
    )


    X = vectorizer.fit_transform(
        df["clean_statement"]
    )

    y = df["status"]


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


    model.fit(
        X,
        y
    )


    # --------------------------------------------------------
    # MODEL BUNDLE
    # --------------------------------------------------------

    bundle = {
        "model": model,
        "model_name": "Logistic Regression",
        "feature_type": "tfidf",
        "tfidf_vectorizer": vectorizer,
        "svd": None,
        "labels": sorted(
            y.unique()
        )
    }


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    try:

        joblib.dump(
            bundle,
            BUNDLE_PATH
        )

    except Exception:

        # App can still work even if
        # the environment doesn't allow writing.
        pass


    return bundle


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size: 1.55rem;
        font-weight: 800;
        color: #25275f;
        line-height: 1.2;
    ">
        🧠 Mental Health<br>
        Sentiment Analysis
    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(
    """
    <div style="
        color: #777991;
        font-size: 0.95rem;
        margin-top: 15px;
        line-height: 1.5;
    ">
        Text Analysis Using Machine Learning and NLP
    </div>
    """,
    unsafe_allow_html=True
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
        "About"
    ]
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
        "sharing permissions, and confirm that the "
        "file contains 'statement' and 'status' columns."
    )

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "Home":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                <span class="hero-icon">🧠</span>
                Welcome to Mental Health<br>
                Sentiment Analysis
            </div>

            <div class="hero-subtitle">
                Text Analysis Using Machine Learning and NLP
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Records after cleaning",
            f"{len(df):,}"
        )


    with c2:

        st.metric(
            "Dataset categories",
            f"{df['status'].nunique():,}"
        )


    with c3:

        st.metric(
            "Average words per statement",
            f"{df['word_count'].mean():.1f}"
        )


    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            How it works
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <ol class="steps">

            <li>
                Enter a text statement on the
                <b>Predict</b> page.
            </li>

            <li>
                The text is cleaned using the same
                preprocessing function.
            </li>

            <li>
                TF-IDF converts the text into
                numerical features.
            </li>

            <li>
                Logistic Regression predicts a
                category from the dataset.
            </li>

        </ol>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "The first prediction may take some time because "
        "the model may need to be trained. A saved model "
        "may be reused when available."
    )


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                💬 Enter a text statement
            </div>

            <div class="hero-subtitle">
                Analyze text using Machine Learning and NLP.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    statement = st.text_area(
        "Your statement",
        placeholder="Type a statement here...",
        height=180,
        max_chars=5000
    )


    if st.button(
        "Predict category",
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


                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="prediction-card">

                        <div class="prediction-title">
                            Predicted Dataset Category
                        </div>

                        <div class="prediction-value">
                            {prediction}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
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
                    "and should not be used to make health decisions."
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
    # STATISTICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Records after cleaning",
            f"{len(df):,}"
        )


    with c2:

        st.metric(
            "Categories",
            f"{df['status'].nunique():,}"
        )


    with c3:

        st.metric(
            "Unique statements",
            f"{df['statement'].nunique():,}"
        )


    with c4:

        st.metric(
            "Average words",
            f"{df['word_count'].mean():.1f}"
        )


    # --------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Category distribution"
    )


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
    # SAMPLE RECORDS
    # --------------------------------------------------------

    st.subheader(
        "Sample records"
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
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.title(
        "📈 Model Information"
    )


    st.markdown(
        """
        <div class="info-card">

            <div class="info-card-title">
                Selected Model
            </div>

            <div class="info-card-text">
                Logistic Regression
            </div>

        </div>


        <div class="info-card">

            <div class="info-card-title">
                Text Representation
            </div>

            <div class="info-card-text">
                TF-IDF using unigrams and bigrams.
            </div>

        </div>


        <div class="info-card">

            <div class="info-card-title">
                Model Selection
            </div>

            <div class="info-card-text">
                The original notebook selected Logistic
                Regression as the best model using Macro
                F1-score on its held-out test set.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "Recorded notebook results"
    )


    r1, r2, r3 = st.columns(3)


    with r1:

        st.metric(
            "Accuracy",
            "0.77"
        )


    with r2:

        st.metric(
            "Macro F1-score",
            "0.71"
        )


    with r3:

        st.metric(
            "Weighted F1-score",
            "0.77"
        )


    st.info(
        "The model trained by this app uses the full cleaned "
        "dataset. The metrics above are recorded notebook "
        "results and are not a new evaluation of this "
        "app-trained model."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title(
        "📊 Model Evaluation"
    )


    st.caption(
        "These are the recorded held-out test results "
        "from the notebook. They are displayed for "
        "reference; this page does not retrain or "
        "re-evaluate the models."
    )


    # --------------------------------------------------------
    # MODEL COMPARISON
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
                "Training Time (seconds)": 14.53
            },
            {
                "Model": "Linear SVM",
                "Accuracy": 0.7633,
                "Precision (Macro)": 0.7021,
                "Recall (Macro)": 0.6993,
                "F1 Score (Macro)": 0.6993,
                "F1 Score (Weighted)": 0.7610,
                "Training Time (seconds)": 10.06
            },
            {
                "Model": "Random Forest",
                "Accuracy": 0.7274,
                "Precision (Macro)": 0.7433,
                "Recall (Macro)": 0.5964,
                "F1 Score (Macro)": 0.6282,
                "F1 Score (Weighted)": 0.7209,
                "Training Time (seconds)": 34.78
            },
            {
                "Model": "HistGradientBoosting",
                "Accuracy": 0.7019,
                "Precision (Macro)": 0.5894,
                "Recall (Macro)": 0.5230,
                "F1 Score (Macro)": 0.5441,
                "F1 Score (Weighted)": 0.6910,
                "Training Time (seconds)": 3.79
            },
            {
                "Model": "Multinomial Naive Bayes",
                "Accuracy": 0.6951,
                "Precision (Macro)": 0.6316,
                "Recall (Macro)": 0.4720,
                "F1 Score (Macro)": 0.4949,
                "F1 Score (Weighted)": 0.6748,
                "Training Time (seconds)": 0.21
            },
            {
                "Model": "KNN",
                "Accuracy": 0.6491,
                "Precision (Macro)": 0.4980,
                "Recall (Macro)": 0.4676,
                "F1 Score (Macro)": 0.4704,
                "F1 Score (Weighted)": 0.6334,
                "Training Time (seconds)": 0.06
            }
        ]
    )


    st.subheader(
        "Comparison of six machine-learning models"
    )


    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision (Macro)": "{:.4f}",
                "Recall (Macro)": "{:.4f}",
                "F1 Score (Macro)": "{:.4f}",
                "F1 Score (Weighted)": "{:.4f}",
                "Training Time (seconds)": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MACRO F1 CHART
    # --------------------------------------------------------

    st.subheader(
        "Macro F1-score comparison"
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
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "Classification report — Logistic Regression"
    )


    report = pd.DataFrame(
        [
            {
                "Category": "Anxiety",
                "Precision": 0.75,
                "Recall": 0.82,
                "F1-score": 0.78,
                "Support": 725
            },
            {
                "Category": "Bipolar",
                "Precision": 0.73,
                "Recall": 0.78,
                "F1-score": 0.75,
                "Support": 500
            },
            {
                "Category": "Depression",
                "Precision": 0.80,
                "Recall": 0.62,
                "F1-score": 0.70,
                "Support": 3019
            },
            {
                "Category": "Normal",
                "Precision": 0.90,
                "Recall": 0.93,
                "F1-score": 0.91,
                "Support": 3208
            },
            {
                "Category": "Personality disorder",
                "Precision": 0.48,
                "Recall": 0.60,
                "F1-score": 0.53,
                "Support": 179
            },
            {
                "Category": "Stress",
                "Precision": 0.49,
                "Recall": 0.70,
                "F1-score": 0.58,
                "Support": 459
            },
            {
                "Category": "Suicidal",
                "Precision": 0.68,
                "Recall": 0.76,
                "F1-score": 0.72,
                "Support": 2128
            }
        ]
    )


    st.dataframe(
        report.style.format(
            {
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-score": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        """
        **Overall test results:**

        Accuracy **0.77** ·
        Macro F1-score **0.71** ·
        Weighted F1-score **0.77** ·
        Test samples **10,218**
        """
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "Confusion matrix — Logistic Regression"
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
            use_container_width=True
        )

    else:

        st.info(
            "The confusion matrix image is not in the "
            "project folder. Add confusion_matrix.png "
            "beside app.py to display it."
        )


    st.warning(
        "These values are recorded notebook results, "
        "not a medical assessment. Predictions must "
        "not be used to diagnose mental health conditions."
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
        <div class="info-card">

            <div class="info-card-title">
                Project
            </div>

            <div class="info-card-text">
                Mental Health Sentiment Analysis
            </div>

        </div>


        <div class="info-card">

            <div class="info-card-title">
                Description
            </div>

            <div class="info-card-text">
                Text Analysis Using Machine Learning and NLP
            </div>

        </div>


        <div class="info-card">

            <div class="info-card-title">
                Technologies
            </div>

            <div class="info-card-text">
                Python, Pandas, Scikit-learn, TF-IDF,
                Logistic Regression, Streamlit and Seaborn.
            </div>

        </div>


        <div class="info-card">

            <div class="info-card-title">
                Purpose
            </div>

            <div class="info-card-text">
                Demonstrate supervised text classification
                using the labels present in the dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.warning(
        "This project is for educational demonstration only. "
        "It does not diagnose mental health conditions or "
        "replace assessment by a qualified professional."
    )

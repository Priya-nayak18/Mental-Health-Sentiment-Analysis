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


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "Combined Data.csv"

BUNDLE_PATH = BASE_DIR / "mental_health_model_bundle.joblib"

CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"


# ============================================================
# GOOGLE DRIVE DATASET
# ============================================================

GOOGLE_DRIVE_FILE_ID = "1bo432nn1YkbnS7G8QChCtJgZcueyUQE9"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN APPLICATION
    -------------------------------------------------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f7ff 0%,
            #f5fbff 100%
        );
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: #f0efff;
        border-right: 1px solid #e2e0f5;
    }

    [data-testid="stSidebar"] h1 {
        color: #25275f;
        font-weight: 800;
    }

    [data-testid="stSidebar"] p {
        color: #73758c;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        padding: 2.6rem 2.4rem;
        border-radius: 22px;

        background:
            linear-gradient(
                120deg,
                #e9e7ff 0%,
                #e5f7f4 100%
            );

        border: 1px solid #dfddf5;

        margin-bottom: 2rem;

        box-shadow:
            0 8px 30px rgba(45, 43, 90, 0.06);
    }


    .hero-title {
        font-size: 3.1rem;
        line-height: 1.12;
        font-weight: 800;
        color: #25275f;
        margin-bottom: 1rem;
    }


    .hero-icon {
        font-size: 3.1rem;
        vertical-align: middle;
        margin-right: 0.35rem;
    }


    .hero-subtitle {
        font-size: 1.25rem;
        color: #5e617d;
        margin-top: 0.7rem;
    }


    /* --------------------------------------------------------
       PAGE HEADER
    -------------------------------------------------------- */

    .page-header {
        margin-bottom: 1.5rem;
    }

    .page-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #25275f;
        margin-bottom: 0.4rem;
    }

    .page-subtitle {
        font-size: 1.05rem;
        color: #70738b;
    }


    /* --------------------------------------------------------
       CARDS
    -------------------------------------------------------- */

    .info-card {
        background: white;
        border: 1px solid #e2e0f5;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;

        box-shadow:
            0 5px 20px rgba(45, 43, 90, 0.04);
    }


    .info-card h3 {
        color: #292b68;
        margin-top: 0;
    }


    .info-card p {
        color: #55586e;
        line-height: 1.7;
    }


    /* --------------------------------------------------------
       PREDICTION RESULT
    -------------------------------------------------------- */

    .prediction-card {
        background:
            linear-gradient(
                135deg,
                #eeeeff,
                #e7faf6
            );

        border: 1px solid #dcdcf7;

        border-radius: 18px;

        padding: 1.8rem;

        margin-top: 1.2rem;

        text-align: center;
    }


    .prediction-label {
        color: #6c6e86;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }


    .prediction-value {
        color: #292b68;
        font-size: 2.2rem;
        font-weight: 800;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    div.stButton > button {
        border-radius: 11px;
        font-weight: 700;
        background: #625cf6;
        color: white;
        border: none;
        padding: 0.65rem 1.2rem;
    }


    div.stButton > button:hover {
        background: #514be5;
        color: white;
    }


    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid #e5e3f3;
        padding: 1rem;
        border-radius: 15px;
    }


    /* --------------------------------------------------------
       TEXT AREA
    -------------------------------------------------------- */

    textarea {
        border-radius: 12px !important;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid #dddbea;
        color: #77798e;
        font-size: 0.9rem;
        text-align: center;
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
    Basic preprocessing used by the application.
    """

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# DOWNLOAD DATASET FROM GOOGLE DRIVE
# ============================================================

def download_dataset():
    """
    Download the CSV from Google Drive if it does not exist locally.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # If the CSV already exists and is not empty, use it.
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        return

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
            # New versions of gdown may not support it.
            result = gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=str(DATA_PATH),
                quiet=False
            )

        if result is None:
            raise RuntimeError(
                "gdown could not download the file."
            )

        if not DATA_PATH.exists():
            raise RuntimeError(
                "The downloaded dataset file was not created."
            )

        if DATA_PATH.stat().st_size == 0:
            raise RuntimeError(
                "The downloaded dataset file is empty."
            )

    except Exception as exc:

        raise RuntimeError(
            f"Dataset download failed: {exc}"
        ) from exc


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    download_dataset()

    try:

        raw = pd.read_csv(
            DATA_PATH
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not read the downloaded file as a CSV. "
            "Please make sure the Google Drive file is actually "
            "a CSV dataset."
        ) from exc


    # Remove unwanted index column if present.
    raw = raw.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )


    # Required columns.
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


    # Remove missing rows.
    raw = raw.dropna(
        subset=[
            "statement",
            "status"
        ]
    ).copy()


    # Convert to strings.
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


    # Remove empty rows.
    raw = raw[
        (raw["statement"] != "")
        &
        (raw["status"] != "")
    ].copy()


    # Remove duplicate records.
    raw = raw.drop_duplicates(
        subset=[
            "statement",
            "status"
        ]
    ).copy()


    # Clean text.
    raw["clean_statement"] = (
        raw["statement"]
        .apply(clean_text)
    )


    # Remove empty cleaned statements.
    raw = raw[
        raw["clean_statement"].str.len() > 0
    ].copy()


    # Word count.
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
    # Try loading saved model first.
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

            # Invalid model.
            # Train again below.
            pass


    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    df = load_dataset()


    if (
        df.empty
        or
        df["status"].nunique() < 2
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
    # Save model bundle
    # --------------------------------------------------------

    bundle = {

        "model": model,

        "model_name":
            "Logistic Regression",

        "feature_type":
            "tfidf",

        "tfidf_vectorizer":
            vectorizer,

        "svd":
            None,

        "labels":
            sorted(
                y.unique()
            ),
    }


    try:

        joblib.dump(
            bundle,
            BUNDLE_PATH
        )

    except Exception:

        pass


    return bundle


# ============================================================
# HELPER FUNCTIONS FOR UI
# ============================================================

def page_header(
    title,
    subtitle=""
):

    st.markdown(
        f"""
        <div class="page-header">

            <div class="page-title">
                {title}
            </div>

            <div class="page-subtitle">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def footer():

    st.markdown(
        """
        <div class="footer">
            🧠 Mental Health Sentiment Analysis
            <br>
            Educational use only — not a medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding: 0.5rem 0 1rem 0;
    ">

        <div style="
            font-size: 1.8rem;
            font-weight: 800;
            color: #25275f;
        ">
            🧠 Mental Health
        </div>

        <div style="
            font-size: 1.8rem;
            font-weight: 800;
            color: #25275f;
        ">
            Sentiment Analysis
        </div>

        <div style="
            color: #77798e;
            margin-top: 0.8rem;
            line-height: 1.5;
        ">
            Text Analysis Using Machine Learning and NLP
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    <div style="
        font-size: 1rem;
        color: #4d5065;
        margin-bottom: 0.5rem;
    ">
        Navigate
    </div>
    """,
    unsafe_allow_html=True
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
    label_visibility="collapsed"
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    <div style="
        color: #77798e;
        line-height: 1.6;
        font-size: 0.9rem;
    ">
        Educational use only —
        not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
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
        "Make sure the Google Drive file is shared so that "
        "the application can access it. "
        "The configured Google Drive file ID is:"
    )

    st.code(
        GOOGLE_DRIVE_FILE_ID
    )

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "Home":

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


    # Metrics.
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


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-card">

            <h3>How it works</h3>

            <p>
                <b>1. Enter a statement</b><br>
                Go to the Predict page and enter a text statement.
            </p>

            <p>
                <b>2. Text preprocessing</b><br>
                The text is cleaned before classification.
            </p>

            <p>
                <b>3. TF-IDF transformation</b><br>
                The text is converted into numerical features
                using TF-IDF.
            </p>

            <p>
                <b>4. Machine-learning prediction</b><br>
                Logistic Regression predicts one of the
                categories available in the dataset.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "The model is trained automatically on the first prediction "
        "and saved as a joblib file. Later predictions reuse the "
        "saved model."
    )


    footer()


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                💬 Enter a Text Statement
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
        placeholder=(
            "Type a statement here..."
        ),
        height=180,
        max_chars=5000
    )


    st.caption(
        "Enter a text statement and click Predict Category."
    )


    if st.button(
        "Predict Category",
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

                    bundle = (
                        load_or_train_model()
                    )

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


                # Result card.
                st.markdown(
                    f"""
                    <div class="prediction-card">

                        <div class="prediction-label">
                            Predicted Dataset Category
                        </div>

                        <div class="prediction-value">
                            {prediction}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    "<br>",
                    unsafe_allow_html=True
                )


                st.success(
                    "Prediction completed successfully."
                )


                st.caption(
                    "Model used: "
                    +
                    bundle.get(
                        "model_name",
                        "Logistic Regression"
                    )
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


    footer()


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    page_header(
        "📊 Dataset Overview",
        "Explore the dataset used by the application."
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Records",
        f"{len(df):,}"
    )


    c2.metric(
        "Categories",
        f"{df['status'].nunique():,}"
    )


    c3.metric(
        "Unique Statements",
        f"{df['statement'].nunique():,}"
    )


    c4.metric(
        "Average Words",
        f"{df['word_count'].mean():.1f}"
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.subheader(
        "Category Distribution"
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
        "Number of Statements"
    )

    ax.set_ylabel("")


    fig.tight_layout()


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


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
        hide_index=True
    )


    footer()


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    page_header(
        "📈 Model Information",
        "Information about the machine-learning model."
    )


    st.markdown(
        """
        <div class="info-card">

            <h3>Selected Model</h3>

            <p>
                <b>Logistic Regression</b>
            </p>

            <p>
                Logistic Regression is used as the final
                text-classification model.
            </p>

        </div>


        <div class="info-card">

            <h3>Text Representation</h3>

            <p>
                <b>TF-IDF</b> with unigrams and bigrams.
            </p>

            <p>
                The vectorizer uses a maximum of 12,000 features,
                minimum document frequency of 2, sublinear TF
                scaling, and Unicode accent stripping.
            </p>

        </div>


        <div class="info-card">

            <h3>Model Configuration</h3>

            <p>
                Maximum iterations: <b>1000</b>
            </p>

            <p>
                Class weighting: <b>Balanced</b>
            </p>

            <p>
                Random state: <b>42</b>
            </p>

        </div>


        <div class="info-card">

            <h3>Recorded Notebook Results</h3>

            <p>
                Accuracy: <b>0.77</b>
            </p>

            <p>
                Macro F1-score: <b>0.71</b>
            </p>

            <p>
                Weighted F1-score: <b>0.77</b>
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        "The recorded metrics above are the evaluation results "
        "from the notebook. The application trains its model "
        "using the cleaned dataset."
    )


    footer()


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    page_header(
        "📊 Model Evaluation",
        "Comparison of the machine-learning models."
    )


    st.caption(
        "These are the recorded held-out test results "
        "from the Jupyter Notebook."
    )


    comparison = pd.DataFrame(
        [

            {
                "Model":
                    "Logistic Regression",

                "Accuracy":
                    0.7696,

                "Precision (Macro)":
                    0.6895,

                "Recall (Macro)":
                    0.7448,

                "F1 Score (Macro)":
                    0.7108,

                "F1 Score (Weighted)":
                    0.7695,

                "Training Time (seconds)":
                    14.53,
            },


            {
                "Model":
                    "Linear SVM",

                "Accuracy":
                    0.7633,

                "Precision (Macro)":
                    0.7021,

                "Recall (Macro)":
                    0.6993,

                "F1 Score (Macro)":
                    0.6993,

                "F1 Score (Weighted)":
                    0.7610,

                "Training Time (seconds)":
                    10.06,
            },


            {
                "Model":
                    "Random Forest",

                "Accuracy":
                    0.7274,

                "Precision (Macro)":
                    0.7433,

                "Recall (Macro)":
                    0.5964,

                "F1 Score (Macro)":
                    0.6282,

                "F1 Score (Weighted)":
                    0.7209,

                "Training Time (seconds)":
                    34.78,
            },


            {
                "Model":
                    "HistGradientBoosting",

                "Accuracy":
                    0.7019,

                "Precision (Macro)":
                    0.5894,

                "Recall (Macro)":
                    0.5230,

                "F1 Score (Macro)":
                    0.5441,

                "F1 Score (Weighted)":
                    0.6910,

                "Training Time (seconds)":
                    3.79,
            },


            {
                "Model":
                    "Multinomial Naive Bayes",

                "Accuracy":
                    0.6951,

                "Precision (Macro)":
                    0.6316,

                "Recall (Macro)":
                    0.4720,

                "F1 Score (Macro)":
                    0.4949,

                "F1 Score (Weighted)":
                    0.6748,

                "Training Time (seconds)":
                    0.21,
            },


            {
                "Model":
                    "KNN",

                "Accuracy":
                    0.6491,

                "Precision (Macro)":
                    0.4980,

                "Recall (Macro)":
                    0.4676,

                "F1 Score (Macro)":
                    0.4704,

                "F1 Score (Weighted)":
                    0.6334,

                "Training Time (seconds)":
                    0.06,
            },

        ]
    )


    st.subheader(
        "Comparison of Six Machine-Learning Models"
    )


    st.dataframe(
        comparison.style.format(
            {
                "Accuracy":
                    "{:.4f}",

                "Precision (Macro)":
                    "{:.4f}",

                "Recall (Macro)":
                    "{:.4f}",

                "F1 Score (Macro)":
                    "{:.4f}",

                "F1 Score (Weighted)":
                    "{:.4f}",

                "Training Time (seconds)":
                    "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Macro F1-Score Comparison"
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
                "Category":
                    "Anxiety",

                "Precision":
                    0.75,

                "Recall":
                    0.82,

                "F1-score":
                    0.78,

                "Support":
                    725,
            },


            {
                "Category":
                    "Bipolar",

                "Precision":
                    0.73,

                "Recall":
                    0.78,

                "F1-score":
                    0.75,

                "Support":
                    500,
            },


            {
                "Category":
                    "Depression",

                "Precision":
                    0.80,

                "Recall":
                    0.62,

                "F1-score":
                    0.70,

                "Support":
                    3019,
            },


            {
                "Category":
                    "Normal",

                "Precision":
                    0.90,

                "Recall":
                    0.93,

                "F1-score":
                    0.91,

                "Support":
                    3208,
            },


            {
                "Category":
                    "Personality disorder",

                "Precision":
                    0.48,

                "Recall":
                    0.60,

                "F1-score":
                    0.53,

                "Support":
                    179,
            },


            {
                "Category":
                    "Stress",

                "Precision":
                    0.49,

                "Recall":
                    0.70,

                "F1-score":
                    0.58,

                "Support":
                    459,
            },


            {
                "Category":
                    "Suicidal",

                "Precision":
                    0.68,

                "Recall":
                    0.76,

                "F1-score":
                    0.72,

                "Support":
                    2128,
            },

        ]
    )


    st.dataframe(
        report.style.format(
            {
                "Precision":
                    "{:.2f}",

                "Recall":
                    "{:.2f}",

                "F1-score":
                    "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        """
        **Overall test results**

        Accuracy: **0.77**

        Macro F1-score: **0.71**

        Weighted F1-score: **0.77**

        Test samples: **10,218**
        """
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix — Logistic Regression"
    )


    if CONFUSION_MATRIX_PATH.exists():

        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption=(
                "Confusion matrix from the notebook's "
                "held-out test set"
            ),
            use_container_width=True
        )

    else:

        st.info(
            "confusion_matrix.png was not found in the project "
            "folder. Add the image beside app.py if you want "
            "it displayed here."
        )


    st.warning(
        "These values are recorded notebook evaluation results. "
        "They are not a medical assessment."
    )


    footer()


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    page_header(
        "ℹ️ About",
        "About the Mental Health Sentiment Analysis project."
    )


    st.markdown(
        """
        <div class="info-card">

            <h3>🧠 Project</h3>

            <p>
                <b>Mental Health Sentiment Analysis</b>
            </p>

            <p>
                Text Analysis Using Machine Learning and NLP.
            </p>

        </div>


        <div class="info-card">

            <h3>🛠 Technologies</h3>

            <p>
                Python
            </p>

            <p>
                Pandas
            </p>

            <p>
                Scikit-learn
            </p>

            <p>
                TF-IDF
            </p>

            <p>
                Logistic Regression
            </p>

            <p>
                Streamlit
            </p>

        </div>


        <div class="info-card">

            <h3>🎯 Purpose</h3>

            <p>
                The purpose of this project is to demonstrate
                supervised text classification using the labels
                present in the dataset.
            </p>

        </div>


        <div class="info-card">

            <h3>📁 Dataset</h3>

            <p>
                The dataset is automatically downloaded from
                Google Drive when the local CSV file is not
                available.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.warning(
        "This project is for educational demonstration only. "
        "It does not diagnose mental health conditions and "
        "does not replace assessment by a qualified professional."
    )


    footer()

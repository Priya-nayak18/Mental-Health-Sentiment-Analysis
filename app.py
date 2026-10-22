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
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS AND GOOGLE DRIVE SETTINGS
# ============================================================

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

    /* --------------------------------------------------------
       MAIN APPLICATION
    -------------------------------------------------------- */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #f8f7ff 0%,
                #f5fbff 100%
            );
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #f0efff 0%,
                #f4f3ff 100%
            );
        border-right: 1px solid #e1def5;
    }

    [data-testid="stSidebar"] h1 {
        color: #25275f;
        font-weight: 700;
    }

    [data-testid="stSidebar"] p {
        color: #77798e;
    }


    /* --------------------------------------------------------
       SIDEBAR NAVIGATION
    -------------------------------------------------------- */

    [data-testid="stSidebar"] .stRadio label {
        font-size: 16px;
        color: #34364a;
    }


    /* --------------------------------------------------------
       HERO SECTION
    -------------------------------------------------------- */

    .hero-container {
        padding: 2.8rem 2.8rem;
        border-radius: 24px;

        background:
            linear-gradient(
                120deg,
                #e9e7ff 0%,
                #e5f7f4 100%
            );

        border: 1px solid #dedcf5;

        box-shadow:
            0 8px 25px rgba(50, 50, 100, 0.04);

        margin-bottom: 2rem;
    }


    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #292c6b;
        line-height: 1.15;
        margin-bottom: 0.8rem;
    }


    .hero-subtitle {
        font-size: 1.15rem;
        color: #555a73;
        margin-top: 0.5rem;
    }


    .hero-icon {
        font-size: 4rem;
        margin-right: 0.5rem;
    }


    /* --------------------------------------------------------
       PAGE TITLE
    -------------------------------------------------------- */

    .page-title {
        font-size: 2.4rem;
        font-weight: 750;
        color: #292c6b;
        margin-bottom: 0.4rem;
    }


    .page-subtitle {
        color: #6e7187;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }


    /* --------------------------------------------------------
       WELCOME CARD
    -------------------------------------------------------- */

    .welcome-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #dedcf5;
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.8rem;

        box-shadow:
            0 5px 20px rgba(50, 50, 100, 0.03);
    }


    .welcome-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #30334b;
        margin-bottom: 0.7rem;
    }


    .welcome-text {
        font-size: 1rem;
        color: #45485c;
        line-height: 1.7;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    div.stButton > button {
        border-radius: 10px;
        font-weight: 650;

        background:
            linear-gradient(
                90deg,
                #625cf6,
                #716cf8
            );

        color: white;
        border: none;

        padding: 0.7rem 1.2rem;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }


    div.stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 5px 15px rgba(98, 92, 246, 0.25);
    }


    /* --------------------------------------------------------
       TEXT AREA
    -------------------------------------------------------- */

    textarea {
        border-radius: 12px !important;
    }


    /* --------------------------------------------------------
       METRICS
    -------------------------------------------------------- */

    [data-testid="stMetric"] {
        background: transparent;
        padding: 0.4rem 0.2rem;
    }

    [data-testid="stMetricLabel"] {
        color: #55586c;
        font-size: 0.95rem;
    }

    [data-testid="stMetricValue"] {
        color: #30334b;
        font-weight: 600;
    }


    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* --------------------------------------------------------
       INFO / WARNING / SUCCESS BOXES
    -------------------------------------------------------- */

    .stAlert {
        border-radius: 12px;
    }


    /* --------------------------------------------------------
       SECTION HEADINGS
    -------------------------------------------------------- */

    h1, h2, h3 {
        color: #30334b;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #deddea;

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
    Basic text cleaning used before TF-IDF transformation.
    """

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
    )

    return text.strip()


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    DATA_DIR.mkdir(
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

        try:

            with st.spinner(
                "Downloading dataset from Google Drive..."
            ):

                # IMPORTANT:
                # Do NOT use fuzzy=True.
                # It causes an error with newer gdown versions.

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
            "Could not read the dataset as a CSV. "
            "Make sure the Google Drive file is the correct CSV."
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
            "status",
        ]
    ).copy()


    # --------------------------------------------------------
    # Convert columns to strings
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
    # Remove empty rows
    # --------------------------------------------------------

    raw = raw[
        (raw["statement"] != "")
        &
        (raw["status"] != "")
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
    # Try loading existing model
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
    """
    <div style="
        font-size: 2rem;
        margin-bottom: 0.2rem;
    ">
        🧠
    </div>

    <div style="
        font-size: 1.45rem;
        font-weight: 750;
        color: #25275f;
        line-height: 1.25;
    ">
        Mental Health<br>
        Sentiment Analysis
    </div>

    <div style="
        margin-top: 1rem;
        color: #77798e;
        font-size: 0.95rem;
        line-height: 1.6;
    ">
        Text Analysis Using Machine Learning and NLP
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    "<br>",
    unsafe_allow_html=True,
)


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


st.sidebar.markdown(
    "---"
)


st.sidebar.markdown(
    """
    <div style="
        color: #77798e;
        font-size: 0.9rem;
        line-height: 1.6;
    ">
        Educational use only — not a medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = load_dataset()

except Exception as exc:

    st.error(
        f"Could not load the dataset: {exc}"
    )

    st.info(
        "Please check your Google Drive file ID, "
        "sharing permissions, and confirm that the CSV "
        "contains 'statement' and 'status' columns."
    )

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "Home":

    # --------------------------------------------------------
    # Hero
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="hero-container">

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
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # Metrics
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


    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # How it works
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="page-title">
            How it works
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div style="
            font-size: 1.05rem;
            line-height: 2;
            color: #45485c;
        ">

        <ol>
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
                Logistic Regression predicts one
                of the dataset's existing categories.
            </li>
        </ol>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.info(
        "The first prediction may take some time because "
        "the model may need to be trained. Once trained, "
        "the saved model can be reused."
    )


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    st.markdown(
        """
        <div class="hero-container">

            <div class="hero-title">
                💬 Enter a text statement
            </div>

            <div class="hero-subtitle">
                Analyze text using Machine Learning and NLP.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    statement = st.text_area(
        "Your statement",
        placeholder=(
            "Type a statement here..."
        ),
        height=200,
        max_chars=5000,
    )


    if st.button(
        "Predict category",
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
                        bundle["model"]
                        .predict(features)[0]
                    )


                # ------------------------------------------------
                # Result
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Probability if available
                # ------------------------------------------------

                model = bundle["model"]


                if hasattr(
                    model,
                    "predict_proba",
                ):

                    probabilities = (
                        model.predict_proba(
                            features
                        )[0]
                    )


                    classes = (
                        model.classes_
                    )


                    probability_df = pd.DataFrame(
                        {
                            "Category":
                                classes,

                            "Probability":
                                probabilities,
                        }
                    ).sort_values(
                        "Probability",
                        ascending=False,
                    )


                    st.subheader(
                        "Prediction probabilities"
                    )


                    st.dataframe(
                        probability_df.style.format(
                            {
                                "Probability":
                                    "{:.2%}"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


                st.warning(
                    "This is a text-classification "
                    "prediction, not a medical diagnosis. "
                    "It may be incorrect and must not be "
                    "used to make health decisions."
                )


            except Exception as exc:

                st.error(
                    f"Prediction could not be completed: {exc}"
                )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.markdown(
        """
        <div class="page-title">
            📊 Dataset Overview
        </div>

        <div class="page-subtitle">
            Summary of the cleaned dataset used by the application.
        </div>
        """,
        unsafe_allow_html=True,
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Records after cleaning",
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
        figsize=(10, 5)
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


    st.pyplot(
        fig,
        use_container_width=True,
    )


    plt.close(fig)


    # --------------------------------------------------------
    # Sample records
    # --------------------------------------------------------

    st.subheader(
        "Sample records"
    )


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

    st.markdown(
        """
        <div class="page-title">
            📈 Model Information
        </div>

        <div class="page-subtitle">
            Machine-learning model and text representation used
            by the application.
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        - Unicode accent stripping
        - Sublinear TF scaling

        ### Model Configuration

        Logistic Regression is configured with:

        - `max_iter = 1000`
        - `class_weight = "balanced"`
        - `random_state = 42`

        ### Dataset Categories

        The model predicts categories contained in the
        dataset's `status` column.
        """
    )


    st.info(
        "The application trains the Logistic Regression model "
        "using the cleaned dataset when a saved model bundle "
        "is not available."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.markdown(
        """
        <div class="page-title">
            📊 Model Evaluation
        </div>

        <div class="page-subtitle">
            Recorded held-out test results from the original
            notebook.
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.caption(
        "These values are displayed for reference. "
        "This page does not retrain or re-evaluate the models."
    )


    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

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
        "Comparison of six machine-learning models"
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

        hide_index=True,
    )


    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    st.subheader(
        "Macro F1-score comparison"
    )


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

    st.subheader(
        "Classification report — Logistic Regression"
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

        hide_index=True,
    )


    st.markdown(
        """
        **Overall test results**

        Accuracy: **0.77** ·
        Macro F1-score: **0.71** ·
        Weighted F1-score: **0.77** ·
        Test samples: **10,218**
        """
    )


    # --------------------------------------------------------
    # Confusion matrix
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

            use_container_width=True,
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

    st.markdown(
        """
        <div class="page-title">
            ℹ️ About Mental Health Sentiment Analysis
        </div>

        <div class="page-subtitle">
            Information about the project and its purpose.
        </div>
        """,
        unsafe_allow_html=True,
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
        - Matplotlib
        - Seaborn

        ### Purpose

        The purpose of this project is to demonstrate
        supervised text classification using the labels
        present in the dataset.

        The application converts text into TF-IDF numerical
        features and uses Logistic Regression to predict
        one of the categories found in the dataset.
        """
    )


    st.warning(
        "This project is for educational demonstration only. "
        "It does not diagnose mental health conditions and "
        "does not replace assessment by a qualified professional."
    )



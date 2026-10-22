# Mental Health Sentiment Analysis — Streamlit Project

This is a runnable Streamlit app for the educational project **Mental Health Sentiment Analysis and Bot Text Analysis Using Machine Learning and NLP**.

## Included
- `app.py` — Streamlit website
- `requirements.txt` — Python packages
- `data/Combined Data.csv` — dataset
- The model bundle is created automatically on the first prediction and saved as `mental_health_model_bundle.joblib`.

## Run on Windows
1. Extract this ZIP file.
2. Open the extracted `Mental_Health_Sentiment_Analysis` folder.
3. Click the File Explorer address bar, type `cmd`, and press Enter.
4. Run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `python` is not recognized, install Python 3.10 or newer and select **Add Python to PATH** during installation.

The first time you click **Predict category**, the app trains Logistic Regression on the included cleaned dataset. This may take several minutes. It then saves `mental_health_model_bundle.joblib`; subsequent predictions reuse the saved model.

## Deploy to Streamlit Community Cloud
1. Upload the project files to a GitHub repository.
2. In Streamlit Community Cloud, choose that repository and set the app file path to `app.py`.
3. Keep `requirements.txt` and the `data` folder in the repository.
4. Deploy. The model is trained on the first prediction, so the initial prediction may take time.

## Important
- The app uses the same text-cleaning function and TF-IDF settings as the uploaded notebook.
- The model is retrained on the full cleaned dataset for deployment. The metrics shown in Model Information are the notebook's held-out evaluation results, not new app evaluation results.
- This is an educational classifier, not a medical diagnostic tool. Predictions may be wrong and must not be used as clinical advice.

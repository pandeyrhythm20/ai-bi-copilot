import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AI BI Copilot",
    layout="wide"
)

# -----------------------------------
# APP TITLE
# -----------------------------------

st.title("AI Business Intelligence Copilot")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Business Dataset (CSV)",
    type=["csv"]
)

# -----------------------------------
# MAIN APPLICATION
# -----------------------------------

if uploaded_file:

    # -----------------------------------
    # READ CSV
    # -----------------------------------

    df = pd.read_csv(uploaded_file)

    # -----------------------------------
    # DATA CLEANING
    # -----------------------------------

    df = df.dropna().head(5000)

    # -----------------------------------
    # DATA PREVIEW
    # -----------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # -----------------------------------
    # DATA TYPES
    # -----------------------------------

    st.subheader("Dataset Information")

    st.write(df.dtypes)

    # -----------------------------------
    # DATASET SHAPE
    # -----------------------------------

    st.subheader("Dataset Shape")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # -----------------------------------
    # STATISTICAL SUMMARY
    # -----------------------------------

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    # -----------------------------------
    # TARGET COLUMN SELECTION
    # -----------------------------------

    st.subheader("Target Column Selection")

    valid_target_columns = [
    col for col in df.columns
    if df[col].nunique() <= 10
]

    target_column = st.selectbox(
    "Select Target Column for Prediction",
    valid_target_columns
)

    # -----------------------------------
    # CORRELATION MATRIX
    # -----------------------------------

    st.subheader("Correlation Matrix")

    correlation = df.corr(numeric_only=True)

    st.dataframe(correlation)

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------

    numeric_cols = df.select_dtypes(
        include='number'
    ).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select Column for Visualization",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_col,
            title=f"Distribution of {selected_col}"
        )

        st.plotly_chart(fig)

       # -----------------------------------
    # MACHINE LEARNING DATAFRAME
    # -----------------------------------

    df_ml = df.copy()

    # -----------------------------------
    # REMOVE HIGH CARDINALITY COLUMNS
    # -----------------------------------

    for col in df_ml.columns:

        if df_ml[col].nunique() > 100:

            df_ml = df_ml.drop(columns=[col])

    # -----------------------------------
    # LABEL ENCODING
    # -----------------------------------

    label_encoders = {}

    for col in df_ml.select_dtypes(
        include='object'
    ).columns:

        le = LabelEncoder()

        df_ml[col] = le.fit_transform(df_ml[col])

        label_encoders[col] = le

    # -----------------------------------
    # FEATURES AND TARGET
    # -----------------------------------

    X = df_ml.drop(columns=[target_column])

    y = df_ml[target_column]

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    st.subheader("ML Feature Information")

    st.write(f"Number of Features: {X.shape[1]}")
    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------
    model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42
)
    model.fit(X_train, y_train)

    # -----------------------------------
    # PREDICTIONS
    # -----------------------------------

    predictions = model.predict(X_test)

    # -----------------------------------
    # MODEL ACCURACY
    # -----------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.subheader("Model Accuracy")

    st.write(f"Accuracy: {accuracy:.2f}")

    # -----------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("Feature Importance")

    st.dataframe(importance_df)

    # -----------------------------------
    # AI GENERATED BUSINESS INSIGHTS
    # -----------------------------------

    top_features = importance_df.head(5)

    prompt = f"""
    You are a senior business analyst and executive advisor.

    Analyze this machine learning model output and provide strategic business recommendations.

    **Model Performance:**
    - Accuracy: {accuracy:.2%}
    - Number of Features Analyzed: {X.shape[1]}
    - Dataset Size: {X.shape[0]} records

    **Top 5 Most Important Features:**
    {top_features.to_string(index=False)}

    Please provide in a clear, executive-friendly format:
    1. **Key Business Insights** - What does this model reveal about our business?
    2. **Risk Analysis** - What are the top risks we should address?
    3. **Opportunities** - What growth opportunities exist based on this analysis?
    4. **Actionable Recommendations** - What specific actions should we take?
    5. **Executive Summary** - One paragraph summary for C-suite stakeholders.
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )

        ai_insights = response.choices[0].message.content

        st.subheader("🤖 AI Generated Business Insights")
        st.markdown(ai_insights)

    except Exception as e:
        st.error(f"⚠️ Error generating AI insights: {str(e)}")
        st.info("Make sure your OPENROUTER_API_KEY is set correctly in your .env file")

    # -----------------------------------
    # PRIORITY TIER ASSIGNMENT
    # -----------------------------------

    probabilities = model.predict_proba(X_test)

    max_probs = probabilities.max(axis=1)

    priority = []

    for p in max_probs:

        if p > 0.8:
            priority.append("🔴 High")

        elif p > 0.5:
            priority.append("🟡 Medium")

        else:
            priority.append("🟢 Low")

    priority_df = pd.DataFrame({
        "Priority Tier": priority,
        "Confidence": max_probs
    })

    st.subheader("Priority Tier Assignment")

    st.dataframe(priority_df.head(10))
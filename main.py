# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📈 Sales Forecasting Dashboard")
st.markdown(
    "Machine Learning Based Sales Prediction System using XGBoost 🚀"
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Superstore CSV Dataset",
    type=["csv"]
)

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

if uploaded_file is not None:

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    df = pd.read_csv(uploaded_file, encoding='latin1')

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # DATA CLEANING
    # ---------------------------------------------------

    df['Order Date'] = pd.to_datetime(df['Order Date'])

    df = df.sort_values('Order Date')

    # ---------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------

    df['day'] = df['Order Date'].dt.day
    df['month'] = df['Order Date'].dt.month
    df['year'] = df['Order Date'].dt.year
    df['weekday'] = df['Order Date'].dt.weekday
    df['quarter'] = df['Order Date'].dt.quarter

    # Lag Features
    df['lag_1'] = df['Sales'].shift(1)

    df['lag_7'] = df['Sales'].shift(7)

    # Rolling Mean Features
    df['rolling_mean_7'] = (
        df['Sales']
        .rolling(window=7)
        .mean()
    )

    df['rolling_mean_30'] = (
        df['Sales']
        .rolling(window=30)
        .mean()
    )

    # Remove NaN rows
    df = df.dropna()

    # ---------------------------------------------------
    # MONTHLY SALES TREND
    # ---------------------------------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = df.groupby(
        pd.Grouper(key='Order Date', freq='M')
    )['Sales'].sum()

    fig1, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(
        monthly_sales.index,
        monthly_sales.values,
        linewidth=2
    )

    ax1.set_title("Monthly Sales Trend")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")

    plt.xticks(rotation=45)

    st.pyplot(fig1)

    # ---------------------------------------------------
    # FEATURES
    # ---------------------------------------------------

    features = [
        'day',
        'month',
        'year',
        'weekday',
        'quarter',
        'lag_1',
        'lag_7',
        'rolling_mean_7',
        'rolling_mean_30'
    ]

    X = df[features]

    y = df['Sales']

    # ---------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ---------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ---------------------------------------------------
    # MODEL EVALUATION
    # ---------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # ---------------------------------------------------
    # METRICS DISPLAY
    # ---------------------------------------------------

    st.subheader("📌 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    # ---------------------------------------------------
    # ACTUAL VS PREDICTED GRAPH
    # ---------------------------------------------------

    st.subheader("📉 Actual vs Predicted Sales")

    fig2, ax2 = plt.subplots(figsize=(14, 5))

    ax2.plot(
        y_test.values[:200],
        label='Actual Sales',
        linewidth=2
    )

    ax2.plot(
        predictions[:200],
        label='Predicted Sales',
        linewidth=2
    )

    ax2.set_title(
        "Actual vs Predicted Sales"
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(fig2)

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------

    st.subheader("🔥 Feature Importance")

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.bar(
        feature_importance['Feature'],
        feature_importance['Importance']
    )

    ax3.set_title(
        "Feature Importance"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig3)

    # ---------------------------------------------------
    # FUTURE FORECASTING
    # ---------------------------------------------------

    st.subheader("🔮 Future Sales Forecast")

    forecast_days = st.slider(
        "Select Number of Forecast Days",
        7,
        90,
        30
    )

    future_dates = pd.date_range(
        start=df['Order Date'].max(),
        periods=forecast_days,
        freq='D'
    )

    future_df = pd.DataFrame({
        'Order Date': future_dates
    })

    # Future Features

    future_df['day'] = (
        future_df['Order Date'].dt.day
    )

    future_df['month'] = (
        future_df['Order Date'].dt.month
    )

    future_df['year'] = (
        future_df['Order Date'].dt.year
    )

    future_df['weekday'] = (
        future_df['Order Date'].dt.weekday
    )

    future_df['quarter'] = (
        future_df['Order Date'].dt.quarter
    )

    # Static Lag Values

    future_df['lag_1'] = (
        df['Sales'].iloc[-1]
    )

    future_df['lag_7'] = (
        df['Sales'].iloc[-7]
    )

    future_df['rolling_mean_7'] = (
        df['Sales']
        .rolling(7)
        .mean()
        .iloc[-1]
    )

    future_df['rolling_mean_30'] = (
        df['Sales']
        .rolling(30)
        .mean()
        .iloc[-1]
    )

    # Prediction

    X_future = future_df[features]

    future_predictions = model.predict(X_future)

    future_df['Predicted Sales'] = (
        future_predictions
    )

    # ---------------------------------------------------
    # FORECAST GRAPH
    # ---------------------------------------------------

    fig4, ax4 = plt.subplots(figsize=(14, 5))

    ax4.plot(
        future_df['Order Date'],
        future_df['Predicted Sales'],
        marker='o',
        linewidth=2
    )

    ax4.set_title(
        "Future Sales Forecast"
    )

    ax4.set_xlabel("Date")

    ax4.set_ylabel("Predicted Sales")

    ax4.grid(True)

    plt.xticks(rotation=45)

    st.pyplot(fig4)

    # ---------------------------------------------------
    # FORECAST TABLE
    # ---------------------------------------------------

    st.subheader("📋 Forecast Data")

    st.dataframe(
        future_df[
            ['Order Date', 'Predicted Sales']
        ]
    )

else:

    st.info(
        "Please upload your Superstore CSV dataset."
    )
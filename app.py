import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

st.set_page_config(page_title="SA Crime Dashboard", layout="wide")
st.title("🚔 South Africa Crime Analytics & Forecasting")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('merged_data.csv', parse_dates=['Date'], dayfirst=True)
    return df

df = load_data()
st.sidebar.header("Filters")

# Detect crime columns
crime_columns = [c for c in df.columns if 'crime' in c.lower() or 'burglary' in c.lower() or 'assault' in c.lower() or 'vehicle' in c.lower()]
if not crime_columns:
    crime_columns = ['Total_Crime']

crime_choice = st.sidebar.selectbox("Crime category", crime_columns)
provinces = ['All'] + sorted(df['Province'].fillna('Unknown').unique().tolist())
province_choice = st.sidebar.selectbox("Province", provinces)
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filter
filtered = df.copy()
if province_choice != 'All':
    filtered = filtered[filtered['Province']==province_choice]
filtered = filtered[(filtered['Date'].dt.date >= date_range[0]) & (filtered['Date'].dt.date <= date_range[1])]

# -------------------------------
# EDA
# -------------------------------
st.header("📊 Exploratory data")
col1, col2 = st.columns((2,1))

with col1:
    st.subheader(f"{crime_choice} over time")
    ts = filtered.groupby(pd.Grouper(key='Date', freq='M'))[crime_choice].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts['Date'], ts[crime_choice], marker='o')
    ax.set_xlabel('Date'); ax.set_ylabel('Count')
    st.pyplot(fig)

with col2:
    st.subheader("Top stations")
    top_st = filtered.groupby('Station')[crime_choice].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_st)

# -------------------------------
# Classification (on-the-fly small model, cached)
# -------------------------------
st.header("🎯 Hotspot Classification (on filtered data)")

# Create hotspot label (75th percentile) per station using aggregated Total_Crime in filter
agg = filtered.groupby('Station').agg({crime_choice:'sum', 'Population':'first', 'Area':'first', 'Density':'first'}).reset_index()
agg['Hotspot'] = (agg[crime_choice] >= agg[crime_choice].quantile(0.75)).astype(int)

st.write("Hotspot label counts:")
st.write(agg['Hotspot'].value_counts())

@st.cache_data
def train_classifier(data):
    # small feature set
    X = data[['Population','Area','Density', crime_choice]].fillna(0)
    y = data['Hotspot']
    if len(data) < 10 or y.nunique() < 2:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline([('impute', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features)])
    pipeline = Pipeline([('pre', preprocessor), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return pipeline, acc, cm

model, acc, cm = train_classifier(agg)

if model is None:
    st.info("Not enough data to train classifier for this selection.")
else:
    st.metric("Accuracy", f"{acc:.2%}")
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=['Not','Hot'], yticklabels=['Not','Hot'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    st.pyplot(fig_cm)

# -------------------------------
# Forecasting: load precomputed forecast if exists; else try local Prophet (if installed)
# -------------------------------
st.header("📅 Forecast")

fc_path = f"precomputed_forecasts/forecast_{crime_choice}.csv"
if os.path.exists(fc_path):
    st.success("Loaded precomputed forecast")
    fc = pd.read_csv(fc_path, parse_dates=['ds'])
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(fc['ds'], fc['yhat'], label='yhat')
    ax.fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'], alpha=0.2)
    ax.set_title(f"Precomputed Forecast for {crime_choice}")
    st.pyplot(fig)
else:
    st.info("No precomputed forecast found. Attempting to compute forecast on server (requires prophet installed).")
    try:
        from prophet import Prophet
        df_prop = filtered[['Date', crime_choice]].rename(columns={'Date':'ds', crime_choice:'y'}).dropna()
        df_prop = df_prop.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()
        m = Prophet(yearly_seasonality=True)
        m.fit(df_prop)
        future = m.make_future_dataframe(periods=12, freq='M')
        fc = m.predict(future)
        fig = m.plot(fc)
        st.pyplot(fig)
    except Exception as e:
        st.error("Could not compute forecast on server. Consider precomputing forecasts locally and adding them to precomputed_forecasts/ in the repo.")
        st.write(e)

# -------------------------------
# Download filtered data
# -------------------------------
st.download_button("Download filtered CSV", data=filtered.to_csv(index=False), file_name="filtered_data.csv")

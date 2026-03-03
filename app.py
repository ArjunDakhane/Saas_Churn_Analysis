# ==========================================
# STREAMLIT BUSINESS-READY CHURN SYSTEM
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import os

st.set_page_config(page_title="Churn Prediction System", layout="wide")

st.title("SaaS Customer Churn Prediction System")

# ------------------------------------------
# LOAD & TRAIN MODEL (Cached)
# ------------------------------------------

@st.cache_resource
def load_and_train():

    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    df = pd.read_csv("Telco_customer_churn2.csv")
    df.columns = df.columns.str.strip()

    leakage_cols = ["Churn Label", "Churn Score", "Churn Reason"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna()

    target = "Churn Value"

    customer_ids = df["CustomerID"]
    customer_names = df["NAME"] if "NAME" in df.columns else None
    monthly_charges = df["Monthly Charges"]

    drop_cols = ["CustomerID", "Lat Long", "Latitude", "Longitude", "Zip Code", "NAME"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    joblib.dump((model, X, y, encoders, roc, customer_names), "model.pkl")

    return model, X, y, encoders, roc, customer_names


model, X, y, encoders, roc, customer_names = load_and_train()

st.success(f"Model Trained | ROC-AUC: {round(roc,3)}")

# ------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------

y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred)

st.subheader("Model Confusion Matrix (Threshold = 0.5)")
cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Churn (0)", "Actual: Churn (1)"],
    columns=["Predicted: No Churn (0)", "Predicted: Churn (1)"]
)

st.write(cm_df)

# ------------------------------------------
# DASHBOARD SUMMARY
# ------------------------------------------

probs = model.predict_proba(X)[:, 1]

df_results = pd.DataFrame({
    "Name": customer_names,
    "Churn_Probability": probs
})

df_results["Risk_Level"] = np.where(
    df_results["Churn_Probability"] >= 0.7, "High",
    np.where(df_results["Churn_Probability"] >= 0.4, "Medium", "Low")
)

col1, col2, col3 = st.columns(3)

col1.metric("High Risk Customers", (df_results["Risk_Level"] == "High").sum())
col2.metric("Medium Risk Customers", (df_results["Risk_Level"] == "Medium").sum())
col3.metric("Low Risk Customers", (df_results["Risk_Level"] == "Low").sum())

st.subheader("View Customers by Risk Level")

if st.button("Show High Risk Customers"):
    high_risk = df_results[df_results["Risk_Level"] == "High"][["Name", "Churn_Probability"]]
    st.write(high_risk)

if st.button("Show Medium Risk Customers"):
    medium_risk = df_results[df_results["Risk_Level"] == "Medium"][["Name", "Churn_Probability"]]
    st.write(medium_risk)

if st.button("Show Low Risk Customers"):
    low_risk = df_results[df_results["Risk_Level"] == "Low"][["Name", "Churn_Probability"]]
    st.write(low_risk)

st.divider()

# ------------------------------------------
# INDIVIDUAL CUSTOMER PREDICTION
# ------------------------------------------

st.header("Predict Individual Customer Churn")

input_data = {}

for col in X.columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        val = st.selectbox(col, options)
        input_data[col] = encoders[col].transform([val])[0]
    else:
        input_data[col] = st.number_input(col, value=float(X[col].mean()))

if st.button("Predict Churn"):

    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Churn Probability: {prob*100:.2f}%")

    if prob >= 0.7:
        st.error("High Risk Customer")
    elif prob >= 0.4:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")

    # Feature contribution
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(input_df)
    contribs = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    contrib_series = pd.Series(contribs[0][:-1], index=X.columns)
    top_features = contrib_series.abs().sort_values(ascending=False).head(3)

    st.subheader("Top 3 Features Driving Churn")
    st.write(top_features)

    # Recommended Action
    if prob >= 0.7:
        st.write("Recommended Action: Immediate retention call + Offer annual discount + Feature onboarding")
    elif prob >= 0.4:
        st.write("Recommended Action: Send education email + Limited-time discount")
    else:
        st.write("Recommended Action: Upsell premium plan")

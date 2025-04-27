import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score

# Streamlit setup
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🔁 Customer Churn Prediction Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    from analysis.cltv import summary, cltv_df
    return full_orders, summary, cltv_df

full_orders, summary, cltv_df = load_data()


st.dataframe(summary.head(10))
st.dataframe(cltv_df)


# --- Data Preparation ---
st.header("📦 Data Preparation")

full_orders["order_purchase_timestamp"] = pd.to_datetime(full_orders["order_purchase_timestamp"])
max_date = full_orders["order_purchase_timestamp"].max()
cutoff_date = max_date - timedelta(days=180)

filtered_orders = full_orders[full_orders["order_purchase_timestamp"] <= cutoff_date]

# Last purchase per customer
last_purchase_df = (
    filtered_orders.groupby("customer_unique_id")["order_purchase_timestamp"]
    .max()
    .reset_index()
    .rename(columns={"order_purchase_timestamp": "last_purchase"})
)

# Churn labeling
future_orders = full_orders[full_orders["order_purchase_timestamp"] > cutoff_date]
churned_customers = set(last_purchase_df["customer_unique_id"]) - set(future_orders["customer_unique_id"])
last_purchase_df["churned"] = last_purchase_df["customer_unique_id"].isin(churned_customers).astype(int)

# Join with CLTV summary
data = pd.merge(last_purchase_df, cltv_df, on="customer_unique_id", how="inner")

# Feature Engineering
data["days_since_last_purchase"] = (cutoff_date - data["last_purchase"]).dt.days

st.write("Total customers:", data.shape[0])
st.write("Churn distribution:")
fig = px.bar(data['churned'].value_counts().reset_index(), x='churned', y='count', labels={'churned':'Churned', 'count':'Count'})
st.plotly_chart(fig)

if st.checkbox("Show insights for data preparation"):
    st.info("""
    **What was done:**
    - Customers were labeled as churned if they did not purchase in the 180 days after a cutoff.
    - Features are built only from the data available **before** the cutoff to prevent leakage.

    **Why:**
    - This setup simulates real-world prediction without looking into the future.

    **Recommendations:**
    - Consider also adding dynamic features (e.g., purchase frequency, average order value).
    - Include more robust segmentation, like RFM (Recency, Frequency, Monetary) features.
    """)

# --- Feature Exploration ---
st.header("🔎 Exploratory Data Analysis")

# Churn rate by CLTV segment
st.subheader("Churn Rate by CLTV Segment")
cltv_churn = data.groupby("CLTV_new_Segment")["churned"].mean().sort_values()
fig = px.bar(cltv_churn, labels={'value':'Churn Rate', 'CLTV_new_Segment':'CLTV Segment'})
st.plotly_chart(fig)

# Recency vs Churn
st.subheader("Recency vs Churn")
fig = px.box(data, x="churned", y="days_since_last_purchase", labels={'churned':'Churned (0=No, 1=Yes)', 'days_since_last_purchase':'Days Since Last Purchase'})
st.plotly_chart(fig)

if st.checkbox("Show insights for feature exploration"):
    st.info("""
    **What was done:**
    - Analyzed churn rate across different CLTV segments.
    - Visualized recency (days since last purchase) against churn behavior.

    **Why:**
    - Understanding correlations helps in feature selection and model building.

    **Recommendations:**
    - Higher recency (longer time since last purchase) correlates with higher churn.
    - Lower CLTV segments show higher churn rates; target them with retention campaigns.
    """)

# Encode categorical features
le = LabelEncoder()
data["cltv_segment_encoded"] = le.fit_transform(data["CLTV_new_Segment"])

X = data.drop(columns=["customer_unique_id", "last_purchase", "CLTV_new_Segment", "churned"])
y = data["churned"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
st.header("🧠 Model Training & Evaluation")

# Dummy baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
st.write("Baseline Accuracy (Dummy Model):", dummy.score(X_test, y_test))

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

st.subheader("📊 Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
st.plotly_chart(fig)

if st.checkbox("Show insights for model evaluation"):
    st.info("""
    **What was done:**
    - Compared Random Forest model against a Dummy Classifier (predicts most frequent class).
    - Evaluated using accuracy, precision, recall, F1-score.

    **Why:**
    - Baseline models help gauge if our model is actually learning patterns.

    **Recommendations:**
    - If your model barely beats the dummy, rethink feature engineering.
    - Focus on improving recall and F1-score, especially if churners are minority class.
    """)

# --- ROC Curve ---
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc_score:.2f}'))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='ROC Curve')
st.plotly_chart(fig)

# Precision-Recall Curve
st.subheader("🎯 Precision-Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_proba)
fig = go.Figure()
fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines'))
fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
st.plotly_chart(fig)

# Cross-Validation
st.subheader("🔁 Cross Validation Accuracy")
cv_score = cross_val_score(clf, X, y, cv=5)
st.write(f"Average CV Accuracy: **{cv_score.mean():.4f}**")

if st.checkbox("Show insights for model performance"):
    st.info("""
    **What was done:**
    - Evaluated model using ROC AUC and Precision-Recall curves.
    - Cross-validated Random Forest across 5 folds.

    **Why:**
    - ROC AUC summarizes performance across thresholds.
    - Precision-Recall is better suited when classes are imbalanced.

    **Recommendations:**
    - Tune model thresholds depending on your business objective (maximize recall if losing a customer is costly).
    """)

# --- Prediction Output ---
st.header("📋 Prediction Results")

results_df = X_test.copy()
results_df["actual_churn"] = y_test
results_df["churn_probability"] = y_proba
results_df["predicted_churn"] = y_pred

st.dataframe(results_df.head(20))

csv = results_df.to_csv(index=False)
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv"
)

if st.checkbox("Show insights for predictions"):
    st.info("""
    **What was done:**
    - Predictions and churn probabilities are shown for easy stakeholder consumption.

    **Why:**
    - Having probability scores allows business users to prioritize interventions.

    **Recommendations:**
    - Focus retention offers on customers with churn probability between 0.5 to 0.8 (uncertain cases).
    - Highly likely churners (>0.8) may require aggressive win-back campaigns.
    """)


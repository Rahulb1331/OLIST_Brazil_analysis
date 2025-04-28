import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

# --- Streamlit setup
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🔁 Customer Churn Prediction Dashboard")

# --- Load Data
@st.cache_data
def load_data():
    from analysis.Preprocessing import full_orders
    from analysis.cltv import cltv_df
    return full_orders, cltv_df

full_orders, cltv_df = load_data()

# --- Data Preparation
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

cltv_df = cltv_df[cltv_df["total_orders"] > 1].dropna()

# Create features
customer_features = filtered_orders.groupby("customer_unique_id").agg({
    "order_id": "count",
    "payment_value": "sum",
    "order_purchase_timestamp": ["max", "min"]
}).reset_index()

customer_features.columns = ["customer_unique_id", "total_orders", "total_revenue", "last_order", "first_order"]
customer_features["recency_days"] = (cutoff_date - customer_features["last_order"]).dt.days
customer_features["customer_age_days"] = (cutoff_date - customer_features["first_order"]).dt.days
customer_features["avg_order_value"] = customer_features["total_revenue"] / customer_features["total_orders"]

# Label churn
future_orders = full_orders[full_orders["order_purchase_timestamp"] > cutoff_date]
churned_customers = set(customer_features["customer_unique_id"]) - set(future_orders["customer_unique_id"])
customer_features["churned"] = customer_features["customer_unique_id"].isin(churned_customers).astype(int)

# Join with CLTV
customer_features = customer_features[customer_features["customer_unique_id"].isin(cltv_df["customer_unique_id"])]
data = pd.merge(customer_features, cltv_df[['customer_unique_id', 'CLTV_new_Segment']], on="customer_unique_id", how="left")

# Feature Engineering
data["days_since_last_purchase"] = (cutoff_date - data["last_order"]).dt.days
le = LabelEncoder()
data["cltv_segment_encoded"] = le.fit_transform(data["CLTV_new_Segment"])

X = data.drop(columns=["customer_unique_id", "last_order", "first_order", "CLTV_new_Segment", "churned"])
y = data["churned"]

# --- Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training
st.header("🧠 Model Training")

@st.cache_data
def train_all_models(X_train, y_train, X_test, y_test, X, y):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
        )
    }

    model_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, preds, output_dict=True)
        confusion = confusion_matrix(y_test, preds)
        roc_auc = roc_auc_score(y_test, proba)
        cv = cross_val_score(model, X, y, cv=5).mean()

        model_results[name] = {
            "model": model,
            "preds": preds,
            "proba": proba,
            "classification_report": report,
            "confusion_matrix": confusion,
            "roc_auc": roc_auc,
            "cross_val": cv
        }

    return model_results
    
model_results = train_all_models(X_train, y_train, X_test, y_test, X, y)

# --- Model Selection
st.header("🎛 Select Model to Evaluate")

selected_model = st.selectbox("Select the model you want to evaluate:", list(model_results.keys()))
results = model_results[selected_model]

# --- Results
st.subheader(f"📊 Results for {selected_model}")

# ROC Curve
st.subheader(f"📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, results['proba'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={results["roc_auc"]:.2f}'))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig)

# Precision-Recall Curve
st.subheader(f"🎯 Precision-Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, results['proba'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines'))
fig.update_layout(xaxis_title='Recall', yaxis_title='Precision')
st.plotly_chart(fig)

# Classification Report
st.subheader(f"📋 Classification Report")
st.json(results['classification_report'])

# Confusion Matrix
st.subheader(f"🧮 Confusion Matrix")
fig = px.imshow(results['confusion_matrix'], text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
st.plotly_chart(fig)

# Cross Validation Score
st.subheader(f"🔁 Cross-Validation Score")
st.write(f"**{results['cross_val']:.4f}**")

# --- Model Comparison
st.header("📈 Model Comparison Table")

comparison_df = pd.DataFrame([
    {
        "Model": model,
        "Accuracy": model_results[model]['classification_report']['accuracy'],
        "Precision": model_results[model]['classification_report']['1']['precision'],
        "Recall": model_results[model]['classification_report']['1']['recall'],
        "F1-Score": model_results[model]['classification_report']['1']['f1-score'],
        "ROC AUC": model_results[model]['roc_auc'],
        "Cross-Val Score": model_results[model]['cross_val']
    }
    for model in model_results
])

st.dataframe(comparison_df.sort_values(by="F1-Score", ascending=False))


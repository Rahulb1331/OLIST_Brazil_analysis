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
# ✅ Merge RFM features

st.write("Total customers:", data.shape[0])
st.write("Churn distribution:")
fig = px.bar(data['churned'].value_counts().reset_index(), x='churned', y='count', labels={'churned':'Churned', 'count':'Count'})
st.plotly_chart(fig)

if st.checkbox("Show insights for data preparation", key = "key1"):
    st.info("""
    **What was done:**
    - Customers were labeled as churned if they did not purchase in the 180 days after the cutoff date (i.e., the date on which the last order was recorded).
    - Features are built only from the data available **before** the cutoff to prevent leakage.

    **Observations:**
    - An approximately 77% churn rate across all segments suggests that Olist’s current strategies—such as follow-up communication, personalization, or loyalty programs—might not be effective in encouraging repeat purchases and building long-term relationships. 
    - It is also possible that competitive pressures, pricing strategies, or broader market trends are influencing customer behavior. The customers might be switching to the competitors. 
    """)

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

if st.checkbox("Show insights for feature exploration", key = "key2"):
    st.info("""
    **What was done:**
    - Analyzed churn rate across different CLTV segments.
    - Visualized recency (days since last purchase) against churn behavior.

    **Why:**
    - **Recency** is computed using the customer’s **last pre‑cutoff** purchase and the cutoff date (max_date − 180 days).
    - **Churn** is flagged **only** if they make **no** purchases **after** that cutoff window.
    - So a point like **561 days, churn = 0** means the customer went 561 days before cutoff but **did** return in the post‑cutoff period—hence they aren’t marked as churned.

    **Recommendations:**
    - When interpreting recency vs. churn, remember these two windows are separate—consider showing both pre‑cutoff recency and “days since last order overall” if you need deeper context.
    - This can be used to identify “at‑risk” customers who had long pre‑cutoff gaps but eventually returned (case in point, a non churned customer with days since last purchase being 561 days), and see if they need different re‑engagement tactics.
    """)

le = LabelEncoder()
data["cltv_segment_encoded"] = le.fit_transform(data["CLTV_new_Segment"])

X = data.drop(columns=["customer_unique_id", "last_order", "first_order", "CLTV_new_Segment", "churned"])
y = data["churned"]

#Correlational heatmap

with st.expander("Show feature correlation heatmap", expanded=False):
    st.subheader("Feature Correlation Matrix")
    corr = data.drop(columns=["customer_unique_id", "last_order", "first_order", "CLTV_new_Segment"]).corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
    st.plotly_chart(fig)

    if st.checkbox("Show insights for feature exploration", key = "key4"):
        st.info("""
        **Observations:**
        - Features like total_revenue and avg_order_value are having a correlation of 0.69031, while recency_days-customer_age_days and days_since_last_purchase-recency_days are showing very high correlation. In such cases what we can do is we can merge these pairs to form a single column which would prevent overlapping of redundant information. Also, merging such features can reduce multicollinearity.
        - We can also see that the **churned** column is having modest correlation with the predicting parameters. This might be due to the reason that a simple Pearson correlation only captures linear relationships. The effect of columns like say recency_days on churn might be non-linear. Also, Churn is often a multi-dimensional behavior influenced by interactions among several factors, its true impact might only be visible when combined with other features 
        - We can see that churned is having a negative correlation with the CLTV. This is typically expected as higher CLTV indicates that a customer is more valuable to the company and, in many cases, tends to receive more attention through support, tailored offers, or loyalty programs. As a result, such customers are generally more engaged and less likely to churn. Thus, as CLTV increases, the churn probability often decreases—resulting in a negative correlation.
        
        **Next Steps:**
        - I'll be doing feature engineering where features with high correlation will be dealt with, by merging them together or transforming them.
        """)


# --- Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training
st.header("🧠 Model Training")
# Dummy baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
st.write("Baseline Accuracy (Dummy Model):", dummy.score(X_test, y_test))

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

if st.checkbox("Show insights for model evaluation", key = "key5"):
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


# Cross Validation Score
st.subheader(f"🔁 Cross-Validation Score")
st.write(f"**{results['cross_val']:.4f}**")

if st.checkbox("Show insights for model performance", key = "key6"):
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

if st.checkbox("Show insights for predictions", key = "key7"):
    st.info("""
    **What was done:**
    - Predictions and churn probabilities are shown for easy stakeholder consumption.

    **Why:**
    - Having probability scores allows business users to prioritize interventions.

    **Recommendations:**
    - Focus retention offers on customers with churn probability between 0.5 to 0.8 (uncertain cases).
    - Highly likely churners (>0.8) may require aggressive win-back campaigns.
    """)

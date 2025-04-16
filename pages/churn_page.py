import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Scripts.config import setup_environment
setup_environment()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, min, datediff, lit, count, avg, sum as spark_sum

# Spark & data imports
from analysis.Preprocessing import full_orders
from analysis.cltv import summary

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Streamlit setup
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🔁 Customer Churn Prediction Dashboard")

# --- Data Preparation ---
st.header("📦 Data Preparation")

# Max and min purchase dates
max_date = full_orders.select(spark_max("order_purchase_timestamp")).first()[0]
min_date = full_orders.select(min("order_purchase_timestamp")).first()[0]

# Last purchase per customer
last_purchase_df = full_orders.groupBy("customer_unique_id") \
    .agg(spark_max("order_purchase_timestamp").alias("last_purchase"))

# Churn label: 1 if no purchase in last 180 days
churn_df = last_purchase_df.withColumn(
    "churned",
    (datediff(lit(max_date), col("last_purchase")) > 180).cast("int")
)

# Create Spark DataFrame from CLTV summary
summary_df = spark.createDataFrame(summary)

# Join churn labels with CLTV summary
churn_features_df = churn_df.join(summary_df, on="customer_unique_id", how="inner")

# Convert to Pandas
churn_features_pd = churn_features_df.toPandas()
churn_features_pd["last_purchase"] = pd.to_datetime(churn_features_pd["last_purchase"])

# Encode cltv_segment
le = LabelEncoder()
churn_features_pd["cltv_segment_encoded"] = le.fit_transform(churn_features_pd["cltv_segment"])

# Date calculations
max_date = datetime.combine(max_date, datetime.min.time())
churn_features_pd["days_since_last_purchase"] = (max_date - churn_features_pd["last_purchase"]).dt.days

# Drop unnecessary columns
churn_features_pd = churn_features_pd.drop(columns=["customer_unique_id", "cltv_segment", "last_purchase"])

# Split features & target
X = churn_features_pd.drop(columns=["churned"])
y = churn_features_pd["churned"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
st.header("🧠 Model Training & Evaluation")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

st.subheader("📊 Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("🧮 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# --- Feature Importance ---
st.subheader("⭐ Feature Importance")
importances = clf.feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", palette="viridis", ax=ax)
st.pyplot(fig)

# --- ROC Curve ---
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2)
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# --- Cross Validation ---
st.subheader("🔁 Cross Validation Accuracy")
cv_score = cross_val_score(clf, X, y, cv=5)
st.write(f"Average CV Accuracy: **{cv_score.mean():.4f}**")

# --- Prediction Output ---
st.header("📋 Prediction Results")

results_df = X_test.copy()
results_df["actual_churn"] = y_test
results_df["churn_probability"] = y_proba
results_df["predicted_churn"] = y_pred

st.dataframe(results_df.head(20))

# Optional: download results
csv = results_df.to_csv(index=False)
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv"
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)

from analysis.Preprocessing import full_orders
from analysis.cltv import summary

# 1. Last purchase date in the dataset
max_date = full_orders.select(spark_max("order_purchase_timestamp")).first()[0]
min_date = full_orders.select(min("order_purchase_timestamp")).first()[0]


# 2. Latest purchase per customer
last_purchase_df = (
    full_orders.groupby("customer_unique_id")["order_purchase_timestamp"]
    .max()
    .reset_index()
    .rename(columns={"order_purchase_timestamp": "last_purchase"})
)

# 3. Churn Label: 1 if >180 days before max_date, else 0
ast_purchase_df["churned"] = (max_date - last_purchase_df["last_purchase"]).dt.days > 180
last_purchase_df["churned"] = last_purchase_df["churned"].astype(int)


# Join churn labels with CLTV features
churn_features_df = pd.merge(last_purchase_df, summary, on="customer_unique_id", how="inner")

# Converting to Pandas to use scikit-learn for modeling.
churn_features_df['last_purchase'] = pd.to_datetime(churn_features_df['last_purchase'])

print("Total customers in churn_df: ", len(last_purchase_df))
print("Total customers in summary: ", len(summary))
print("Customers after join: ", len(churn_features_df))


# Encode categorical columns like cltv_segment

le = LabelEncoder()
churn_features_df["cltv_segment_encoded"] = le.fit_transform(churn_features_df["cltv_segment"])

# Drop non-numeric or redundant columns
churn_features_df = churn_features_df.drop(columns=["customer_unique_id", "cltv_segment"])


# Convert to datetime if it's a date object
max_date = datetime.combine(max_date, datetime.min.time())

# Create a new numeric feature: days since last purchase
churn_features_df["days_since_last_purchase"] = (max_date - churn_features_df["last_purchase"]).dt.days

# Drop the datetime column
churn_features_df = churn_features_df.drop(columns=["last_purchase"])


# Features & Target
X = churn_features_df.drop(columns=["churned"])
y = churn_features_df["churned"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#Feature Importance to see which features drive churn
importances = clf.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance for Churn Prediction")
plt.tight_layout()
plt.show()

# Getting Churn Probabilities
print(y_proba)

# Cross Validation To verify robustness across different subsets:
scores = cross_val_score(clf, X, y, cv=5)
print("CV Accuracy:", scores.mean())

# ROC AUC Curve:

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Churn Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Save Probabilities with Predictions
#If we want to review predictions with churn probabilities
results_df = X_test.copy()
results_df["actual_churn"] = y_test
results_df["churn_probability"] = y_proba
results_df["predicted_churn"] = clf.predict(X_test)


# Without CLTV as a variable
# Aggregate features per customer
customer_features = full_orders.groupby("customer_unique_id").agg({
    "order_id": "count",
    "payment_value": ["mean", "sum"],
    "order_purchase_timestamp": ["min", "max"]
})
# Flatten MultiIndex
customer_features.columns = [
    "_".join(col).strip()
    for col in customer_features.columns.values
]
customer_features = customer_features.reset_index()

# Rename columns for clarity
customer_features = customer_features.rename(columns={
    "order_id_count": "num_orders",
    "payment_value_mean": "avg_order_value",
    "payment_value_sum": "total_revenue",
    "order_purchase_timestamp_min": "first_purchase",
    "order_purchase_timestamp_max": "last_purchase"
})

# Calculate recency & frequency-like metrics
customer_features["first_purchase"] = pd.to_datetime(customer_features["first_purchase"])
customer_features["last_purchase"] = pd.to_datetime(customer_features["last_purchase"])

customer_features["recency_days"] = (max_date - customer_features["last_purchase"]).dt.days
customer_features["purchase_span_days"] = (customer_features["last_purchase"] - customer_features["first_purchase"]).dt.days
customer_features["order_frequency"] = customer_features["purchase_span_days"] / customer_features["num_orders"]

# Join with churn labels
churned_df = last_purchase_df[["customer_unique_id", "churned"]]
churn_features_pd = pd.merge(customer_features, churned_df, on="customer_unique_id", how="inner")

print(churn_features_pd.columns)

# Remove duplicate columns if present
churn_features_pd = churn_features_pd.loc[:, ~churn_features_pd.columns.duplicated()]

# Now safely convert last_purchase if needed
if not pd.api.types.is_datetime64_any_dtype(churn_features_pd['last_purchase']):
    churn_features_pd['last_purchase'] = pd.to_datetime(churn_features_pd['last_purchase'])

# Optional datetime cleanup
churn_features_pd['first_purchase'] = pd.to_datetime(churn_features_pd['first_purchase'])

# Drop datetime cols (or keep if you want to extract year/month later)
churn_features_pd = churn_features_pd.drop(columns=["first_purchase", "last_purchase"])


# Features & Target
X = churn_features_pd.drop(columns=["customer_unique_id", "churned"], errors='ignore')
y = churn_features_pd["churned"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict probabilities
y_probs = clf.predict_proba(X_test)[:, 1]  # Churn probability
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC AUC
roc_auc = roc_auc_score(y_test, y_probs)
fpr, tpr, _ = roc_curve(y_test, y_probs)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC - Churn Prediction")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Create a DataFrame
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importances - Churn Prediction")
plt.tight_layout()
plt.show()

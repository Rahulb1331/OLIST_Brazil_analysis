from analysis.Preprocessing import full_orders
from pyspark.sql.functions import min as spark_min, max as spark_max
import pandas as pd
from analysis.cltv import summary

# CHURN PREDICTION

from pyspark.sql.functions import col, max as spark_max, min, datediff, to_date, lit

# 1. Last purchase date in the dataset
max_date = full_orders.select(spark_max("order_purchase_timestamp")).first()[0]
min_date = full_orders.select(min("order_purchase_timestamp")).first()[0]


# 2. Latest purchase per customer
last_purchase_df = full_orders.groupBy("customer_unique_id") \
    .agg(spark_max("order_purchase_timestamp").alias("last_purchase"))

# 3. Churn Label: 1 if >180 days before max_date, else 0
churn_df = last_purchase_df.withColumn(
    "churned",
    (datediff(lit(max_date), col("last_purchase")) > 180).cast("int")
)

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

summary = spark.createDataFrame(summary)

# Join churn labels with CLTV features
churn_features_df = churn_df.join(
    summary,
    on="customer_unique_id",
    how="inner"
)

# Converting to Pandas to use scikit-learn for modeling.
churn_features_pd = churn_features_df.toPandas()
churn_features_pd['last_purchase'] = pd.to_datetime(churn_features_pd['last_purchase'])

print("Total customers in churn_df: ", churn_df.count())
print("Total customers in summary: ", summary.count())
print("Customers after join: ", churn_features_df.count())


# Encode categorical columns like cltv_segment
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
churn_features_pd["cltv_segment_encoded"] = le.fit_transform(churn_features_pd["cltv_segment"])

# Drop non-numeric or redundant columns
churn_features_pd = churn_features_pd.drop(columns=["customer_unique_id", "cltv_segment"])

from datetime import datetime

# Convert to datetime if it's a date object
max_date = datetime.combine(max_date, datetime.min.time())

# Create a new numeric feature: days since last purchase
churn_features_pd['days_since_last_purchase'] = (max_date - churn_features_pd['last_purchase']).dt.days

# Drop the datetime column
churn_features_pd = churn_features_pd.drop(columns=['last_purchase'])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Features & Target
X = churn_features_pd.drop(columns=["churned"])
y = churn_features_pd["churned"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#Feature Importance to see which features drive churn
import matplotlib.pyplot as plt
import seaborn as sns

importances = clf.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance for Churn Prediction")
plt.tight_layout()
plt.show()

# Getting Churn Probabilities
y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class "1" (churned)
print(y_proba)

# Cross Validation To verify robustness across different subsets:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print("CV Accuracy:", scores.mean())

# ROC AUC Curve:
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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
from pyspark.sql.functions import count, avg, sum as spark_sum, min, max as spark_max, datediff

# Aggregate features per customer
customer_features = full_orders.groupBy("customer_unique_id").agg(
    count("order_id").alias("num_orders"),
    avg("payment_value").alias("avg_order_value"),
    spark_sum("payment_value").alias("total_revenue"),
    spark_min("order_purchase_timestamp").alias("first_purchase"),
    spark_max("order_purchase_timestamp").alias("last_purchase")
)

# Calculate recency & frequency-like metrics
customer_features = customer_features.withColumn(
    "recency_days", datediff(lit(max_date), col("last_purchase"))
).withColumn(
    "purchase_span_days", datediff(col("last_purchase"), col("first_purchase"))
).withColumn(
    "order_frequency", (col("purchase_span_days") / col("num_orders"))
)

# Join with churn labels
churn_features_df = customer_features.join(churn_df, on=["customer_unique_id"], how="inner")

churn_features_pd = churn_features_df.toPandas()

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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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

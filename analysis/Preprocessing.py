from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, mean, when
from pyspark.sql.types import *

venv_python_path = sys.executable
print("the path is ",venv_python_path)
# Initialize Spark Session
spark = SparkSession.builder \
    .appName("E-Commerce Data Processing") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.executorEnv.PYSPARK_PYTHON", venv_python_path) \
    .config("spark.driverEnv.PYSPARK_PYTHON", venv_python_path) \
    .config("spark.pyspark.python", venv_python_path) \
    .config("spark.pyspark.driver.python", venv_python_path) \
    .getOrCreate()

review_schema = StructType([
    StructField("review_id", StringType(), True),
    StructField("order_id", StringType(), True),
    StructField("review_score", IntegerType(), True),
    StructField("review_comment_title", StringType(), True),
    StructField("review_comment_message", StringType(), True),
    StructField("review_creation_date", StringType(), True),
    StructField("review_answer_timestamp", StringType(), True)
])

# Corresponding direct download links from Google Drive
dataset_links = {
    "olist_orders_dataset": "https://drive.google.com/file/d/1IW8RCm8SsxMTnxwBhbY2ki7UQFbjpWJQ/view?usp=sharing",
    "olist_customers_dataset": "https://drive.google.com/file/d/1GlfbdUR7Htaoa23ZaDaJU2BONVpd-46s/view?usp=sharing",
    "olist_order_items_dataset": "https://drive.google.com/file/d/1fzKgJiI8nrpOioDMNEH3FTGjNk38na4K/view?usp=sharing",
    "olist_geolocation_dataset": "https://drive.google.com/file/d/14Ov5-Ulw1pRPQwl-d1HGEDkrdeqAZdsf/view?usp=sharing",
    "olist_order_payments_dataset": "https://drive.google.com/file/d/1Yhb25SAM6uYOKb3LuiZNI87MwpzcMzcm/view?usp=sharing",
    "olist_order_reviews_dataset": "https://drive.google.com/file/d/129XEZCdH-e7LS6RxwJ8yIzTBEO2zSJIZ/view?usp=sharing",
    "olist_products_dataset": "https://drive.google.com/file/d/17jhNuSGKgXTWSop0vsjPGw9CP6eBJto7/view?usp=sharing",
    "olist_sellers_dataset": "https://drive.google.com/file/d/1vhjeb7QmtXiMWBELCylT4vL9-s8s1P_s/view?usp=sharing",
    "product_category_name_translation": "https://drive.google.com/file/d/1viI3NGEKJoN0M8I0DhTGzE47wGRfNB2r/view?usp=sharing"
}

# Load them into a dictionary of Spark DataFrames
dfs = {}

for name in dataset_names:
    if name == "olist_order_reviews_dataset":
        # Handle reviews separately if using custom schema or multiLine
        df = spark.read.csv(
            dataset_links[name],
            header=True,
            schema=review_schema,  # make sure this is defined earlier
            multiLine=True,
            escape='"'
        )
    else:
        df = spark.read.csv(dataset_links[name], header=True, inferSchema=True)
    dfs[name] = df

orders = dfs["olist_orders_dataset"]
customers = dfs["olist_customers_dataset"]
geolocation = dfs["olist_geolocation_dataset"]
product_category = dfs["product_category_name_translation"]
sellers = dfs["olist_sellers_dataset"]
products = dfs["olist_products_dataset"]
order_reviews = dfs["olist_order_reviews_dataset"]
order_payments = dfs["olist_order_payments_dataset"]
order_items = dfs["olist_order_items_dataset"]

# Show Sample Data
data = [product_category,
sellers,
products,
order_reviews,
order_payments,
geolocation,
customers,
orders,
order_items]

order_reviews.show(10, truncate=False)


for dataframe in data:
    print(dataframe)
    # Show the first 5 rows of the dataframe
    dataframe.show(5)
    # Print the schema of the dataframe
    dataframe.printSchema()
    #describe
    dataframe.describe().show()


## DATA Preprocessing and DATA Cleaning
# Convert Dates to Proper Format
orders = orders.withColumn("order_purchase_timestamp", to_date(col("order_purchase_timestamp")))
order_reviews = order_reviews.withColumn("review_creation_date", to_date(col("review_creation_date")))

# Convert review scores to integer
order_reviews = order_reviews.withColumn("review_score", col("review_score").cast(IntegerType()))

# Join on product_category_name (Portuguese)
products_with_english = products.join(
    product_category,
    on="product_category_name",
    how="left"
)

# Replace the original with English (and drop the Portuguese version)
products = products_with_english.drop("product_category_name") \
                                             .withColumnRenamed("product_category_name_english", "product_category")

# Drop rows where product_category is missing
products = products.na.drop(subset=["product_category"])

# Impute missing numerical values with their mean
num_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]

for col_name in num_cols:
    mean_val = products.select(mean(col(col_name))).collect()[0][0]
    products = products.fillna({col_name: mean_val})

# Drop rows where review_score is missing
order_reviews = order_reviews.na.drop(subset=["review_score"])
order_reviews = order_reviews.na.drop(subset=["review_id"])

#Add a new column to order_reviews to check the sentiment of the review
order_reviews = order_reviews.withColumn("review_sentiment",
    when(col("review_score") >= 4, "Positive")
    .when(col("review_score") == 3, "Neutral")
    .otherwise("Negative")
)


# Remove negative or zero prices
order_items = order_items.filter((col("price") > 0) & (col("freight_value") >= 0))

# Join orders + customers
orders_with_customers = orders.join(customers, on="customer_id", how="left")

# Join orders + order items
full_orders = orders_with_customers.join(order_items, on="order_id", how="left")

# Join products
full_orders = full_orders.join(products, on="product_id", how="left")

# Join payments
full_orders = full_orders.join(order_payments, on="order_id", how="left")

# Cache full_orders for performance
full_orders.cache()

__all__ = ["full_orders", "geolocation", "order_reviews", "sellers", "spark"]

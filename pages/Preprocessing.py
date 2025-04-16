from pyspark.sql import SparkSession
#from pyspark.sql.connect.functions import to_timestamp
from pyspark.sql.functions import col, to_date, mean, when
from pyspark.sql.types import *

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("E-Commerce Data Processing") \
    .config("spark.sql.shuffle.partitions", "50") \
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

# Load Dataset
orders = spark.read.csv("../Data/olist_orders_dataset.csv", header=True, inferSchema=True)
customers = spark.read.csv("../Data/olist_customers_dataset.csv", header=True, inferSchema=True)
order_items = spark.read.csv("../Data/olist_order_items_dataset.csv", header=True, inferSchema=True)
geolocation = spark.read.csv("../Data/olist_geolocation_dataset.csv", header=True, inferSchema=True)
order_payments = spark.read.csv("../Data/olist_order_payments_dataset.csv", header=True, inferSchema=True)
order_reviews = spark.read.csv(
    "../Data/olist_order_reviews_dataset.csv",
    header=True,
    schema=review_schema,
    multiLine=True,
    escape='"'
)
products = spark.read.csv("../Data/olist_products_dataset.csv", header=True, inferSchema=True)
sellers = spark.read.csv("../Data/olist_sellers_dataset.csv", header=True, inferSchema=True)
product_category = spark.read.csv("../Data/product_category_name_translation.csv", header=True, inferSchema=True)


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

full_orders.cache()
full_orders.show(5)
order_reviews.show(5)
__all__ = ['full_orders']

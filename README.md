# insurance-claim-prediction
Machine Learning project using PySpark to predict insurance claims

Source Code of this project--
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Insurance Claim Prediction") \
    .getOrCreate()
df = spark.read.csv(
    "C:/Users/satul/Downloads/Insurance claims data.csv",
    header=True,
    inferSchema=True
)

df = df.limit(500)
df.show(5)
from pyspark.sql.functions import col

df = df.dropna()

df = df.fillna({
    "subscription_length": 0,
    "vehicle_age": 0,
    "customer_age": df.selectExpr("avg(customer_age)").first()[0]
})
from pyspark.sql.functions import when

binary_cols = [
    "is_esc", "is_adjustable_steering", "is_tpms",
    "is_parking_sensors", "is_parking_camera",
    "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger",
    "is_brake_assist", "is_power_door_locks",
    "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert"
]

for c in binary_cols:
    df = df.withColumn(c, when(col(c) == "Yes", 1).otherwise(0))
df = df.drop("policy_id", "max_torque", "max_power", "engine_type")
df = df.withColumn(
    "weight",
    when(col("claim_status") == 1, 2).otherwise(1)
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder

categorical_cols = ["fuel_type", "transmission_type", "segment", "model"]

indexers = [
    StringIndexer(inputCol=c, outputCol=c+"_index")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=c+"_index", outputCol=c+"_vec")
    for c in categorical_cols
]
from pyspark.ml.feature import VectorAssembler

numerical_cols = [
    "subscription_length", "vehicle_age", "customer_age",
    "region_density", "displacement", "cylinder",
    "length", "width", "gross_weight", "ncap_rating"
] + binary_cols

feature_cols = numerical_cols + [c+"_vec" for c in categorical_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="claim_status",
    weightCol="weight"
)
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

predictions.select("claim_status", "prediction").show(10)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = MulticlassClassificationEvaluator(
    labelCol="claim_status",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="claim_status",
    predictionCol="prediction",
    metricName="f1"
).evaluate(predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="claim_status",
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(predictions)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
predictions.groupBy("claim_status", "prediction").count().show()
import matplotlib.pyplot as plt

pdf = df.groupBy("fuel_type", "claim_status").count().toPandas()

pivot_df = pdf.pivot(index="fuel_type", columns="claim_status", values="count").fillna(0)

ax = pivot_df.plot(kind="bar", figsize=(9,5))

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.title("Insurance Claim Distribution by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Claims")
plt.xticks(rotation=0)
plt.legend(["Rejected (0)", "Approved (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
df = df.withColumn(
    "age_group",
    when(col("customer_age") < 30, "Young")
    .when(col("customer_age") < 50, "Middle")
    .otherwise("Senior")
)

pdf = df.groupBy("age_group", "claim_status").count().toPandas()

pivot_df = pdf.pivot(index="age_group", columns="claim_status", values="count").fillna(0)

ax = pivot_df.plot(kind="bar", figsize=(8,5))

for container in ax.containers:
    ax.bar_label(container)

plt.title("Claims by Customer Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Claims")
plt.xticks(rotation=0)
plt.legend(["Rejected (0)", "Approved (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
pdf = df.groupBy("airbags", "claim_status").count().toPandas()

pivot_df = pdf.pivot(index="airbags", columns="claim_status", values="count").fillna(0)

ax = pivot_df.plot(kind="bar", figsize=(8,5))

for container in ax.containers:
    ax.bar_label(container)

plt.title("Claims by Number of Airbags")
plt.xlabel("Airbags")
plt.ylabel("Number of Claims")
plt.legend(["Rejected (0)", "Approved (1)"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()




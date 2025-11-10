from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session:
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Load Data:
data_path = r"/home/giangnt/Downloads/NLP_DL/nlp-labs/data/sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Convert -1/1 labels to 0/1: Normalize sentiment labels
df = df.withColumn("label", ((col("sentiment").cast("integer") + 1) / 2))
# Drop rows with null sentiment values before processing
initial_row_count = df.count()
df = df.dropna(subset=["sentiment"])

print(f"Dataset size after cleaning: {df.count()} rows")
df.show(5)

# 3. Buil Preprocessing Pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# 4. Train the Model:
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(trainingData)
# -- make predictions
predictions = model.transform(testData)
predictions.select("text", "prediction", "label").show(10, truncate=False)

# 5. Evaluate the Model:
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.4f}")

f1_evaluator = MulticlassClassificationEvaluator(metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.4f}")

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re

# Initialize Spark
spark = SparkSession.builder.appName("Lab5_Full_Improvement_Experiment").getOrCreate()

# Load Dataset
data_path = "/home/giangnt/Downloads/NLP_DL/nlp-labs/data/sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.withColumn("label", ((col("sentiment").cast("integer") + 1) / 2))
df = df.dropna(subset=["sentiment"])

# Clean text function
def clean_text(text):
    if text:
        text = re.sub(r"http\S+", "", text)  # remove URLs
        text = re.sub(r"<.*?>", "", text)    # remove HTML tags
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-letters
        text = text.lower()
    return text

clean_udf = udf(clean_text, StringType())
df = df.withColumn("clean_text", clean_udf(col("text")))

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Evaluation function
def evaluate(predictions, label):
    acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(metricName="f1")
    acc = acc_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    print(f"{label:<45} Accuracy={acc:.4f} | F1={f1:.4f}")
    return acc, f1

# 1. Baseline: Raw TF-IDF + Logistic Regression
tokenizer_raw = Tokenizer(inputCol="text", outputCol="words")
stopwords_raw = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF_raw = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=10000)
idf_raw = IDF(inputCol="raw_features", outputCol="features")
lr_raw = LogisticRegression(maxIter=10, regParam=0.001)

pipeline_base = Pipeline(stages=[tokenizer_raw, stopwords_raw, hashingTF_raw, idf_raw, lr_raw])
model_base = pipeline_base.fit(train)
pred_base = model_base.transform(test)
acc_base, f1_base = evaluate(pred_base, "A. Baseline (Raw TF-IDF + LR)")

# 2. Improved Preprocessing: CleanText + TF-IDF + Logistic Regression
tokenizer_clean = Tokenizer(inputCol="clean_text", outputCol="words")
stopwords_clean = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF_clean = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=20000)
idf_clean = IDF(inputCol="raw_features", outputCol="features")
lr_clean = LogisticRegression(maxIter=20, regParam=0.001)

pipeline_clean = Pipeline(stages=[tokenizer_clean, stopwords_clean, hashingTF_clean, idf_clean, lr_clean])
model_clean = pipeline_clean.fit(train)
pred_clean = model_clean.transform(test)
acc_clean, f1_clean = evaluate(pred_clean, "B. CleanText + TF-IDF + LR")

# 3. Advanced Embedding: CleanText + Word2Vec + Logistic Regression
word2Vec = Word2Vec(inputCol="filtered", outputCol="features", vectorSize=100, minCount=2)
lr_w2v = LogisticRegression(maxIter=20, regParam=0.01)
pipeline_w2v = Pipeline(stages=[tokenizer_clean, stopwords_clean, word2Vec, lr_w2v])
model_w2v = pipeline_w2v.fit(train)
pred_w2v = model_w2v.transform(test)
acc_w2v, f1_w2v = evaluate(pred_w2v, "C. CleanText + Word2Vec + LR")

# 4. New Model: Raw TF-IDF + Naive Bayes
nb_raw = NaiveBayes(modelType="multinomial", smoothing=1.0)
pipeline_nb_raw = Pipeline(stages=[tokenizer_raw, stopwords_raw, hashingTF_raw, idf_raw, nb_raw])
model_nb_raw = pipeline_nb_raw.fit(train)
pred_nb_raw = model_nb_raw.transform(test)
acc_nb_raw, f1_nb_raw = evaluate(pred_nb_raw, "D. Raw TF-IDF + NaiveBayes")

# 5. CleanText + TF-IDF + Naive Bayes
nb_clean = NaiveBayes(modelType="multinomial", smoothing=1.0)
pipeline_nb_clean = Pipeline(stages=[tokenizer_clean, stopwords_clean, hashingTF_clean, idf_clean, nb_clean])
model_nb_clean = pipeline_nb_clean.fit(train)
pred_nb_clean = model_nb_clean.transform(test)
acc_nb_clean, f1_nb_clean = evaluate(pred_nb_clean, "E. CleanText + TF-IDF + NaiveBayes")

# Summary
print("\n------------- SUMMARY -------------")
print(f"{'Model':<45} {'Accuracy':<10} {'F1':<10}")
print("-"*65)
results = [
    ("A. Raw TF-IDF + LR", acc_base, f1_base),
    ("B. CleanText + TF-IDF + LR", acc_clean, f1_clean),
    ("C. CleanText + Word2Vec + LR", acc_w2v, f1_w2v),
    ("D. Raw TF-IDF + NaiveBayes", acc_nb_raw, f1_nb_raw),
    ("E. CleanText + TF-IDF + NaiveBayes", acc_nb_clean, f1_nb_clean),
]
for r in results:
    print(f"{r[0]:<45} {r[1]:<10.4f} {r[2]:<10.4f}")

spark.stop()

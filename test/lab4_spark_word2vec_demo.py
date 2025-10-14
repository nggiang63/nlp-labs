
import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace

def main():
    # 1 Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("Lab4_Spark_Word2Vec") \
        .master("local[*]") \
        .getOrCreate()

    print(" Spark session initialized successfully.")

    # 2 Load dataset (giả lập nếu không có C4)
    # File JSON thật: c4-train.00000-of-01024-30K.json
    data_path = r"/home/giangnt/Downloads/NLP_DL/nlp-labs/data/c4-train.00000-of-01024-30K.json"

    try:
        df = spark.read.json(data_path)
        df = df[:100000]
        print(f" Loaded JSON data from: {data_path}")
    except Exception:
        print(" Dataset not found — using sample data instead.")
        sample_data = [
            {"text": "King Queen Man Woman Prince Princess"},
            {"text": "Paris France Berlin Germany Tokyo Japan"},
            {"text": "Apple Banana Fruit Food"},
            {"text": "Dog Cat Animal"},
            {"text": "The queen rules the country"},
            {"text": "The king leads the kingdom"},
            {"text": "Computers and software are technology"},
            {"text": "The computer is a useful machine"},
        ]
        df = spark.createDataFrame(sample_data)

    # 3 Tiền xử lý dữ liệu
    # - lowercase
    # - remove punctuation/special chars
    # - tokenize (split words)
    df_clean = df.select(lower(col("text")).alias("text"))
    df_clean = df_clean.withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", ""))
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    df_tokens = tokenizer.transform(df_clean)

    print(" Dữ liệu sau tiền xử lý:")
    df_tokens.select("words").show(truncate=False)

    # 4 Huấn luyện mô hình Word2Vec
    word2vec = Word2Vec(
        vectorSize=100,
        minCount=1,
        inputCol="words",
        outputCol="result"
    )
    model = word2vec.fit(df_tokens)
    print(" Spark Word2Vec training completed.")

    # 5 Tìm từ tương tự (top 5)
    try:
        print(" Các từ tương tự với 'computer':")
        synonyms = model.findSynonyms("computer", 5)
        synonyms.show(truncate=False)
    except Exception as e:
        print(f" Lỗi khi tìm từ tương tự: {e}")

    # 6 Kết thúc Spark
    spark.stop()
    print(" Spark session stopped.")


if __name__ == "__main__":
    main()

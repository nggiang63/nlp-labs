# LAB 5: TEXT CLASSIFICATION
Giang Nguyen Thi - 22001254
2025-11-10
---

## Task 1: Data Preparation

### Mục tiêu
Chuẩn bị dữ liệu huấn luyện cho bài toán phân loại cảm xúc bằng cách:
- Tập văn bản nhỏ (in-memory dataset).  
- Biến đổi văn bản sang dạng số thông qua CountVectorizer / TfidfVectorizer.  
- Hiển thị vocabulary và feature matrix để kiểm tra vector hóa.

### Các bước thực hiện
- Tập dữ liệu gồm 6 câu (3 positive - 3 negative):
   ```python
   texts = [
       "This movie is fantastic and I love it!",
       "I hate this film, it's terrible.",
       "The acting was superb, a truly great experience.",
       "What a waste of time, absolutely boring.",
       "Highly recommend this, a masterpiece.",
       "Could not finish watching, so bad."
   ]
   labels = [1, 0, 1, 0, 1, 0]
   ```
- Dùng `CountVectorizer()` để biến đổi văn bản → ma trận đếm (Count Matrix).  
- In ra từ vựng (vocabulary) và ma trận đặc trưng để xác nhận pipeline hoạt động đúng.

### Hướng dẫn chạy code
```bash
cd ~/Downloads/NLP_DL/nlp-labs/src/lab5_20251021
python task1_data_preparation.py
```

### Kết quả (theo log thực tế)
```
Vocabulary: {'!': 0, "'": 1, ',': 2, '.': 3, 'a': 4, 'absolutely': 5, 'acting': 6, 'and': 7, 'bad': 8, 'boring': 9, 'could': 10, 'experience': 11, 'fantastic': 12, 'film': 13, 'finish': 14, 'great': 15, 'hate': 16, 'highly': 17, 'i': 18, 'is': 19, 'it': 20, 'love': 21, 'masterpiece': 22, 'movie': 23, 'not': 24, 'of': 25, 'recommend': 26, 's': 27, 'so': 28, 'superb': 29, 'terrible': 30, 'the': 31, 'this': 32, 'time': 33, 'truly': 34, 'was': 35, 'waste': 36, 'watching': 37, 'what': 38}

Feature matrix (Count Vectors):
Text 0: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Text 1: [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
Text 2: [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
Text 3: [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
Text 4: [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Text 5: [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
```

### Nhận xét
- Dữ liệu được vector hóa chính xác với 39 tokens.  
- Mỗi văn bản biểu diễn thành vector nhị phân, là đầu vào cho các mô hình huấn luyện.

---

## Task 2: Implementing the TextClassifier

### Mục tiêu
Xây dựng lớp `TextClassifier` để huấn luyện và đánh giá mô hình Logistic Regression.

### Các bước thực hiện
- Tạo file `src/models/text_classifier.py`.
- Viết class `TextClassifier` với các hàm `fit`, `predict`, `evaluate`.
- Dùng `LogisticRegression(solver='liblinear')` phù hợp cho dữ liệu nhỏ.
- Đánh giá mô hình bằng `accuracy_score`, `precision_score`, `recall_score`, `f1_score`.

### Hướng dẫn chạy code
```bash
cd ~/Downloads/NLP_DL/nlp-labs/test
python lab5_task3_test_20251021.py
```

### Kết quả (theo log thực tế)
```
Accuracy: 0.5000
Precision: 0.5000
Recall: 1.0000
F1: 0.6667
Sample Predictions:
This movie is fantastic and I love it! -> Positive
I hate this film, it's terrible. -> Positive
```

### Nhận xét
- Pipeline chạy thành công, mô hình hoạt động đúng chức năng.  
- Accuracy thấp do dataset rất nhỏ (chỉ 6 câu), dẫn đến kết quả chưa ổn định.  
- Các hàm trong class `TextClassifier` đã được cài đặt và chạy chính xác.

---

## Task 3: Spark Sentiment Analysis

### Mục tiêu
Xây dựng pipeline xử lý văn bản trên Spark ML để phân loại cảm xúc trên tập dữ liệu lớn.

### Các bước thực hiện
- Khởi tạo SparkSession và đọc file dữ liệu `sentiments.csv` (gồm cột text, sentiment).  
- Tiền xử lý gồm Tokenizer, StopWordsRemover, HashingTF, IDF.  
- Huấn luyện mô hình Logistic Regression với `featuresCol='features'`, `labelCol='label'`.  
- Đánh giá Accuracy và F1-score bằng MulticlassClassificationEvaluator.

### Hướng dẫn chạy code
```bash
cd ~/Downloads/NLP_DL/nlp-labs/test
python lab5_spark_sentiment_analysis.py
```

### Kết quả (theo log thực tế)
```
Dataset size after cleaning: 5791 rows
Model Accuracy: 0.7295
F1 Score: 0.7266
```

### Nhận xét
- Pipeline Spark ML chạy thành công, dữ liệu được làm sạch và vector hóa đúng.  
- Accuracy đạt 0.7295, F1 đạt 0.7266, cao hơn so với mô hình sklearn nhỏ.  
- Các cảnh báo NativeCodeLoader không ảnh hưởng đến kết quả.

---

## Task 4: Model Improvement Experiment

### Mục tiêu
Thử nghiệm và so sánh các kỹ thuật cải thiện hiệu suất mô hình:
- Thêm bước CleanText.
- So sánh giữa TF-IDF, Word2Vec, và Naive Bayes.

### Các bước thực hiện
- Tạo file `test/lab5_improve_test.py`.
- Thực hiện 5 thí nghiệm mô hình khác nhau:
   - Baseline: Raw TF-IDF + LR.
   - CleanText + TF-IDF + LR.
   - CleanText + Word2Vec + LR.
   - Raw TF-IDF + Naive Bayes.
   - CleanText + TF-IDF + Naive Bayes.

### Hướng dẫn chạy code
```bash
python lab5_improve_test.py
```

### Kết quả (theo log thực tế)
```
A. Baseline (Raw TF-IDF + LR)                 Accuracy=0.7295 | F1=0.7266
B. CleanText + TF-IDF + LR                    Accuracy=0.7746 | F1=0.7747
C. CleanText + Word2Vec + LR                  Accuracy=0.6619 | F1=0.6064
D. Raw TF-IDF + NaiveBayes                    Accuracy=0.6844 | F1=0.6842
E. CleanText + TF-IDF + NaiveBayes            Accuracy=0.7259 | F1=0.7261
```

### Nhận xét
- Mô hình CleanText + TF-IDF + Logistic Regression cho kết quả tốt nhất với Accuracy 0.7746 và F1 0.7747.  
- Việc tiền xử lý văn bản giúp giảm nhiễu và tăng chất lượng đặc trưng TF-IDF.  
- Word2Vec cho kết quả thấp do dữ liệu nhỏ, embedding chưa đủ ngữ nghĩa.  
- Naive Bayes hiệu quả trung bình nhưng kém hơn LR vì giả định độc lập đơn giản.

---

## Tài liệu tham khảo
1. *scikit-learn Documentation*: https://scikit-learn.org/stable/  
2. *Apache Spark MLlib Guide*: https://spark.apache.org/docs/latest/ml-guide.html  

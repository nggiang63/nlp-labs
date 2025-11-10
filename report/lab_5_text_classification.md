# LAB 5: TEXT CLASSIFICATION

**Giang Nguyen Thi - 22001254**  
2025-11-10  

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
- Khởi tạo tokenizer để tách từ và vectorizer để biến đổi dữ liệu văn bản thành ma trận đặc trưng số.
- Gọi hàm fit() của vectorizer để học từ vựng (vocabulary) từ toàn bộ tập dữ liệu.
- Gọi transform() để chuyển từng câu thành vector có độ dài bằng số lượng từ trong từ vựng. Mỗi phần tử trong vector biểu thị tần suất xuất hiện của từ tương ứng.
- In ra kết quả bao gồm từ vựng (vocabulary) và ma trận đặc trưng (feature matrix) để kiểm tra pipeline hoạt động đúng.

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

## Task 2: Basic Test Case

### Mục tiêu
Xây dựng lớp `TextClassifier` để huấn luyện và đánh giá mô hình Logistic Regression.

### Các bước thực hiện
- Tạo file `src/models/text_classifier.py` để định nghĩa lớp TextClassifier.
- Viết ba phương thức chính: fit để huấn luyện mô hình, predict để dự đoán nhãn, và evaluate để tính các chỉ số đánh giá.
- Sử dụng LogisticRegression(solver='liblinear') cho bài toán phân loại nhị phân với dữ liệu nhỏ.
- Tạo file kiểm thử  `test/lab5_task3_test_20251021.py`.
- Viết hàm `load_data()` để tạo dữ liệu văn bản và nhãn mẫu từ 6 câu.
- Chia dữ liệu thành tập huấn luyện và tập kiểm thử bằng train_test_split.
- Khởi tạo RegexTokenizer và CountVectorizer để chuẩn bị dữ liệu đầu vào.
- Khởi tạo đối tượng TextClassifier với vectorizer vừa tạo.
- Huấn luyện mô hình bằng phương thức fit.
- Dự đoán nhãn trên tập kiểm thử bằng phương thức predict.
- Đánh giá kết quả dự đoán bằng phương thức evaluate để lấy Accuracy, Precision, Recall, F1-score.
- In ra kết quả dự đoán và các chỉ số đánh giá để xác nhận pipeline hoạt động chính xác.

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

## Advanced Example: Spark Sentiment Analysis

### Mục tiêu
Xây dựng pipeline xử lý văn bản trên Spark ML để phân loại cảm xúc trên tập dữ liệu lớn.

### Các bước thực hiện
- Tạo file `test/lab5_spark_sentiment_analysis.py`.
- Khởi tạo SparkSession.
- Đọc file dữ liệu `data/sentiments.csv` chứa hai cột chính: text (nội dung văn bản) và sentiment (nhãn -1 hoặc 1).
- Chuẩn hóa nhãn cảm xúc bằng cách chuyển sentiment từ giá trị (-1, 1) sang (0, 1) và loại bỏ các dòng có giá trị null.
- Tạo pipeline tiền xử lý gồm các giai đoạn:
      + Tokenizer: tách câu thành danh sách các từ.
      + StopWordsRemover: loại bỏ những từ dừng phổ biến không mang ý nghĩa phân loại.
      + HashingTF: ánh xạ các từ thành vector đặc trưng dựa trên tần suất.
      + IDF: tính trọng số ngược tần suất từ để tăng độ quan trọng cho các từ hiếm.
- Thêm giai đoạn huấn luyện LogisticRegression với các tham số `maxIter=10`, `regParam=0.001`, `featuresCol='features'`, `labelCol='label'`.
- Gộp tất cả các giai đoạn trên vào Pipeline của Spark ML để tự động hóa toàn bộ quy trình xử lý.
- Chia dữ liệu thành tập huấn luyện và tập kiểm thử bằng randomSplit([0.8, 0.2], seed=42).
- Huấn luyện mô hình trên tập train và dự đoán nhãn trên tập test.
- Đánh giá kết quả bằng MulticlassClassificationEvaluator với hai chỉ số Accuracy và F1-score.
- In ra kết quả mô hình và hiển thị một số dòng dữ liệu dự đoán để kiểm tra pipeline hoạt động chính xác.

### Hướng dẫn chạy code
```bash
cd ~/Downloads/NLP_DL/nlp-labs/test
python lab5_spark_sentiment_analysis.py
```

### Kết quả (theo log thực tế)
```
Model Accuracy: 0.7295
F1 Score: 0.7266
```

### Nhận xét
- Pipeline Spark ML chạy hoàn chỉnh, gồm cả tiền xử lý và huấn luyện Logistic Regression.
- Accuracy đạt 0.7295, F1 đạt 0.7266, cao hơn so với mô hình sklearn nhỏ.  

---

## Task 4: Model Improvement Experiment

### Mục tiêu
Thử nghiệm và so sánh các kỹ thuật cải thiện hiệu suất mô hình:
- Thêm bước CleanText.
- So sánh giữa TF-IDF, Word2Vec, và Naive Bayes.

### Các bước thực hiện
- Tạo file `test/lab5_improve_test.py` để tổ chức toàn bộ thí nghiệm.
- Khởi tạo SparkSession và đọc dữ liệu `sentiments.csv` (gồm cột text, sentiment).
- Chuyển đổi nhãn cảm xúc từ (-1, 1) sang (0, 1) và loại bỏ các dòng thiếu dữ liệu.
- Xây dựng hàm `clean_text()` để làm sạch văn bản:
      + Xóa URL, thẻ HTML, ký tự đặc biệt và chuyển toàn bộ chữ thành chữ thường.
      + Đăng ký hàm này thành UDF để áp dụng trên toàn bộ DataFrame Spark.
      + Tạo thêm cột clean_text chứa phiên bản đã làm sạch của văn bản.
- Chia dữ liệu thành tập huấn luyện và kiểm thử bằng randomSplit([0.8, 0.2], seed=42).
- Xây dựng hàm `evaluate(predictions, label)` để tự động tính Accuracy và F1-score cho từng mô hình.

### Các mô hình thử nghiệm
Thử nghiệm nhiều mô hình và đặc trưng khác nhau nhằm so sánh hiệu quả giữa các cách biểu diễn và thuật toán phân loại:
- TF-IDF (Raw) + Logistic Regression: Sử dụng Tokenizer, StopWordsRemover, HashingTF và IDF để sinh đặc trưng TF-IDF từ văn bản gốc, sau đó huấn luyện mô hình Logistic Regression.
- TF-IDF (CleanText) + Logistic Regression: Làm sạch văn bản trước khi sinh đặc trưng TF-IDF (xóa URL, ký tự đặc biệt, chuyển chữ thường), giúp giảm nhiễu và cải thiện chất lượng đặc trưng đầu vào cho Logistic Regression.
- Word2Vec (CleanText) + Logistic Regression: Dùng Word2Vec để học vector ngữ nghĩa cho từng từ trong văn bản đã làm sạch, sau đó lấy trung bình vector của câu và huấn luyện bằng Logistic Regression để phân loại cảm xúc.
- TF-IDF (Raw) + Naive Bayes:
Sử dụng văn bản gốc và sinh đặc trưng TF-IDF như pipeline cơ bản, nhưng thay Logistic Regression bằng Naive Bayes để so sánh hiệu quả giữa hai thuật toán phân loại khác nhau.
- TF-IDF (CleanText) + Naive Bayes:
Kết hợp bước làm sạch văn bản với mô hình Naive Bayes, kiểm tra khả năng tận dụng đặc trưng TF-IDF trên dữ liệu đã được chuẩn hóa.

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
- CleanText + TF-IDF + Logistic Regression đạt kết quả cao nhất (Accuracy = 0.7746, F1 = 0.7747), cho thấy việc làm sạch dữ liệu và chuẩn hóa văn bản giúp mô hình học đặc trưng TF-IDF chính xác hơn.
- TF-IDF + Logistic Regression (raw) vẫn đạt kết quả tốt, chứng minh rằng TF-IDF là phương pháp biểu diễn hiệu quả cho bài toán cảm xúc, nhưng dễ bị ảnh hưởng bởi nhiễu trong dữ liệu.
- Word2Vec + Logistic Regression cho hiệu suất thấp hơn do dữ liệu huấn luyện nhỏ, vector ngữ nghĩa chưa ổn định và không nắm bắt tốt ngữ cảnh cảm xúc.
- Naive Bayes hoạt động nhanh và đơn giản nhưng kém hơn Logistic Regression do giả định độc lập giữa các đặc trưng, khiến mô hình không tận dụng được mối quan hệ giữa các từ.
-> Tổng thể, Logistic Regression kết hợp TF-IDF và tiền xử lý tốt là lựa chọn cân bằng giữa độ chính xác, tốc độ và tính ổn định cho bài toán phân loại cảm xúc văn bản.

---

## Tài liệu tham khảo
1. *scikit-learn Documentation*: https://scikit-learn.org/stable/ (Các module LogisticRegression, CountVectorizer, TfidfVectorizer, và các hàm đánh giá như accuracy_score, precision_score, recall_score, f1_score trong phần cài đặt và đánh giá mô hình)
2. *Apache Spark MLlib Guide*: https://spark.apache.org/docs/latest/ml-guide.html (Các thành phần Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec, cùng mô hình LogisticRegression và NaiveBayes trong Spark ML.)

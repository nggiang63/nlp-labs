# Report - Lab 4: Word Embeddings
Giang Nguyen Thi - 22001254 
2025-10-14


---


## Task 1 - Setup


### Mục tiêu
Chuẩn bị môi trường và tải mô hình pre-trained để sử dụng trong các bài thử nghiệm.


### Các bước thực hiện
1) Cài đặt thư viện (trích `requirements.txt`): `gensim`, `pyspark`, `scikit-learn`, `matplotlib`, `numpy`. 
  Cài đặt:
  ```bash
  pip install -r requirements.txt
  ```
2) Tải pre-trained embedding: dùng `gensim.downloader` tải `glove-wiki-gigaword-50` (50 chiều, huấn luyện trên Wikipedia + Gigaword, ~65MB, tự cache lần sau).
  ```python
  import gensim.downloader as api
  model = api.load("glove-wiki-gigaword-50")
  print("Mô hình `glove-wiki-gigaword-50` đã được tải.")
  ```


### Hướng dẫn chạy code
```bash
# tại thư mục test/
python lab4.py
```


### Kết quả thực tế
```
Đang tải mô hình: glove-wiki-gigaword-50 ...
Tải mô hình thành công.
```


---


## Task 2 - Word Embedding Exploration (GloVe Pre-trained)


### Mục tiêu
Thao tác với word vectors: truy vấn vector, tính cosine similarity, tìm most similar.


### Các bước thực hiện
- Viết hàm/lớp trợ giúp:
 - `get_vector(word)` - trả về vector nếu có, nếu OOV trả về zero vector.
 - `get_similarity(w1, w2)` - tính cosine similarity.
 - `get_most_similar(word, topn)` - truy vấn lân cận theo cosine.
- Nguồn vector: `glove-wiki-gigaword-50`.


### Hướng dẫn chạy code
```bash
python lab4.py
```


### Kết quả thực tế
```
--- Vector của 'king' ---
Kích thước vector: (50,)
10 phần tử đầu tiên: [ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
 0.47377  -0.61798  -0.31012 ]


--- Độ tương đồng ---
king - queen: 0.7839043
king - man  : 0.53093773
apple - banana: 0.5607928


--- Các từ gần 'computer' ---
  computers : 0.9165
   software : 0.8815
 technology : 0.8526
 electronic : 0.8126
   internet : 0.8060
```


### Phân tích kết quả
- Độ tương đồng hợp lý: `king-queen` > `king-man` phản ánh quan hệ vai trò cùng miền “royalty”. `apple-banana` ~0.56 thể hiện cùng trường nghĩa trái cây.
- Hàng xóm của `computer` là một cụm công nghệ rõ nét (`computers, software, technology, electronic, internet`) -> chứng tỏ embedding nắm bắt đồng-ngữ cảnh tốt.
- GloVe pre-trained mang tri thức rộng, phù hợp làm baseline tin cậy cho các so sánh sau.


---


## Task 3 - Document Embedding


### Mục tiêu
Biểu diễn câu/đoạn bằng trung bình cộng các word vectors (average pooling).


### Các bước thực hiện
1) Tokenize, chuẩn hóa; loại bỏ các token OOV khi tính trung bình. 
2) Nếu không có token hợp lệ -> trả về zero vector cùng kích thước. 
3) Ngược lại -> trung bình các vector để thu được document embedding (50D).


### Hướng dẫn chạy code
- Đã tích hợp vào `lab4.py`. Chạy:
 ```bash
 python lab4.py
 ```


### Kết quả thực tế 
```
--- Vector biểu diễn câu ---
Kích thước vector: (50,)
10 phần tử đầu tiên: [ 0.04564168  0.36530998 -0.55974334  0.04014383  0.09655549  0.15623933
-0.33622834 -0.12495166 -0.01031508 -0.5006717 ]


--- Kiểm tra từ OOV ---
Từ 'khongtontai' không có trong từ điển của mô hình.
[0. 0. 0. ... 0.]  # Zero vector 50D khi toàn OOV
```


### Phân tích kết quả
- Average pooling cho một vector gọn nhẹ, ổn định, hữu ích cho baseline như so khớp truy vấn-văn bản. 
- Trường hợp OOV dẫn đến zero vector giúp tránh lỗi định dạng; nếu cần giảm OOV, cân nhắc FastText/subword.


---


## Bonus Task - Training Word2Vec on UD English-EWT (Gensim)


### Mục tiêu
Tự huấn luyện Word2Vec để đối chiếu với pre-trained.


### Các bước thực hiện
- Đọc `../data/UD_English-EWT/en_ewt-ud-train.txt`. 
- Huấn luyện Word2Vec theo cấu hình trong script; lưu model ra `../results/lab4/word2vec_ewt.model`. 
- Kiểm tra most_similar và analogy.


### Hướng dẫn chạy code
```bash
python lab4_embedding_training_demo.py
# Saved to: ../results/lab4/word2vec_ewt.model
```


### Kết quả thực tế 
```
Đang đọc dữ liệu từ: ../data/UD_English-EWT/en_ewt-ud-train.txt
Đang huấn luyện mô hình Word2Vec ...
Huấn luyện xong!
Đã lưu mô hình tại: ../results/lab4/word2vec_ewt.model


--- Các từ gần 'king' ---
[('assh', 0.7484), ('shedding', 0.7245), ('snakes', 0.7165),
('gyanendra', 0.7123), ('nepalese', 0.7073)]


--- Phép suy luận ngữ nghĩa ---
king - man + woman ≈ ?
[('shedding', 0.5908), ('meat', 0.5882), ('neat', 0.5842)]
```


### Phân tích kết quả
- Nearest của `king` chứa token hiếm/ngoài miền -> chất lượng kém ổn định do corpus nhỏ và phân bố đồng xuất hiện yếu. 
- Analogy không ra `queen` (thất bại). Cần corpus lớn/sạch hơn, tăng epoch, điều chỉnh `min_count`, `window`, `sg`, subsampling.


---


## Advanced Task - Scaling Word2Vec with Apache Spark


### Mục tiêu
Huấn luyện Word2Vec ở quy mô lớn bằng Spark MLlib và truy vấn synonyms.


### Các bước thực hiện
1) Cài `pyspark`, khởi tạo `SparkSession`. 
2) Đọc dữ liệu JSON lines (lớn), tiền xử lý: lowercase, regex loại ký tự đặc biệt, tokenize. 
3) Huấn luyện `pyspark.ml.feature.Word2Vec` với `vectorSize`, `minCount`. 
4) `findSynonyms("computer", k)` để kiểm tra chất lượng.


### Hướng dẫn chạy code
```bash
python lab4_spark_word2vec_demo.py
```


### Kết quả thực tế  
```
Huấn luyện Word2Vec với Spark hoàn tất.


Các từ tương tự với 'computer':
+---------+------------------+
|word     |similarity        |
+---------+------------------+
|desktop  |0.7037132382392883|
|computers|0.6769197487831116|
|uwowned  |0.6721119785308838|
|device   |0.654563844203949 |
|laptop   |0.6494661569595337|
+---------+------------------+
```


### Phân tích kết quả
- Phần lớn hợp lý (desktop/computers/device/laptop), có nhiễu (*uwowned*) do dữ liệu thô. 
- Ưu điểm của Spark là mở rộng dữ liệu và tốc độ; chất lượng vẫn phụ thuộc tiền xử lý và chọn nguồn corpus. 
- Nếu gặp lỗi bộ nhớ: sample dữ liệu và tăng `spark.driver.memory` (ví dụ `4g`).


---


## Visualization - PCA


> Phần trực quan hóa và phân tích ở notebook/script riêng (`src/lab4/lab4_visualization.py`).


---


## So sánh tổng hợp


| Tiêu chí | Pre-trained (GloVe 50d) | Word2Vec tự huấn luyện (UD EWT) | Word2Vec Spark |
|---|---|---|---|
| Nguồn dữ liệu | Wikipedia + Gigaword | Corpus nhỏ, đa mục đích | Lớn, phân tán (JSON lines) |
| Chất lượng lân cận | Ổn định, đúng miền | Nhiễu, tên riêng/hiếm | Khá tốt, nhưng phụ thuộc preprocessing |
| Analogy | Thường đúng | Thường fail | Phụ thuộc dữ liệu/quy mô |
| Dùng khi nào | Baseline mạnh | Tùy biến theo miền | Quy mô lớn / tốc độ huấn luyện |


Kết luận ngắn: Pre-trained là chuẩn so sánh đáng tin; tự huấn luyện cần dữ liệu lớn để đạt chất lượng; Spark giúp mở rộng quy mô, nhưng vẫn cần tiền xử lý kỹ.


---


## Khó khăn & giải pháp


| Vấn đề gặp phải | Nguyên nhân | Giải pháp áp dụng |
|---|---|---|
| Tải GloVe lần đầu chậm | Download ~65MB | Dùng cache `gensim`, kiểm tra mạng, retry |
| OOV nhiều ở một số câu | Từ hiếm/không có trong vocab | Bỏ OOV khi average; cân nhắc FastText/subword |
| Word2Vec (gensim) cho nearest/analogy nhiễu | Corpus nhỏ, tần suất thấp | Tăng dữ liệu/epoch; chỉnh `min_count`, `window`, `sg`, subsampling |
| Spark xuất hiện token rác (`uwowned`) | Văn bản thô, regex nhẹ | Làm sạch mạnh hơn (regex, lọc tần suất, stopwords), chuẩn hóa |


---


## Tài liệu tham khảo


1. Pennington, J., Socher, R., & Manning, C. D. (2014). 
  *GloVe: Global Vectors for Word Representation.* 
  Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 
  - Website: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) 
  - Paper PDF: [https://aclanthology.org/D14-1162.pdf](https://aclanthology.org/D14-1162.pdf)


2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). 
  *Efficient Estimation of Word Representations in Vector Space.* 
  arXiv preprint arXiv:1301.3781. 
  - Link: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)


3. Gensim Documentation. 
  *Using Pretrained Word Embeddings and KeyedVectors.* 
  - Tutorial: [https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) 
  - API Reference: [https://radimrehurek.com/gensim/models/keyedvectors.html](https://radimrehurek.com/gensim/models/keyedvectors.html)


4. PySpark MLlib Documentation. 
  *Word2Vec API for feature learning of word embeddings.* 
  - Link: [https://spark.apache.org/docs/latest/ml-features.html#word2vec](https://spark.apache.org/docs/latest/ml-features.html#word2vec)


5. scikit-learn Documentation. 
  *Dimensionality reduction techniques: PCA* 
  - PCA: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 
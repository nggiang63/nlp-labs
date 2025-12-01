# Natural Language Processing (NLP) Labs

## 1. Thông tin sinh viên

- Họ và tên: Nguyễn Thị Giang  
- Mã sinh viên: 22001254  
- Lớp: K67A5 - Khoa học Dữ liệu  
- Học phần: Xử lý ngôn ngữ tự nhiên và học sâu

---

## 2. Mục tiêu repository

Repository này lưu trữ toàn bộ mã nguồn, notebook và báo cáo cho các Lab môn NLP&DL.  
Mục tiêu chính:

- Chuẩn hóa cấu trúc repo theo hướng dẫn của học phần (tách riêng `src/`, `report/`, `notebook/`, `test/`, `data/`, `README.md`, `.gitignore`)   
- Mỗi Lab có báo cáo chi tiết, giúp người xem hiểu rõ cách làm, thí nghiệm và kết quả.  
- Dễ dàng chạy lại, tái hiện các thực nghiệm (reproducible).  

---

## 3. Cấu trúc thư mục

```text
nlp-labs/
│
├── src/                  # Toàn bộ code chính
│   ├── core/             # Logic chính của từng Lab (model, training, evaluation, ...)
│   ├── preprocessing/    # Tiền xử lý văn bản: tokenizer, làm sạch dữ liệu, ...
│   ├── utils/            # Hàm tiện ích dùng chung
│   └── ...               
│
├── notebook/             # Bao gồm notebook của một số Lab
│   ├── lab1.ipynb
│   └── ...               
│
├── report/               # Báo cáo chi tiết cho từng Lab
│   ├── lab1.md
│   └── ...               
│
├── test/                 # Script / test cho các Lab
│   ├── test_lab1.py
│   ├── test_lab2.py
│   └── ...
│
├── requirements.txt      # Danh sách thư viện Python cần cài đặt
├── README.md             # Giới thiệu tổng quan repository 
└── .gitignore            # Loại bỏ file không cần track 
```

## 4. Nội dung các Lab

Dưới đây là tóm tắt nội dung chính từng Lab trong môn NLP:

---

### Lab 1 - Tokenizer cơ bản

- Cài đặt và so sánh các bộ tách từ cơ bản:
  - `SimpleTokenizer` (tách theo khoảng trắng, ký tự đơn giản).
  - `RegexTokenizer` (sử dụng biểu thức chính quy để tách từ/chấm câu).
- Quan sát ưu - nhược điểm của từng cách tokenization.

---

### Lab 2 - Biểu diễn văn bản bằng Bag-of-Words

- Cài đặt `CountVectorizer` để:
  - Biểu diễn văn bản dưới dạng vector số (Bag-of-Words).
  - Xây dựng vocabulary, đếm tần suất xuất hiện từ.
- Chuẩn bị đầu vào cho các mô hình học máy đơn giản.

---

### Lab 3 - Word Embedding (Word2Vec)

- Biểu diễn từ dưới dạng dense vector (word embedding).
- Huấn luyện hoặc sử dụng mô hình `Word2Vec`:
  - So sánh các từ gần nhau trong không gian vector.
  - Minh họa khả năng bắt quan hệ ngữ nghĩa và ngữ pháp giữa các từ.

---

### Lab 4 - Text Classification

- Bài toán phân loại văn bản sử dụng pipeline:
  - Các bước tiền xử lý từ các Lab trước (tokenizer, BoW, TF-IDF hoặc embedding).
  - Các mô hình học máy đơn giản (Logistic Regression, SVM, Naive Bayes).
- Đánh giá mô hình qua các metric:
  - accuracy, precision, recall, F1-score.

---

### Lab 5 - RNNs, LSTMs, GRUs cho token classification

- Sử dụng các kiến trúc tuần tự:
  - RNN, LSTM, GRU.
- Ứng dụng cho các bài toán:
  - POS tagging (gán nhãn từ loại).
  - NER (Named Entity Recognition - nhận dạng thực thể).
  - Các bài toán token classification khác.
- So sánh chất lượng giữa các kiến trúc (RNN thuần vs LSTM/GRU).

---

### Lab 6 - Introduction to Transformers

- Làm quen với các mô hình Transformer pretrained:
  - Ví dụ: BERT, RoBERTa hoặc các mô hình tương tự.
- Ứng dụng cho tác vụ NLP cơ bản:
  - Phân loại văn bản.
  - Token classification (POS/NER).
- Thực hành:
  - Fine-tuning mô hình Transformer trên dataset nhỏ.
  - So sánh hiệu quả với mô hình ở Lab 4 và Lab 5.

---

## 5. Cách cài đặt và chạy

### Cài đặt môi trường Python

```bash
pip install -r requirements.txt
```

### Chạy code cho từng Lab

Ví dụ:

```bash
python test/lab1.py
...
```

Hoặc chạy trực tiếp trong notebook:

```bash
notebook/lab5.ipynb
...
```
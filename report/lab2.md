# Lab 2: Count Vectorization

Giang Nguyen Thi - 22001254  

2025-09-16  

---

## 1. Mục tiêu và mô tả công việc

Count Vectorization là bước quan trọng trong pipeline NLP giúp biến đổi văn bản thành dạng số (vector) để làm đầu vào cho các thuật toán học máy.  
Trong lab này, mục tiêu bao gồm:

- Hiểu cơ chế làm việc của Bag-of-Words.
- Hiện thực Vectorizer Interface với 3 phương thức:
  ```python
  fit(self, corpus: list[str])
  transform(self, documents: list[str]) -> list[list[int]]
  fit_transform(self, corpus: list[str]) -> list[list[int]]
  ```
- Xây dựng CountVectorizer (`src/representations/count_vectorizer.py`):
  - Nhận một tokenizer từ Lab 1.
  - Học vocabulary từ corpus.
  - Sinh document-term matrix bằng cách đếm tần suất token.
- Viết script demo (`labs/lab2_vectorization.py`) để thử nghiệm:
  - Với corpus đơn giản.
  - Với dataset UD_English-EWT.

---

## 2. Các bước triển khai

### 2.1. Interface Vectorizer
File: `src/core/interfaces.py`  
Interface chuẩn cho mọi vectorizer.

### 2.2. CountVectorizer
- Tokenize từng văn bản bằng tokenizer từ Lab 1.  
- Xây dựng vocabulary:
  - Duyệt toàn bộ corpus.
  - Lưu các token duy nhất vào dictionary dạng `{token: index}`.
- Biến đổi văn bản thành vector số:
  - Khởi tạo vector có kích thước bằng số lượng token trong vocabulary.
  - Mỗi lần gặp một token, tăng giá trị tại index tương ứng.
- Hỗ trợ 3 hàm:
  - `fit` → học vocabulary  
  - `transform` → sinh vector cho documents mới  
  - `fit_transform` → kết hợp cả hai  

### 2.3. Chạy file test
Chạy thử:
- Corpus mẫu trong lab.
- Một đoạn văn bản từ UD_English-EWT và in ra kích thước vocabulary & vector mẫu.

---

## 3. Hướng dẫn chạy code 

### 3.1. Môi trường
- Python 3.10+  
- Đã cài tokenizer của Lab 1  

Kiểm tra Python:

```bash
python3 --version
```

---

### 3.2. Chạy script test

```bash
python test/lab2.py
```

Nếu gặp lỗi import, thêm vào đầu file:

```python
import sys, os
sys.path.append(os.getcwd())
```

---

## 4. Kết quả chạy code

### Ví dụ corpus
Corpus:
```python
["I love NLP.", "I love programming.", "NLP is a subfield of AI."]
```

### Vocabulary:
```
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
```

### Document-Term Matrix:
```
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

### Ví dụ UD_English-EWT (minh họa)
- Vocabulary size: ~500-2000 token (tuỳ đoạn lấy).  
- Vector mẫu (rút gọn):

```
[0, 1, 0, 0, 2, 0, 1, 0, 0, ...]
```

---

## 5. Phân tích kết quả

### 5.1. Vocabulary
- Là danh sách tất cả token duy nhất xuất hiện trong corpus.
- Thứ tự token phụ thuộc vào quá trình duyệt corpus.

### 5.2. Document-Term Matrix
- Mỗi văn bản được biểu diễn dưới dạng vector đếm tần suất.
- Ma trận này được dùng làm đầu vào cho ML models:
  - Logistic Regression  
  - Naive Bayes  
  - SVM  
  - MLP  

### 5.3. Nhận xét
- CountVectorizer đơn giản nhưng hiệu quả với văn bản nhỏ.
- Không hiểu ngữ nghĩa (semantic), chỉ đếm số lượng.

---

## 6. Khó khăn và cách giải quyết

### 1. Vocabulary quá lớn khi corpus lớn
- Vấn đề: tăng mạnh kích thước vector.
- Giải pháp:
  - Loại bỏ stopwords.
  - Giới hạn vocabulary theo tần suất.

### 2. Ma trận thưa (Sparse Matrix)
- CountVectorizer tạo ma trận nhiều số 0.
- Giải pháp: dùng cấu trúc `scipy.sparse` khi implement thực tế (ngoài phạm vi lab).

### 3. Tokenizer ảnh hưởng trực tiếp đến vocabulary
- SimpleTokenizer vs RegexTokenizer → kết quả rất khác nhau.
- Giải pháp: thống nhất tokenizer trước khi vector hoá.

### 4. Dữ liệu thật (UD-EWT) có token lạ
- Giải pháp: normalize lowercase và lọc token đặc biệt.

### 5. Debug sai khác thứ tự token
- Giải pháp: cố định cách duyệt corpus khi build vocabulary.

---

## 7. Tài liệu tham khảo

1. Python Software Foundation. _Python Regular Expression Documentation._  
   https://docs.python.org/3/library/re.html

2. Scikit-learn Developers. _CountVectorizer Documentation._  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

3. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze. _Introduction to Information Retrieval - Vector Space Model Section._  
   https://nlp.stanford.edu/IR-book/

4. HuggingFace. _Text Vectorization Overview._  
   https://huggingface.co/docs/transformers/main/en/preprocessing

---

# Lab 2: Count Vectorization
**Giang Nguyen Thi - 22001254**  
2025-09-16  

## 1. Mô tả công việc
- Cài đặt interface `Vectorizer` trong `src/core/interfaces.py` với 3 phương thức:
  ```python
  fit(self, corpus: list[str])
  transform(self, documents: list[str]) -> list[list[int]]
  fit_transform(self, corpus: list[str]) -> list[list[int]]
  ```
- Xây dựng **CountVectorizer** (`src/representations/count_vectorizer.py`):
  - Nhận một tokenizer từ Lab 1.
  - Học **vocabulary** từ corpus.
  - Biến đổi corpus thành **document-term matrix**.
- Tạo script demo (`labs/lab2_vectorization.py`) để thử nghiệm:
  - Với corpus mẫu (3 câu ngắn).
  - Với dataset **UD_English-EWT**.

---

## 2. Kết quả chạy code

### Ví dụ corpus
Corpus:
```python
["I love NLP.", "I love programming.", "NLP is a subfield of AI."]
```

- Vocabulary:  
```
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
```

- Document-Term Matrix:  
```
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

---

## 3. Giải thích kết quả
- Vocabulary: tập hợp tất cả token duy nhất xuất hiện trong corpus/dataset.  
- Document-term matrix: mỗi hàng là một văn bản, mỗi cột là một token, giá trị là số lần xuất hiện.  
- **CountVectorizer** biến đổi văn bản thành vector số => đầu vào cho mô hình ML (Naive Bayes, Logistic Regression, SVM…).  
- **Khó khăn**: corpus lớn => vocabulary lớn, cần xử lý ma trận thưa để tiết kiệm bộ nhớ.  

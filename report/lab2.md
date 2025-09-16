# Lab 2: Count Vectorization
**Giang Nguyen Thi**  
2025-09-16  

## Objective
Biểu diễn văn bản thành vector số bằng mô hình **Bag-of-Words**, sử dụng lớp `CountVectorizer`.  

## Tasks
1. **Vectorizer Interface**  
   - Trong `src/core/interfaces.py`, định nghĩa abstract class `Vectorizer` với các phương thức:
     - `fit(self, corpus: list[str])`
     - `transform(self, documents: list[str]) -> list[list[int]]`
     - `fit_transform(self, corpus: list[str]) -> list[list[int]]`

2. **CountVectorizer**  
   - File: `src/representations/count_vectorizer.py`  
   - Inherit từ `Vectorizer`  
   - `__init__(self, tokenizer: Tokenizer)` nhận 1 tokenizer (Lab 1)  
   - Có thuộc tính `vocabulary_` (dict[str, int])  
   - `fit`: xây vocabulary từ corpus  
   - `transform`: chuyển document thành vector đếm theo vocabulary  

3. **Evaluation**  
   - File: `labs/lab2_count_vectorization.py`  
   - Dùng `RegexTokenizer` và `CountVectorizer`  
   - Corpus mẫu:
     ```python
     [
       "I love NLP.",
       "I love programming.",
       "NLP is a subfield of AI."
     ]
     ```
   - In vocabulary và document-term matrix  

## Deliverables
- Code trong repo GitHub (public).  
- File `README.md` hoặc `Report.md` mô tả công việc, kết quả chạy code, giải thích **cách CountVectorizer hoạt động** và ý nghĩa ma trận document-term.  

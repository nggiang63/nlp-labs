# Lab 1: Text Tokenization

Giang Nguyen Thi - 22001254 

2025-09-16  

--- 

## 1. Mục tiêu chung

Tokenization (tách từ) là bước đầu tiên và quan trọng trong hầu hết các bài toán xử lý ngôn ngữ tự nhiên (NLP).  
Mục tiêu của Lab 1:

- Hiểu vai trò của tokenization trong pipeline NLP  
- Hiện thực Tokenizer Interface  
- Xây dựng hai tokenizer:  
  - SimpleTokenizer - tách đơn giản dựa vào khoảng trắng và dấu câu  
  - RegexTokenizer - tách chi tiết bằng regex  
- So sánh kết quả trên câu mẫu và dataset UD_English-EWT

Trong lab này, việc triển khai được chia thành bốn phần chính: xây dựng interface Tokenizer, hiện thực SimpleTokenizer, hiện thực RegexTokenizer và đánh giá hai phương pháp trên dataset thật. Các phần tiếp theo sẽ mô tả chi tiết từng bước cài đặt.

--- 

## 2. Các bước triển khai

### 2.1. Tokenizer Interface 
- Định nghĩa abstract class `Tokenizer` trong `src/core/interfaces.py` với phương thức:
  ```python
  def tokenize(self, text: str) -> list[str]:
  ```

### 2.2. SimpleTokenizer
- File: `src/preprocessing/simple_tokenizer.py`  
- Chuyển text thành lowercase  
- Tách từ theo khoảng trắng  
- Xử lý dấu câu cơ bản (.,?! => tách riêng)  

### 2.3. RegexTokenizer
- File: `src/preprocessing/regex_tokenizer.py`  
- Dùng regex `\w+|[^\w\s]` để tách token, robust hơn.  

### 2.4. Evaluation  
- Test với 3 câu:  
  - `"Hello, world! This is a test."`  
  - `"NLP is fascinating... isn't it?"`  
  - `"Let's see how it handles 123 numbers and punctuation!"`  
- Tokenize sample text từ **UD_English-EWT** dataset và in 20 tokens đầu tiên.  

---

## 3. Hướng dẫn chạy code (Code Execution Guide)

### 3.1. Môi trường
- Python 3.10+  
- Không dùng thư viện ngoài  

Kiểm tra:

```bash
python3 --version
```

---

### 3.2. Chạy file test

```bash
python test/lab1_test.py
```

Nếu lỗi import, thêm vào đầu file test:

```python
import sys, os
sys.path.append(os.getcwd())
```

---

## 4. Kết quả thực nghiệm

### 4.1. Câu test 1  
Input: "Hello, world! This is a test."

| Tokenizer        | Output |
|------------------|--------|
| SimpleTokenizer  | ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.'] |
| RegexTokenizer   | ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.'] |

---

### 4.2. Câu test 2  
Input: "NLP is fascinating... isn't it?"

| Tokenizer        | Output |
|------------------|--------|
| SimpleTokenizer  | ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?'] |
| RegexTokenizer   | ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?'] |

---

### 4.3. Câu test 3  
Input: "Let's see how it handles 123 numbers and punctuation!"

| Tokenizer        | Output |
|------------------|--------|
| SimpleTokenizer  | ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!'] |
| RegexTokenizer   | ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!'] |

---

## 5. Phân tích kết quả

### SimpleTokenizer
Ưu điểm
- Dễ cài đặt  
- Nhanh  

Nhược điểm
- Không tách tốt từ chứa dấu (`al-ani,`, "isn't")  
- Không phù hợp NLP hiện đại  

---

### RegexTokenizer
Ưu điểm
- Tách chi tiết hơn  
- Nhận diện dấu câu, ký tự đặc biệt tốt  
- Phù hợp xử lý văn bản phức tạp  

Nhược điểm
- Dễ tách quá mức ("isn't" → isn, ', t)

---

## 6. Khó khăn và cách giải quyết

### 1. Thiết kế interface
Khó xác định cấu trúc chung.  
Giải pháp: dùng abstract class để chuẩn hoá.

### 2. SimpleTokenizer xử lý dấu câu hạn chế  
Giải pháp: dùng regex chèn khoảng trắng quanh dấu.

### 3. RegexTokenizer tách ký tự ' không mong muốn  
Giải pháp: ghi nhận đây là hành vi regex.

### 4. Dataset UD_English-EWT có ký tự lạ  
Giải pháp: chuẩn hoá bằng `.lower()` và regex.

---

## 7. Tài liệu tham khảo

1. Python Software Foundation. Python Regular Expression Documentation.
https://docs.python.org/3/library/re.html
2. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze. Introduction to Information Retrieval - Tokenization Section. Stanford NLP Group.
https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html
3. HuggingFace. Tokenizer Summary - Overview of Modern Tokenization Methods (BPE, WordPiece, SentencePiece).
https://huggingface.co/docs/transformers/tokenizer_summary 

---
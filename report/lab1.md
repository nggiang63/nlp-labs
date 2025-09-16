# Lab 1: Text Tokenization
**Giang Nguyen Thi - 22001254**  
2025-09-16  

## 1. Mô tả công việc

### 1.1. Tokenizer Interface 
- Định nghĩa abstract class `Tokenizer` trong `src/core/interfaces.py` với phương thức:
  ```python
  def tokenize(self, text: str) -> list[str]:
  ```

### 1.2. SimpleTokenizer
- File: `src/preprocessing/simple_tokenizer.py`  
- Chuyển text thành lowercase  
- Tách từ theo khoảng trắng  
- Xử lý dấu câu cơ bản (.,?! => tách riêng)  

### 1.3. RegexTokenizer
- File: `src/preprocessing/regex_tokenizer.py`  
- Dùng regex `\w+|[^\w\s]` để tách token, robust hơn.  

### 1.4. Evaluation  
- Test với 3 câu:  
  - `"Hello, world! This is a test."`  
  - `"NLP is fascinating... isn't it?"`  
  - `"Let's see how it handles 123 numbers and punctuation!"`  
- Tokenize sample text từ **UD_English-EWT** dataset và in 20 tokens đầu tiên.  

---

## 2. Kết quả chạy code

### 2.1. Ví dụ câu test
1. `"Hello, world! This is a test."`  
   - SimpleTokenizer => `['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']`  
   - RegexTokenizer => `['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']`  

2. `"NLP is fascinating... isn't it?"`  
   - SimpleTokenizer => `['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']`  
   - RegexTokenizer => `['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']`  

3. `"Let's see how it handles 123 numbers and punctuation!"`  
   - SimpleTokenizer => `["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']`  
   - RegexTokenizer => `['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']`  

### 2.2. Ví dụ dataset UD_English-EWT (20 tokens đầu)
- SimpleTokenizer =>  
  `['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani,', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim,', 'near', 'the']`  

- RegexTokenizer =>  
  `['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']`  

---

## 3. Giải thích kết quả
- **SimpleTokenizer**: đơn giản, dễ dùng, nhưng giữ nguyên từ dính dấu câu (`"al-ani,"`, `"isn't"`).  
- **RegexTokenizer**: tách chi tiết hơn, đặc biệt với ký tự `'` và `-`, giúp xử lý tốt hơn cho các tác vụ NLP cần token chính xác.  
- So sánh cho thấy lựa chọn tokenizer ảnh hưởng trực tiếp đến số lượng và chất lượng token.  
- **Khó khăn**: RegexTokenizer đôi khi tách quá chi tiết (ví dụ `"isn't"` => `['isn', "'", 't']`).  
- Bài học: Tokenization là bước tiền xử lý quan trọng, ảnh hưởng đến mọi bước tiếp theo trong pipeline NLP (Vectorization, Embedding, Model Training).  
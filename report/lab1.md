# Lab 1: Text Tokenization
**Giang Nguyen Thi**  
2025-09-16  

## Objective
Hiểu và cài đặt bước tiền xử lý cơ bản trong NLP: **Tokenization**.  

## Tasks
1. **Tokenizer Interface**  
   - Định nghĩa abstract class `Tokenizer` trong `src/core/interfaces.py` với phương thức:
     ```python
     def tokenize(self, text: str) -> list[str]:
     ```
2. **SimpleTokenizer**  
   - File: `src/preprocessing/simple_tokenizer.py`  
   - Chuyển text thành lowercase  
   - Tách từ theo khoảng trắng  
   - Xử lý dấu câu cơ bản (.,?! → tách riêng)  

3. **RegexTokenizer (Bonus)**  
   - File: `src/preprocessing/regex_tokenizer.py`  
   - Dùng regex `\w+|[^\w\s]` để tách token, robust hơn.  

4. **Evaluation**  
   - File: `labs/lab1_tokenization.py`  
   - Test với 3 câu:  
     - `"Hello, world! This is a test."`  
     - `"NLP is fascinating... isn't it?"`  
     - `"Let's see how it handles 123 numbers and punctuation!"`  
   - Tokenize sample text từ **UD_English-EWT** dataset và in 20 tokens đầu tiên.  

## Deliverables
- Code trong repo GitHub (public).  
- File `README.md` hoặc `Report.md` mô tả công việc, kết quả chạy code, giải thích so sánh **SimpleTokenizer vs RegexTokenizer**.  

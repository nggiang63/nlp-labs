# Lab 6: Giới thiệu về Transformers

Giang Nguyen Thi - 22001254

2025-11-18

## **Bài 1: Khôi phục Masked Token (Masked Language Modeling)**

**Mục tiêu**
- Làm quen với pipeline `fill-mask` của HuggingFace.
- Dự đoán một token bị che `<mask>`.
- Hiểu vì sao các mô hình Encoder-only (BERT, RoBERTa…) phù hợp với MLM.

**Cách thực hiện**
- Import pipeline từ Transformers.
- Khởi tạo pipeline `"fill-mask"` (mặc định DistilRoBERTa-base).
- Chuẩn bị câu:
  ```
  Hanoi is the <mask> of Vietnam.
  ```
- Mô hình dự đoán 5 token có xác suất cao nhất.
- Ghi nhận token_str và score.

- Lấy top 5 dự đoán.

**Kết quả**
- capital (0.9341)
- Republic (0.0300)
- Capital (0.0105)
- birthplace (0.0054)
- heart (0.0014)

**Nhận xét**
- Mô hình dự đoán đúng token "capital".
- BERT là mô hình bidirectional nên phù hợp với MLM.

**Trả lời câu hỏi**
1. Mô hình đã dự đoán đúng từ capital không?

-> Có. Trong 5 kết quả mô hình trả về, "capital" có độ tin cậy 0.9341, cao nhất và đúng ngữ nghĩa trong câu "Hanoi is the capital of Vietnam."

2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?

-> Vì:
- Self-attention hai chiều (bidirectional) => nhìn được cả ngữ cảnh bên trái và phải của token `<mask>`.  
- MLM là nhiệm vụ pretraining của BERT => mô hình được tối ưu để dự đoán từ bị che.  
- Encoder-only tập trung vào "hiểu" câu => rất phù hợp cho tác vụ điền khuyết từ.
---

## **Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)**

**Mục tiêu**
- Sinh tiếp văn bản từ một câu mồi bằng GPT-2.
- Hiểu vì sao decoder-only phù hợp với text generation.

**Cách thực hiện**
- Dùng pipeline `"text-generation"`.
- Sử dụng mô hình GPT-2 (mặc định).
- Prompt:
```
The best thing about learning NLP is
```

**Kết quả**
Mô hình sinh một đoạn văn dài, mạch lạc và liên quan đến NLP.

**Nhận xét**
- GPT-2 tạo văn bản hợp lý.
- Decoder-only phù hợp vì được pretrain bằng next-token prediction.

**Trả lời câu hỏi**
1. Kết quả sinh ra có hợp lý không?

-> Có, kết quả hợp lý - ở mức ngữ pháp đúng, câu trôi chảy, và bám sát chủ đề.

2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?

-> Vì chúng được huấn luyện theo mục tiêu dự đoán token kế tiếp (next-token prediction), sử dụng cơ chế causal attention một chiều. Điều này cho phép mô hình sinh văn bản theo trình tự, giữ mạch câu và chủ đề tốt hơn so với các mô hình Encoder-only như BERT, vốn được thiết kế cho nhiệm vụ hiểu và điền khuyết thay vì sinh văn bản.

---

## **Bài 3: Tính toán Vector biểu diễn của câu (Sentence Representation)**

**Mục tiêu**
- Tính embedding câu bằng BERT.
- Thực hành Mean Pooling.

**Cách thực hiện**
- Load tokenizer và model BERT-base-uncased.
- Tokenize câu => lấy `last_hidden_state`.
- Thực hiện Mean Pooling:
  - Nhân embedding với attention_mask
  - Chia cho tổng số token thật
- Nhận vector cuối cùng dạng `(1, hidden_size)`.

**Kết quả**
- Vector có dạng tensor([...])
- Kích thước: `torch.Size([1, 768])`

**Nhận xét**
- 768 là hidden_size của bert-base.
- Attention mask giúp bỏ padding khi tính trung bình.

**Trả lời câu hỏi**
1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?

- Kích thước (chiều) của vector biểu diễn là 768.
- Con số này tương ứng với tham số hidden_size của mô hình BERT-base-uncased, tức là số chiều của vector ẩn mà mô hình sinh ra cho mỗi token trong lớp cuối của Transformer.

2. Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling?

Nếu không loại bỏ padding, mean pooling sẽ tính trung bình luôn cả các vector [PAD] - những vector này không mang ý nghĩa ngữ nghĩa và có giá trị gần 0 => làm hỏng embedding.

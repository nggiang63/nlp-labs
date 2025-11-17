# **Lab 5: Xây dựng mô hình RNN cho bài toán Part-of-Speech Tagging**

Giang Nguyen Thi - 22001254   
2025-11-17
---

Toàn bộ code thực thi cho các mô hình được đặt tại:  
**`nlp-labs/notebook/lab5_rnns_pos_tagging.ipynb`**

---

## **Task 1: Tải và Tiền xử lý Dữ liệu**

**Mục tiêu**
- Đọc dữ liệu CoNLL-U.
- Tách dữ liệu thành danh sách các câu, mỗi câu là các cặp (word, upos_tag).

**Cách thực hiện**
- Viết hàm load_conllu:
  - Bỏ dòng metadata (#)
  - Loại token dạng multiword (1-2, 3.1)
  - Tách dữ liệu thành danh sách câu phục vụ training

**Kết quả**
Train sentences: 12544  
Dev sentences: 2001

Ví dụ câu đầu:
[('Al', 'PROPN'), ('-', 'PUNCT'), ('Zaman', 'PROPN'), ('American', 'ADJ'), ('forces', 'NOUN'), ...]

---

## **Task 2: Tạo PyTorch Dataset và DataLoader**

**Mục tiêu**
- Xây dựng vocabulary cho word và POS-tag
- Tạo lớp Dataset tùy chỉnh
- Tạo DataLoader có padding động bằng collate_fn

**Cách thực hiện**

**Vocabulary**
- word_to_ix: 19675 từ
- tag_to_ix: 18 POS-tag
- Thêm token "<PAD>" và "<UNK>"

**Dataset**
- Trả về tensor word_ids và tag_ids

**DataLoader**
- Pad các câu theo độ dài lớn nhất trong batch
- Trả về lengths cho pack_padded_sequence

**Kết quả**
Word vocab size: 19675  
Tag vocab size: 18

---

## **Task 3: Xây dựng Mô hình RNN**

**Mục tiêu**
- Xây dựng mô hình gồm:
  1. Embedding
  2. RNN
  3. Linear (token classification)

**Cách thực hiện**
- Embedding: 128 chiều
- RNN: 128 chiều ẩn, batch_first
- Linear: ánh xạ từ 128 → 18 tags

**Nhận xét**
- Mô hình đơn giản nhưng đủ hiệu quả.
- RNN chưa phải bidirectional nhưng cho kết quả tốt.

---

## **Task 4: Huấn luyện Mô hình**

**Mục tiêu**
- Huấn luyện bằng CrossEntropyLoss
- Bỏ qua padding khi tính loss

**Cách thực hiện**
- Optimizer: Adam
- Loss: CrossEntropyLoss(ignore_index=PAD)
- Train 5 epochs

**Kết quả huấn luyện**
Epoch 1/5 - Loss: 1.0720  
Epoch 2/5 - Loss: 0.5835  
Epoch 3/5 - Loss: 0.4306  
Epoch 4/5 - Loss: 0.3374  
Epoch 5/5 - Loss: 0.2714  

**Nhận xét**
- Loss giảm đều → mô hình học tốt
- Loss ~0.27 sau 5 epoch là hợp lý với RNN một chiều

---

## **Task 5: Đánh giá Mô hình**

**Mục tiêu**
- Tính accuracy trên train/dev
- Chỉ tính token không phải padding

**Kết quả**
Train accuracy: 0.9307497910322274  
Dev accuracy: 0.8552626346972046  

**Dự đoán câu mới**

**Câu**: "I love NLP"
[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'VERB')]

**Câu**:
They will travel to Japan tomorrow
[('They', 'PRON'), ('will', 'AUX'), ('travel', 'VERB'), ('to', 'ADP'), ('Japan', 'PROPN'), ('tomorrow', 'NOUN')]

**Câu**:
This movie is absolutely fantastic
[('This', 'DET'), ('movie', 'NOUN'), ('is', 'AUX'), ('absolutely', 'ADV'), ('fantastic', 'ADJ')]

**Câu**:
Students are studying in the library
('Students', 'VERB') bị sai do từ hiếm

**Nhận xét**
- Dev accuracy ~85.5% → tốt với RNN đơn giản
- Sai ở từ hiếm hoặc plural

---

## **Kết luận**
- Hoàn thành đầy đủ 5 Task theo yêu cầu Lab
- Mô hình hoạt động tốt, pipeline hoàn chỉnh từ load data → train → evaluate → predict
- Có thể nâng cấp bằng BiLSTM, CRF hoặc pretrained embeddings

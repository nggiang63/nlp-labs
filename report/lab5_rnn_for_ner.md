# **Lab5: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)**

Giang Nguyen Thi - 22001254

2025-11-18

## **Task 1: Tải và Tiền xử lý Dữ liệu**

**Mục tiêu**
- Tải bộ dữ liệu CoNLL-2003 cho bài toán NER.
- Tiền xử lý dữ liệu theo chuẩn IOB2.
- Xây dựng vocabulary cho từ và nhãn.

**Cách thực hiện**
- Viết hàm đọc dữ liệu CoNLL theo định dạng word–POS–CHUNK–NER.
- Tải file train/valid/test và gom câu theo dòng rỗng.
- Tạo từ điển word_to_ix và tag_to_ix.
- Thêm token đặc biệt <PAD> và <UNK>.

**Kết quả**
- Train sentences: 14987  
- Valid sentences: 3466  
- Test sentences: 3684  
- Word vocab size: 23626  
- Tag vocab size: 10

**Nhận xét**
- Bộ dữ liệu chuẩn NER, đủ 4 loại thực thể.
- Vocabulary lớn => NER là bài toán khó.
- Dữ liệu đúng format nên không cần chỉnh sửa thêm.

## **Task 2: Tạo PyTorch Dataset và DataLoader**

**Mục tiêu**
- Chuyển dữ liệu văn bản sang dạng tensor.
- Xử lý padding cho các chuỗi độ dài khác nhau.

**Cách thực hiện**
- Tạo lớp NERDataset trả về word_ids và tag_ids.
- Viết collate_fn để pad câu theo độ dài lớn nhất batch.
- Lưu lengths để dùng với pack_padded_sequence.

**Kết quả**
- Tạo thành công train_loader và valid_loader với batch size 32.
- Padding đúng cho cả câu và nhãn.

**Nhận xét**
- Dataset và DataLoader hoạt động đúng chuẩn token classification.
- Pad_sequence giúp xử lý biến độ dài hiệu quả.

## **Task 3: Xây dựng Mô hình RNN**

**Mục tiêu**
- Xây dựng mô hình RNN đơn giản cho NER.

**Cách thực hiện**
- Mô hình gồm:
  - Embedding layer (128 chiều)
  - RNN layer (hidden_dim = 128)
  - Linear layer ánh xạ sang số lượng nhãn
- Sử dụng pack/unpack để tránh tính padding.

**Kết quả**
- Mô hình được khởi tạo đúng tham số:
  - vocab_size = 23626  
  - embedding_dim = 128  
  - hidden_dim = 128  
  - num_tags = 10

**Nhận xét**
- RNN hoạt động đúng nhưng chưa phải lựa chọn tối ưu cho NER.
- Bi-LSTM có thể cải thiện kết quả hơn.

## **Task 4: Huấn luyện Mô hình**

**Mục tiêu**
- Huấn luyện mô hình RNN trên tập train.

**Cách thực hiện**
- Loss: CrossEntropyLoss(ignore_index = PAD_TAG)
- Optimizer: Adam, lr=0.001
- 3 epoch
- Forward => Loss => Backward => Update

**Kết quả**
- Epoch 1 - Loss: 0.1976  
- Epoch 2 - Loss: 0.1391  
- Epoch 3 - Loss: 0.1000  

**Nhận xét**
- Loss giảm đều => mô hình học tốt.
- Giá trị loss thấp hợp lý với mô hình đơn giản.

## **Task 5: Đánh giá Mô hình**

**Mục tiêu**
- Đánh giá bằng Accuracy (token-level)
- Đánh giá bằng Precision–Recall–F1 (entity-level, seqeval)
- Dự đoán câu mới.

**Cách thực hiện**
- So sánh dự đoán và nhãn thật trên token không phải padding.
- Dùng classification_report của seqeval để tính các chỉ số:
  - Precision
  - Recall
  - F1-score theo từng loại thực thể

**Kết quả**
**Hiệu suất theo từng entity:**
- LOC - Precision: 0.42, Recall: 0.80, F1: 0.55  
- MISC - Precision: 0.50, Recall: 0.63, F1: 0.56  
- ORG - Precision: 0.47, Recall: 0.58, F1: 0.52  
- PER - Precision: 0.73, Recall: 0.60, F1: 0.66  

**Tổng quan:**
- Accuracy: 0.9129  
- Precision (micro): 0.5049  
- Recall (micro): 0.6617  
- F1-score (micro): 0.5728  

**Dự đoán câu mới:**
“VNU University is located in Hanoi”  
=> [('VNU','B-ORG'), ('University','I-ORG'), ('is','O'), ('located','O'), ('in','O'), ('Hanoi','B-LOC')]

**Nhận xét**
- Accuracy cao nhưng F1-score thấp hơn => mô hình dự đoán boundary chưa tốt.
- PER dễ nhận diện nhất, MISC và ORG khó hơn.
- RNN hạn chế ngữ cảnh hai chiều => Bi-LSTM sẽ cho kết quả tốt hơn.

## **Khó khăn và Giải pháp**

**Khó khăn**
- Không tải được CoNLL-2003 bằng load_dataset do lỗi script.
- Padding sai sẽ làm RNN học sai.
- Accuracy cao nhưng F1 thấp => khó nhận diện thực thể phức tạp.
- Mô hình RNN không nắm được ngữ cảnh hai chiều.

**Giải pháp**
- Đọc file CoNLL thủ công từ thư mục data.
- Dùng pack_padded_sequence để bỏ padding trong RNN.
- Dùng seqeval để đánh giá đúng chuẩn entity-level.
- Đề xuất nâng cấp mô hình lên Bi-LSTM hoặc Bi-LSTM-CRF.

## **Hướng dẫn chạy code**

**Thực hiện theo 4 bước**
1. Cài thư viện:
   ```
   pip install torch
   pip install seqeval
   ```
2. Đặt dữ liệu vào:
   ```
   data/conll2003/train.txt
   data/conll2003/valid.txt
   data/conll2003/test.txt
   ```
3. Chạy tuần tự các cell theo 5 Task.
4. Dùng `predict_sentence()` để kiểm thử mô hình.

## **Kết luận**
- Pipeline hoàn chỉnh: tải dữ liệu => xử lý => huấn luyện => đánh giá => suy luận.
- Mô hình RNN đạt kết quả ổn với bài toán NER cơ bản.
- Nếu dùng Bi-LSTM hoặc CRF sẽ cải thiện F1-score đáng kể.

## **Tài liệu tham khảo**
1. PyTorch Team. (2024). PyTorch Documentation – RNN, LSTM, GRU Modules.
https://pytorch.org/docs/stable/nn.html

2. HuggingFace Datasets Team. (2024). CoNLL-2003 Named Entity Recognition Dataset.
https://huggingface.co/datasets/eriktks/conll2003/tree/main
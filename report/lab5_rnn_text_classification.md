# **Lab 5: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)**

Giang Nguyen Thi - 22001254 

2025-11-17
---

Toàn bộ code thực thi cho các mô hình được đặt tại:  
**`nlp-labs/notebook/lab5_rnns_text_classification.ipynb`**

---

## **Mục tiêu chung**

Phần thực nghiệm này nhằm:

- Tự tay xây dựng, huấn luyện và so sánh hiệu năng giữa các mô hình: 
  (1) TF-IDF + Logistic Regression  
  (2) Word2Vec trung bình + Dense  
  (3) Embedding Pre-trained + LSTM  
  (4) Embedding học từ đầu + LSTM  
- So sánh hiệu năng giữa 4 mô hình bằng F1-score (macro) và test loss.  
- Phân tích định tính để đánh giá khả năng hiểu ngữ cảnh của LSTM so với các mô hình cổ điển.

---

## **Task 0 - Thiết lập Môi trường và Tải Dữ liệu**

**Mục tiêu**
- Tải đúng 3 tập train/val/test.
- Chuẩn hóa nhãn intent về dạng số nguyên sử dụng `LabelEncoder`.

**Các bước thực hiện**
- Đọc dữ liệu từ thư mục `../data/hwu/`.
- Mỗi file gồm 2 cột: *text*, *intent*.
- Ánh xạ toàn bộ intent sang số bằng một bộ `LabelEncoder` duy nhất.
- Khởi tạo số lớp `num_classes`.

**Nhận xét**
- Dữ liệu HWU được tổ chức tốt, phù hợp cho tác vụ intent classification.
- Dùng chung một `LabelEncoder` cho toàn bộ các tập giúp đảm bảo nhãn nhất quán.

---

## **Task 1 - (Warm-up Ôn bài cũ) Pipeline TF-IDF + Logistic Regression**

**Mục tiêu**
- Thiết lập baseline cổ điển để có chuẩn so sánh với các mô hình LSTM.

**Các bước thực hiện**
- Chuyển văn bản thành vector TF-IDF (`max_features = 5000`).
- Huấn luyện Logistic Regression với `max_iter = 1000`.
- Đánh giá bằng F1-macro và test loss.

**Kết quả**
- **F1-macro:** 0.822567  
- **Test Loss:** 1.052858  

**Nhận xét**
- Đây là mô hình cho kết quả **tốt nhất** trong cả 4 pipeline.  
- TF-IDF rất hiệu quả với câu truy vấn ngắn, khi từ khóa thể hiện intent rất rõ ràng.  
- Dù không học được thứ tự từ, TF-IDF + LR vẫn nhận diện intent rất mạnh nhờ đặc trưng từ vựng.

---

## **Task 2 - (Warm-up Ôn bài cũ) Pipeline Word2Vec (Trung bình) + Dense Layer**

**Mục tiêu**
- Biểu diễn câu bằng trung bình vector từ Word2Vec.  
- So sánh với TF-IDF để thấy hạn chế của biểu diễn “không có ngữ cảnh”.

**Các bước thực hiện**
- Huấn luyện Word2Vec (`vector_size = 100`) trên tập train.
- Tính vector trung bình cho từng câu.
- Huấn luyện mô hình Dense đơn giản.
- Đánh giá bằng F1-macro và test loss.

**Kết quả**
- **F1-macro:** 0.796896  
- **Test Loss:** 0.724759  

**Nhận xét**
- Word2Vec trung bình cho kết quả khá tốt, gần tiệm cận TF-IDF.  
- Dù mất thứ tự từ và không học được ngữ cảnh, nhưng do câu intent ngắn và nhiều từ khóa đặc trưng nên mô hình vẫn hoạt động hiệu quả.  
- Embedding Word2Vec học từ tập nhỏ nên vẫn có hạn chế so với pre-trained lớn hơn.

---

## **Task 3 - Mô hình Nâng cao (Embedding Pre-trained + LSTM)**

**Mục tiêu**
- Khởi tạo embedding từ mô hình Word2Vec đã huấn luyện ở Task 2.  
- Dùng LSTM để học quan hệ chuỗi và ngữ cảnh.

**Các bước thực hiện**
- Tokenizer + padding (*max_len = 50*).  
- Dùng embedding matrix từ Word2Vec và nạp vào `Embedding(trainable=False)`.  
- Mô hình gồm: Embedding => LSTM(128) => Dense softmax.  
- Huấn luyện với EarlyStopping.

**Kết quả**
- **F1-macro:** 0.640867  
- **Test Loss:** 1.050412  

**Nhận xét**
- Mô hình đã học tốt hơn rất nhiều so với phiên bản cũ (F1 tăng mạnh).  
- Tuy nhiên vẫn kém TF-IDF và Word2Vec Avg.  
- Nguyên nhân:
  - Word2Vec tự train vẫn yếu để hỗ trợ LSTM.  
  - Embedding bị đóng băng nên không fine-tune được.  
  - LSTM cần nhiều dữ liệu hơn để học phụ thuộc dài.

---

## **Task 4 - Mô hình Nâng cao (Embedding học từ đầu + LSTM)**

**Mục tiêu**
- Cho LSTM tự học embedding từ đầu.

**Các bước thực hiện**
- Tokenizer + Padding.  
- Embedding(output_dim=100, trainable=True).  
- LSTM(128) + Dense softmax.  
- EarlyStopping.

**Kết quả**
- **F1-macro:** 0.000533  
- **Test Loss:** 4.128992  

**Nhận xét**
- Mô hình **collapse hoàn toàn**, gần như dự đoán 1 lớp.  
- Dataset nhỏ => embedding từ đầu bị overfit nặng.  
- LSTM học không hiệu quả khi dữ liệu quá nhỏ.

---

## **Task 5 - Tổng hợp kết quả và phân tích**

**Bảng tổng hợp định lượng**

| Model                         | F1-macro | Test Loss |
|------------------------------|----------|-----------|
| TF-IDF + Logistic Regression | 0.822567 | 1.052858 |
| Word2Vec Avg + Dense         | 0.796896 | 0.724759 |
| LSTM + Pretrained Embedding  | 0.640867 | 1.050412 |
| LSTM + Scratch Embedding     | 0.000533 | 4.128992 |

---

**Nhận xét chung**

- **TF-IDF + Logistic Regression** vẫn là mô hình tốt nhất trên dataset nhỏ và câu truy vấn ngắn.  
- **Word2Vec Average** đạt kết quả cao hơn mong đợi, gần tiệm cận TF-IDF.  
- **LSTM Pretrained** có tiến bộ rõ rệt nhưng vẫn chưa vượt mô hình cổ điển.  
- **LSTM Scratch** thất bại hoàn toàn.  
- => Với dataset nhỏ, **mô hình cổ điển dựa trên từ khóa vẫn hiệu quả hơn mô hình chuỗi**.

---

Trong phần này, ba câu truy vấn có cấu trúc phức tạp hoặc chứa phủ định được chọn để kiểm tra khả năng hiểu ngữ cảnh của các mô hình.  
Kết quả dự đoán bên dưới là **kết quả thực tế** chạy từ notebook.

## **Phân tích định tính một số câu khó**

Bảng dưới đây trình bày **kết quả dự đoán thực tế** từ 4 mô hình trên 3 câu có cấu trúc phức tạp hoặc chứa phủ định.

| Câu kiểm tra | Nhãn đúng | TF-IDF + LR | Word2Vec Avg + Dense | LSTM Pretrained | LSTM Scratch | Nhận xét |
|--------------|-----------|--------------|------------------------|------------------|---------------|----------|
| *“can you remind me to not call my mom”* | reminder_create | calendar_set | play_game | play_game | music_query | Không mô hình nào hiểu được phủ định “not call”. TF-IDF dự đoán nhầm theo từ khóa “calendar”. LSTM và W2V đoán ngẫu nhiên. |
| *“is it going to be sunny or rainy tomorrow”* | weather_query | weather_query | general_dontcare | takeaway_query | music_query | Chỉ TF-IDF dự đoán đúng nhờ từ “sunny/rainy”. Các mô hình embedding không học được từ vựng thời tiết. |
| *“find a flight from new york to london but not through paris”* | flight_search | general_negate | social_post | lists_remove | music_query | Không mô hình nào hiểu cấu trúc “but not through”. TF-IDF bị nhiễu bởi từ “not”. LSTM hoàn toàn không nắm được ngữ cảnh. |

---

**Kết luận định tính**

- **TF-IDF + LR** cho kết quả tốt nhất, dù vẫn sai ở câu dài/phức tạp.  
- **Word2Vec Avg + Dense** dự đoán thiếu ổn định => mất ngữ cảnh, mất phủ định.  
- **LSTM Pretrained** không phát huy hiệu quả do Word2Vec quá yếu và embedding bị “đóng băng”.  
- **LSTM Scratch** bị **collapse** về 1 lớp (“music_query”) => overfitting rất nặng.

=> **Không mô hình nào thực sự hiểu được ngữ cảnh trong 3 câu khó.**  
=> LSTM không mạnh hơn TF-IDF trong trường hợp dataset nhỏ và embedding yếu.

---

## **Kết luận chung của phần thực nghiệm**

- TF-IDF + Logistic Regression hiện là lựa chọn tốt nhất cho dataset nhỏ, câu ngắn và phân loại intent.  
- Word2Vec trung bình và LSTM đều thể hiện kém khi dữ liệu hạn chế.  
- Kỹ thuật học chuỗi chỉ phát huy khi:
  - Dataset lớn.  
  - Embedding mạnh (GloVe, FastText, BERT…).  
- Thực nghiệm cho thấy tầm quan trọng của **đủ dữ liệu** và **embedding chất lượng cao** khi dùng LSTM.

---

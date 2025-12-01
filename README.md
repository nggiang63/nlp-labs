# Natural Language Processing And Deep Learning Labs

## 1. Thông tin sinh viên

- Họ và tên: Nguyễn Thị Giang  
- Mã sinh viên: 22001254  
- Lớp: K67A5 - Khoa học Dữ liệu  
- Học phần: Xử lý ngôn ngữ tự nhiên và Học sâu



## 2. Mục tiêu repository

Repository này lưu trữ toàn bộ mã nguồn, notebook và báo cáo cho các Lab môn NLP&DL.  
Mục tiêu chính:

- Chuẩn hóa cấu trúc repo theo hướng dẫn của học phần (tách riêng `src/`, `report/`, `notebook/`, `test/`, `data/`, `README.md`, `.gitignore`)   
- Mỗi Lab có báo cáo chi tiết, giúp người xem hiểu rõ cách làm, thí nghiệm và kết quả.  
- Dễ dàng chạy lại, tái hiện các thực nghiệm (reproducible).  



## 3. Cấu trúc thư mục

```text
nlp-labs/
│
├── src/                  # Toàn bộ code chính
│   ├── core/             # Logic chính của từng Lab (model, training, evaluation, ...)
│   ├── preprocessing/    # Tiền xử lý văn bản: tokenizer, làm sạch dữ liệu, ...
│   ├── utils/            # Hàm tiện ích dùng chung
│   └── ...               
│
├── notebook/             # Bao gồm notebook của một số Lab
│   ├── lab1.ipynb
│   └── ...               
│
├── report/               # Báo cáo chi tiết cho từng Lab
│   ├── lab1.md
│   └── ...               
│
├── test/                 # Script / test cho các Lab
│   ├── test_lab1.py
│   ├── test_lab2.py
│   └── ...
│
├── requirements.txt      # Danh sách thư viện Python cần cài đặt
├── README.md             # Giới thiệu tổng quan repository 
└── .gitignore            # Loại bỏ file không cần track 
```

## 4. Cách cài đặt và chạy

### Cài đặt môi trường Python

```bash
pip install -r requirements.txt
```

### Chạy code cho từng Lab

Ví dụ:

```bash
python test/lab1.py
...
```

Hoặc chạy trực tiếp trong notebook:

```bash
notebook/lab5.ipynb
...
```

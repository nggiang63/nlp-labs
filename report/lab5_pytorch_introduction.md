# **Lab 5: Pytorch Introduction**
Giang Nguyen Thi - 22001254 
2025-11-17
---

# **1. Mục tiêu**

- Hiểu và thao tác với **Tensor** – cấu trúc dữ liệu quan trọng nhất của PyTorch.  
- Biết cách sử dụng **autograd** để tự động tính đạo hàm.  
- Làm quen với 2 lớp cơ bản của PyTorch:  
  - `nn.Linear`  
  - `nn.Embedding`  
- Biết cách xây dựng một mô hình neural network bằng cách kế thừa **nn.Module**.  
- Chuẩn bị nền tảng cho các lab tiếp theo: RNN, LSTM, mô hình phân loại văn bản.

---

# **2. Các bước thực hiện**

---

# **PHẦN 1 – TENSOR: KHÁM PHÁ CẤU TRÚC DỮ LIỆU CỐT LÕI**

## **Task 1.1 – Tạo Tensor**

 **Code**
```python
import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(x_data)
print(x_np)
print(x_ones)
print(x_rand)

print(x_rand.shape, x_rand.dtype, x_rand.device)
```

 **Kết quả (rút gọn)**
```
tensor([[1, 2],
        [3, 4]])
tensor([[1, 1],
        [1, 1]])
tensor([[0.3941, 0.2430],
        [0.6930, 0.3968]])
torch.Size([2, 2]) torch.float32 cpu
```

 **Nhận xét**
- Tensor có thể tạo từ list hoặc NumPy array.  
- `ones_like`, `rand_like` tạo tensor theo shape có sẵn.  
- Tất cả tensor mặc định lưu trên **CPU** – có thể chuyển sang GPU với `to('cuda')`.

---

## **Task 1.2 – Các phép toán trên Tensor**

 **Code**
```python
print(x_data + x_data)
print(x_data * 5)
print(x_data @ x_data.T)
```

 **Kết quả**
```
tensor([[2, 4],
        [6, 8]])

tensor([[ 5, 10],
        [15, 20]])

tensor([[ 5, 11],
        [11, 25]])
```

 **Nhận xét**
- PyTorch hỗ trợ toán tử vector/matrix như NumPy.  
- Toán tử `@` dùng cho nhân ma trận.

---

## **Task 1.3 – Indexing & Slicing**

 **Code**
```python
x_data[0]
x_data[:, 1]
x_data[1, 1]
```

 **Nhận xét**
- Cách truy cập phần tử giống NumPy.  
- Hỗ trợ slicing và indexing linh hoạt.

---

## **Task 1.4 – Thay đổi hình dạng Tensor**

 **Code**
```python
x = torch.rand(4,4)
reshaped = x.view(16,1)
print(reshaped)
```

 **Nhận xét**
- `.view()` và `.reshape()` dùng để thay đổi shape.  
- Số phần tử phải giữ nguyên.

---

# **PHẦN 2 – AUTOGRAD: TỰ ĐỘNG TÍNH ĐẠO HÀM**

## **Task 2.1 – Tính gradient của biểu thức**

 **Code**
```python
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 3

z.backward()
print(x.grad)   # dz/dx = 18
```

 **Kết quả**
```
tensor([18.])
```

 **Giải thích**
- \( y = x+2 \)  
- \( z = 3y^2 = 3(x+2)^2 \)  
- \( dz/dx = 6(x+2) \Rightarrow 18 \)

 **Câu hỏi: Nếu gọi z.backward() lần nữa?**

->  PyTorch sẽ báo lỗi:
```
RuntimeError: Trying to backward through the graph a second time...
```

**Lý do:** autograd giải phóng computational graph sau khi backward.  
Muốn backward nhiều lần:

```python
z.backward(retain_graph=True)
```

---

# **PHẦN 3 – Xây dựng mô hình đầu tiên bằng torch.nn**

## **Task 3.1 – Lớp nn.Linear**

 **Code**
```python
linear_layer = nn.Linear(5, 2)
input_tensor = torch.randn(3, 5)
output = linear_layer(input_tensor)
print(output)
```

 **Nhận xét**
- Linear layer thực hiện phép biến đổi:  
  \[
  y = xW^T + b
  \]
- Output có shape (batch_size, output_dim).

---

## **Task 3.2 – Lớp nn.Embedding**

 **Code**
```python
embedding_layer = nn.Embedding(10, 3)
input_idx = torch.LongTensor([1, 5, 0, 8])
emb = embedding_layer(input_idx)
print(emb)
```

 **Nhận xét**
- Embedding hoạt động giống **bảng tra cứu từ vựng**.  
- Rất quan trọng trong NLP.

---

## **Task 3.3 – Định nghĩa mô hình bằng nn.Module**

 **Code**
```python
class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, indices):
        embeds = self.embedding(indices)
        hidden = self.activation(self.linear(embeds))
        return self.output_layer(hidden)

model = MyFirstModel(100, 16, 8, 2)
input_data = torch.LongTensor([[1,2,5,9]])
output_data = model(input_data)
print(output_data)
```

 **Nhận xét**
- Mô hình gồm:  
  - Embedding  
  - Linear  
  - ReLU  
  - Output Linear  
- Đây là pipeline nền tảng cho mọi mô hình NLP dùng embedding + encoder.

---

# **3. Kết luận**

- Nắm vững **Tensor** và các thao tác cơ bản.  
- Hiểu cơ chế **tự động tính gradient**.  
- Làm quen với **nn.Linear**, **nn.Embedding**.  
- Tự định nghĩa mô hình bằng `nn.Module`.  

-> Đây là nền tảng quan trọng để học tiếp LSTM, RNN và mô hình phân loại văn bản ở các Lab sau.

---


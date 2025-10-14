from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

def main():
    # 1. Streams Data: Đọc dữ liệu từ file (giả sử có sẵn)
    data_path = "../data/UD_English-EWT/en_ewt-ud-train.txt"

    if os.path.exists(data_path):
        print(f"Đang đọc dữ liệu từ: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
    else:
        # Nếu chưa có dữ liệu, tạo corpus nhỏ để demo
        print("Không tìm thấy file dữ liệu, dùng corpus mẫu để demo.")
        corpus = [
            "king queen man woman prince princess",
            "paris france berlin germany tokyo japan",
            "apple banana fruit food",
            "dog cat animal",
            "the queen rules the country",
            "the king leads the kingdom",
            "computers and software are technology",
        ]

    # Tiền xử lý dữ liệu (tokenization)
    tokenized = [simple_preprocess(line) for line in corpus]

    # 2. Trains a Model: Huấn luyện mô hình Word2Vec
    print("Đang huấn luyện mô hình Word2Vec ...")
    model = Word2Vec(
        sentences=tokenized,
        vector_size=50,   # 100 chiều (đúng theo hướng dẫn PDF) nhưng kRAM không đủ nên để 50
        window=5,          # phạm vi ngữ cảnh
        min_count=1,       # bỏ qua từ xuất hiện quá ít
        sg=1,              # dùng Skip-gram
        workers=4,
        epochs=50
    )
    print("Huấn luyện xong!")

    # 3. Saves the Model:  Lưu mô hình
    os.makedirs("../results/lab4", exist_ok=True)
    model_path = "../results/lab4/word2vec_ewt.model"
    model.save(model_path)
    print(f"Đã lưu mô hình tại: {model_path}")

    # 4. Demonstrates Usage: I
    # Thử nghiệm: tìm từ tương tự
    print("\n--- Các từ gần 'king' ---")
    try:
        print(model.wv.most_similar("king", topn=5))
    except KeyError:
        print("Từ 'king' không có trong mô hình.")

    # Thử nghiệm phép suy luận ngữ nghĩa (analogy)
    print("\n--- Phép suy luận ngữ nghĩa ---")
    try:
        print("king - man + woman ≈ ?", model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
    except KeyError as e:
        print(f"Lỗi khi thử analogy: {e}")


if __name__ == "__main__":
    main()

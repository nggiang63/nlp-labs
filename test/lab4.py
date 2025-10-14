import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.representations.word_embedder import WordEmbedder

def main():
    # Tạo đối tượng WordEmbedder và tải model
    embedder = WordEmbedder("glove-wiki-gigaword-50")

    # Lấy vector của một từ
    print("\n--- Vector của 'king' ---")
    vector_king = embedder.get_vector("king")
    print("Kích thước vector:", vector_king.shape)
    print("10 phần tử đầu tiên:", vector_king[:10])

    # Tính độ tương đồng giữa các từ
    print("\n--- Độ tương đồng ---")
    print("king - queen:", embedder.get_similarity("king", "queen"))
    print("king - man  :", embedder.get_similarity("king", "man"))
    print("apple - banana:", embedder.get_similarity("apple", "banana"))

    # Tìm các từ tương tự
    print("\n--- Các từ gần 'computer' ---")
    similar_words = embedder.get_most_similar("computer", top_n=5)
    for word, score in similar_words:
        print(f"{word:>12s} : {score:.4f}")

    # Nhúng câu / văn bản
    print("\n--- Vector biểu diễn câu ---")
    sentence = "The queen rules the country."
    doc_vec = embedder.embed_document(sentence)
    print("Kích thước vector:", doc_vec.shape)
    print("10 phần tử đầu tiên:", doc_vec[:10])

    # Kiểm tra trường hợp từ không tồn tại (OOV)
    print("\n--- Kiểm tra từ OOV ---")
    print(embedder.get_vector("khongtontai"))


if __name__ == "__main__":
    main()

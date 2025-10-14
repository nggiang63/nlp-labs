import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.regex_tokenizer import RegexTokenizer
import gensim.downloader as api
import numpy as np


class WordEmbedder:
    """
    Lớp WordEmbedder dùng để làm việc với các mô hình embedding có sẵn (GloVe, Word2Vec, fastText).
    Cung cấp các hàm để lấy vector của từ, tính độ tương đồng, tìm từ gần nhất và nhúng cả câu/văn bản.
    """

    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        """Khởi tạo lớp và tải mô hình embedding."""
        try:
            print(f"Đang tải mô hình: {model_name} ...")
            self.model = api.load(model_name)
            print("Tải mô hình thành công.")
            self.vector_size = self.model.vector_size
        except Exception as e:
            print(f"Lỗi khi tải mô hình '{model_name}': {e}")
            self.model = None
            self.vector_size = 0

    def get_vector(self, word: str):
        """Trả về vector của một từ. Nếu từ không tồn tại, trả về vector 0."""
        if self.model is None:
            print("Mô hình chưa được tải.")
            return None

        try:
            return self.model[word]
        except KeyError:
            print(f"Từ '{word}' không có trong từ điển của mô hình.")
            return np.zeros(self.vector_size)

    def get_similarity(self, word1: str, word2: str):
        """Tính cosine similarity giữa hai từ. Trả về None nếu có lỗi."""
        if self.model is None:
            print("Mô hình chưa được tải.")
            return None

        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            print(f"Lỗi: {e}. Một trong hai từ không có trong từ điển.")
            return None
        except Exception as e:
            print(f"Lỗi không xác định khi tính similarity: {e}")
            return None

    def get_most_similar(self, word: str, top_n: int = 10):
        """Trả về danh sách các từ gần nhất với từ đầu vào."""
        if self.model is None:
            print("Mô hình chưa được tải.")
            return []

        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            print(f"Từ '{word}' không có trong từ điển của mô hình.")
            return []
        except Exception as e:
            print(f"Lỗi khi tìm các từ tương tự: {e}")
            return []

    def embed_document(self, document: str):
        """
        Trả về vector biểu diễn cho cả câu hoặc văn bản
        (bằng cách lấy trung bình các vector của các từ trong câu).
        """
        if self.model is None:
            print("Mô hình chưa được tải.")
            return None

        try:
            tokenizer = RegexTokenizer()
            words = tokenizer.tokenize(document)
            vectors = [self.model[w] for w in words if w in self.model.key_to_index]

            if not vectors:
                print("Không có từ nào trong văn bản nằm trong từ điển của mô hình.")
                return np.zeros(self.vector_size)

            return np.mean(vectors, axis=0)

        except Exception as e:
            print(f"Lỗi khi tính embedding cho văn bản: {e}")
            return np.zeros(self.vector_size)

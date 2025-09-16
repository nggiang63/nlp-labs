from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass
    
class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]): # Tạo vocab
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]: # Biến corpus thành ma trận số.  # corpus: tập hợp văn bản
        pass

    def fit_transform(self, corpus: list[str]) -> list[list[int]]: # Trả về ma trận (document-term matrix).
        self.fit(corpus)
        return self.transform(corpus)

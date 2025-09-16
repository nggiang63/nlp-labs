from src.core.interfaces import Vectorizer
from src.core.interfaces import Tokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}   # token -> index

    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def transform(self, documents: list[str]) -> list[list[int]]:
        vectors = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            vectors.append(vector)
        return vectors

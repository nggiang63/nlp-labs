from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = text.split()
        return tokens

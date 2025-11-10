import sys
import os

# Thêm đường dẫn tới thư mục gốc để import được src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def get_dataset_and_vectorizer():
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    # Fit để học vocabulary
    vectorizer.fit(texts)
    print("Vocabulary:", vectorizer.vocabulary_)

    # Transform để vector hóa văn bản thành đặc trưng số
    X = vectorizer.transform(texts)

    print("\nFeature matrix (Count Vectors):")
    for i, row in enumerate(X):
        print(f"Text {i}: {row}")

    return texts, labels, X


if __name__ == "__main__":
    get_dataset_and_vectorizer()

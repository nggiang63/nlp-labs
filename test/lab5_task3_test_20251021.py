import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from sklearn.model_selection import train_test_split
from src.lab5_20251021.task1_data_preparation import get_dataset_and_vectorizer
from src.models.text_classifier import TextClassifier
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

def evaluation():
    # Data task 1
    texts, labels, X = get_dataset_and_vectorizer()

    # Data sentiments.csv
    # data_path = "/home/giangnt/Downloads/NLP_DL/nlp-labs/data/sentiments.csv"
    # df = pd.read_csv(data_path)

    # if not {"text", "sentiment"}.issubset(df.columns):
    #     raise ValueError("CSV must contain 'text' and 'sentiment' columns")

    # print(f"Loaded {len(df)} rows from {data_path}")

    # df = df.dropna(subset=["text", "sentiment"])
    # df["label"] = df["sentiment"].apply(lambda x: 1 if int(x) == 1 else 0)

    # texts = df["text"].tolist()
    # labels = df["label"].tolist()

    # 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    classifier = TextClassifier(vectorizer)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    metrics = classifier.evaluate(y_test, y_pred)

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    print("\nSample Predictions:")
    for text, pred in zip(X_test[:5], y_pred[:5]):
        label = "Positive" if pred == 1 else "Negative"
        print(f"{text} -> {label}")


if __name__ == "__main__":
    evaluation()

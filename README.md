# Natural Language Processing (NLP) Labs

```
nlp-labs/
│
├── data/                     # dữ liệu gốc (UD_English-EWT, ...)
│
├── src/                      # source code chính (package "src")
│   ├── core/                 # interface, loader, utils
│   │   ├── interfaces.py
│   │   └── dataset_loaders.py
│   │
│   ├── preprocessing/        # các bước tiền xử lý
│   │   ├── __init__.py
│   │   ├── simple_tokenizer.py
│   │   └── regex_tokenizer.py
│   │
│   └── representations/      # cho Lab 2 CountVectorizer
│       └── count_vectorizer.py
│
├── labs/                     # các script chạy theo từng lab
│   ├── lab1_tokenization.py
│   ├── lab2_count_vectorization.py
│   └── ...
│
├── tests/                    # unit test (pytest, optional)
│   └── test_tokenizer.py
│
├── venv/                     # virtual env
├── requirements.txt
├── README.md
└── .gitignore
```

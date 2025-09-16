import os

def load_raw_text_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    return data

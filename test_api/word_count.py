DATA_PATH = "../chromadb_etl/data/Richard_csanaki_CV.txt"

def count_words(filepath: str) -> int:
    try:
        with open(filepath, 'r') as file:
            data = file.read()
            words = data.split()
            return len(words)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return 0

# Example usage:
word_count = count_words(DATA_PATH)
print(f"Number of words in '{DATA_PATH}': {word_count}")

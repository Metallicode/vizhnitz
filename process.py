import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

def calculate_gematria(word):
    """Calculates the standard Gematria value of a Hebrew word."""
    gematria_map = {
        'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
        'י': 10, 'כ': 20, 'ך': 20, 'ל': 30, 'מ': 40, 'ם': 40, 'נ': 50, 'ן': 50,
        'ס': 60, 'ע': 70, 'פ': 80, 'ף': 80, 'צ': 90, 'ץ': 90, 'ק': 100, 'ר': 200,
        'ש': 300, 'ת': 400
    }
    return sum(gematria_map.get(char, 0) for char in word)

def generate_semantic_dataset(input_filepath, output_filepath):
    print("Loading Torah text...")
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filepath}'.")
        return

    # Split the massive text block into a sequential list of words
    words = text.split()
    print(f"Total words to analyze: {len(words)}")

    # Word2Vec learns by reading "sentences". Since we removed punctuation, 
    # we will artificially chunk the Torah into sequential "windows" of 15 words.
    print("Preparing context windows...")
    sentences = [words[i:i+15] for i in range(0, len(words), 15)]

    # --- 1. Train the Word2Vec Model ---
    print("Training the Word2Vec NLP model (this may take a few seconds)...")
    # vector_size = 100 dimensions. 
    # min_count = 1 ensures we keep EVERY unique word, even if it only appears once.
    # window = 5 means it looks at 5 words before and after each word to learn context.
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Extract the vocabulary and the 100D vectors
    word_list = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in word_list])
    
    print(f"Successfully mapped {len(word_list)} unique words.")

    # --- 2. Run t-SNE Dimensionality Reduction ---
    print("Running t-SNE to squash 100 dimensions into 2D (X/Y coordinates)...")
    print("(This is computationally heavy and might take 1-2 minutes)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

    vectors_2d = tsne.fit_transform(vectors)

    # --- 3. Build the final JSON dataset ---
    print("Compiling final dataset with Gematria...")
    dataset = []
    
    for i, word in enumerate(word_list):
        # t-SNE outputs very small float values. We multiply by 20 to physically 
        # spread the clusters out more on your D3.js web canvas.
        x_coord = round(float(vectors_2d[i][0] * 20), 2)
        y_coord = round(float(vectors_2d[i][1] * 20), 2)
        
        dataset.append({
            "word": word,
            "gematria": calculate_gematria(word),
            "x": x_coord,
            "y": y_coord
        })

    # Sort alphabetically for a clean JSON file
    dataset = sorted(dataset, key=lambda d: d['word'])

    # Export to JSON
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(dataset, outfile, ensure_ascii=False, indent=4)

    print(f"\nSuccess! Fully semantic database saved to '{output_filepath}'.")

# --- Run the Script ---
input_file = 'fulltura.txt' # Make sure this is your continuous text file!
output_file = 'torah_semantic_database.json'

generate_semantic_dataset(input_file, output_file)

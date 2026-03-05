

# Torah Semantic Gematria Visualizer 🌌

## Overview

This project bridges ancient Jewish mysticism with modern Machine Learning. It takes the full Hebrew text of the Torah, calculates the Gematria (numerical value) of every unique word, and uses Natural Language Processing (NLP) to cluster words based on their contextual meaning. The result is an interactive, explorable "galaxy" of biblical Hebrew.

## Features

* **Automated Text Cleaning:** Strips HTML, verse numbers, punctuation, and Hebrew vowels (*Niqqud*) to extract pure Hebrew root words.
* **Semantic Clustering (Word2Vec + t-SNE):** AI reads the text to understand context, grouping related words (e.g., family names, tabernacle materials) into distinct visual "islands."
* **Gematria Engine:** Automatically calculates standard Hebrew numerical values including final (*Sofit*) letters.
* **Interactive D3.js Visualization:** A searchable point-cloud interface. Type a word, and the galaxy dims to highlight only the words that share its exact Gematria, mapped across their semantic neighborhoods.

## Prerequisites

To run the data-processing pipeline, you will need Python 3 installed, along with the following libraries:

```bash
pip install beautifulsoup4 gensim scikit-learn numpy

```

## The Data Pipeline (How to Use)

### Step 1: Clean the Raw Text

If your Torah text is copied from the web, it likely contains HTML tags and vowels. Run the text-cleaning script to extract pure Hebrew words:

* **Script:** `clean_torah.py` (Uses BeautifulSoup)
* **Input:** `raw_torah.txt`
* **Output:** `clean_torah.txt`

*(Note: If you have 5 separate books, use the `combine_files.py` script to merge them into a single `fulltura.txt` file before moving to Step 2).*

### Step 2: Generate the Machine Learning Database

This script reads the sequential clean text, trains a Word2Vec model on the context of the words, uses t-SNE to squash those dimensions into 2D (X and Y coordinates), and calculates the Gematria for every unique word.

* **Script:** `ml.py`
* **Input:** `clean_torah.txt` (or `fulltura.txt`)
* **Output:** `torah_semantic_database.json`

### Step 3: Explore the Galaxy

1. Open `gematria_viz.html` in any modern web browser.
2. Click **"Load Semantic JSON"** and upload your generated `torah_semantic_database.json` file.
3. **Hover** over the glowing blue dots to explore the semantic clusters and see the words/Gematria values.
4. **Search:** Type a Hebrew word (e.g., `מים`, `אהבה`) into the search box. The tool will calculate its Gematria, list all matching words in the sidebar, and light them up in gold across the galaxy map.

## Technologies Used

* **Python:** Text parsing and data generation.
* **BeautifulSoup 4:** HTML DOM parsing.
* **Gensim (Word2Vec):** Contextual word embeddings.
* **Scikit-Learn (t-SNE):** Dimensionality reduction for 2D plotting.
* **HTML/CSS/Vanilla JS:** Front-end interface.
* **D3.js:** High-performance point cloud rendering.

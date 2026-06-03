# ightweight-bm25-search-engine# Lightweight BM25 Search Engine 

A lightweight Information Retrieval engine built from scratch in Python. This project implements an inverted index and the probabilistic **BM25 scoring algorithm** to rank text documents based on query relevance, avoiding high-level abstractions to focus on algorithmic fundamentals.

##  Mathematical Foundation

The core of the search ranking relies on the Okapi BM25 formula. Given a query $Q$, containing keywords $q_1, ..., q_n$, the BM25 score of a document $D$ is computed as:

$$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$ is the term frequency in the document.
* $|D|$ is the length of the document in words.
* $\text{avgdl}$ is the average document length in the corpus.
* $k_1 = 1.2$ controls non-linear term frequency saturation.
* $b = 0.75$ controls document length normalization.

##  Features
* **Custom Inverted Index:** Efficient token mapping for fast retrieval.
* **Length Penalization:** Correctly scores dense documents higher than artificially long ones.
* **Zero External Dependencies:** Built purely with Python's standard library (`math`, `re`).

##  Usage

```python
from search_engine import BM25SearchEngine

# 1. Initialize the engine
engine = BM25SearchEngine(k1=1.2, b=0.75)

# 2. Fit your corpus (dictionary of {doc_id: text})
engine.fit(corpus)

# 3. Search
results = engine.search("your search query", top_n=5)

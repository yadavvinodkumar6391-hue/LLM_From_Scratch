#Building LLM_scratch
# LLM Data Preprocessing & Tokenization Engine

An end-to-end implementation of the foundational data-preprocessing and tokenization pipeline for Large Language Models (LLMs), built from scratch using Python and PyTorch. This project covers text cleaning, custom vocabulary tokenization, subword Byte-Pair Encoding (BPE), sliding-window context sampling, and token-positional embedding layers.

---

## 🚀 Live Demo & Notebook
* **Google Colab Notebook:** [View Colab Notebook](https://colab.research.google.com/drive/1qq08-UizGrHCTEgMzlh74BHhl6QkX2Ad?usp=sharing) *(Ensure your link sharing is set to "Anyone with the link can view")*

---

## 🛠️ Key Features & Implementation Details

1. **Custom Tokenization (`SimpleTokenizerV1` & `V2`)**
   * Built custom text tokenizers using regular expressions (`re`) to split text on whitespace and punctuation.
   * Managed vocabulary mapping, integer-to-string decoding, and introduced special fallback tokens (`<|unk|>`, `<|endoftext|>`) for out-of-vocabulary words.
2. **Subword Byte-Pair Encoding (BPE)**
   * Integrated OpenAI's `tiktoken` library (GPT-2 vocabulary size ~50,257) to efficiently handle complex subwords, rare words, and punctuation without data loss.
3. **Sliding-Window Data Pipeline**
   * Implemented a custom PyTorch `Dataset` (`GPTDatasetV1`) and `DataLoader` to convert raw text corpora into overlapping input-target vector pairs $(x, y)$ using configurable context window lengths and stride parameters.
4. **Token & Positional Embeddings**
   * Leveraged PyTorch (`nn.Embedding`) to map token IDs to dense vector spaces (e.g., 256 dimensions) and combined them with absolute positional encodings to retain sequence ordering information.

---

## 📁 Repository Structure
```text
├── llm-preprocessing-pipeline.ipynb   # Main Google Colab notebook containing code & visualizations
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
